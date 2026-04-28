from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

from csnn import set_sym_type
from models.complementarity_RNN import ComplementarityRNN
from models.furuta_pendulum import FurutaPendulum


class FurutaPendulumRNN:
    """MPC approximation with RNN for furuta pendulum system.
    
    This class sets up and solves an optimization problem to learn a neural network
    policy, using a RNN, for furuta pendulum control using multiple parallel shooting with collocation,
    where the control input is directly approximated by the neural network.
    """
    
    def __init__(
        self,
        hidden_sizes=[2, 6, 6, 1],
        batch_size=80,
        horizon=10,
        degree=3,
        beta=0.5,
        regularization=1e-4,
        seed=42,
        complementarity_constraints=True,
        model_dir=None,
    ):
        """Initialize the MPC approximation problem.
        
        Parameters
        ----------
        hidden_sizes : list of int
            Hidden layer sizes for the RNN. 
            If is a list, create multiple RNNs with specified sizes.
        batch_size : int
            Number of parallel trajectories to optimize.
        horizon : int
            MPC horizon length.
        degree : int
            Degree of collocation polynomials.
        beta : float
            Softplus activation beta parameter.
        regularization : float
            L2 regularization weight on network parameters.
        seed : int
            Random seed for reproducibility.
        model_dir : Path or str, optional
            Directory to save model parameters. Defaults to ./models.
        """
        # Store configuration
        self.hidden_sizes = hidden_sizes
        self.NX = 4  # state dimension
        self.NU = 1  # control dimension
        self.NB = batch_size
        self.N = horizon
        self.degree = degree
        self.beta = beta
        self.regularization = regularization
        self.seed = seed
        self.complementarity_constraints = complementarity_constraints
        
        # State bounds
        self.theta_1_bound = ca.inf
        self.omega_1_bound = ca.inf
        self.theta_2_bound = ca.inf
        self.omega_2_bound = ca.inf
        
        # Training bounds
        self.theta_1_train_bound = np.pi/2
        # self.omega_train_bound = alpha * self.omega_1_bound
        
        # Control bounds
        self.u_min = -10.0
        self.u_max = 10.0
        
        # Setup model directory
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent / "models_nn" / "rnn_furuta_pendulum"
        else:
            self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.get_model_name()
        
        # Initialize furuta pendulum system
        self.furuta_pend = FurutaPendulum(dt=0.05, sym_type='SX')
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Initialize storage for results
        self.X_train = np.zeros((self.NX, self.NB))
        self.solution = None
        self.optimal_params = None
        self.x_opt = None
        self.u_opt = None
        self.solver = None
        self.w0 = None
        self.lbw = None
        self.ubw = None
        self.lbg = None
        self.ubg = None
        
        # Setup network
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one RNN layer size")

        for size in self.hidden_sizes:
            if size <= 0:
                raise ValueError(f"Hidden layer sizes must be positive integers. Got {self.hidden_sizes}")
        
        if self.complementarity_constraints:
            print("Setting up complementarity RNN...")
            self.setup_complementarity_network()
        else:
            print("Setting up standard RNN...")
            self.setup_network()
            
        # Generate initial states for training
        self.generate_initial_states(generate_informative=True)
        
        
    def get_model_name(self):
        """Generate model name from hidden sizes."""
        return 'x'.join(map(str, self.hidden_sizes))
    
    def setup_network(self):
        """Set up the RNN approximator."""
        set_sym_type("SX")

        x = ca.SX.sym('x', self.NX)
        x_seq = ca.repmat(x, 1, self.N)

        hidden_seq = x_seq
        output_seq = None

        rnn = ComplementarityRNN(
            input_size=self.NX,
            hidden_size=self.hidden_sizes,
            output_size=self.NU,
            complementarity=False,
            output_bias=True,
        )
        # Build the RNN
        h0 = np.zeros((self.hidden_sizes[0], 1))
        result = rnn.build(hidden_seq, h0)
        
        # Get parameters from build result
        self.params_flattened = result["params_flat"]
        self.n_param = rnn.n_params
        
        output_seq = result["output"]
        hidden_seq = result["hidden"]

        print(f"Number of parameters in the RNN network: {self.n_param}")

        output_vec = ca.reshape(output_seq, self.NU * self.N, 1)
        self.net_fcn = ca.Function('net_fcn', [x, self.params_flattened], [output_vec])
        
    def setup_complementarity_network(self):
        """Set up the complementarity RNN.
        
        Hidden layers are modeled with variables a >= 0 and s >= 0 such that
        z = a - s and a_i * s_i = 0. The last layer is linear.
        """

        x = ca.SX.sym('x', self.NX)
        x_seq = ca.repmat(x, 1, self.N)

        hidden_seq = x_seq
        output_seq = None
        
        rnn = ComplementarityRNN(
            input_size=self.NX,
            hidden_size=self.hidden_sizes,
            output_size=self.NU,
            complementarity=True,
            output_bias=True,
        )
        # Build the RNN
        h0 = np.zeros((self.hidden_sizes[0], 1))
        result = rnn.build(hidden_seq, h0)
        
        # Get parameters and complementarity variables from build result
        self.params_flattened = result["params_flat"]
        self.cc_vars = result["vars"]
        self.cc_vars_lbw = result["lbw"]
        self.cc_vars_ubw = result["ubw"]
        self.cc_g = result["g"]
        self.cc_lbg = result["lbg"]
        self.cc_ubg = result["ubg"]
        self.n_param = rnn.n_params
        print(f"Number of parameters in the RNN network: {self.n_param}")
        
        output_seq = result["output"]
        hidden_seq = result["hidden"]

        output_vec = ca.reshape(output_seq, self.NU * self.N, 1)
        self.net_fcn = ca.Function('net_fcn', [x, self.params_flattened, *self.cc_vars], [output_vec])
        self.cc_fcn = ca.Function('cc_fcn', [x, self.params_flattened, *self.cc_vars], [ca.vertcat(*self.cc_g)])
    
    def initialize_parameters(self, params_file=None):
        """Initialize network parameters.
        
        Parameters
        ----------
        params_file : str or Path, optional
            Path to file with pre-trained parameters. If None, uses He initialization.
            
        Returns
        -------
        params_init : np.ndarray
            Initial parameter values.
        """
        if params_file is not None:
            try:
                params_init = self.load_params(params_file, self.n_param)
                print(f"Loaded pre-optimized parameters from {params_file}")
                input("Press Enter to continue with these parameters, or Ctrl+C to abort ")
                return params_init
            except (FileNotFoundError, ValueError) as e:
                print(f"Could not load parameters from {params_file}: {e}")
                print("Using Kaiming (He) initialization instead.")
        
        print("Using Kaiming (He) initialization.")
        return self.parameter_initialization_he()
    
    def generate_initial_states(self, generate_informative=False):
        """Generate initial states for training.
        
        Parameters
        ----------
        generate_informative : bool
            If True, generates some initial states by solving open-loop MPC trajectories
            and some random states. 
            If False, only samples uniformly from the training bounds.
            
        split_train : float
            Fraction of the batch to use for informative initial states if generate_informative is True.
        """
        if generate_informative:
            # print("Generating informative initial states by solving open-loop MPC...")
            # Solve open-loop MPC for each initial state to generate initial guess trajectories.
            self.X_train = np.random.uniform(
                low=np.array([-self.theta_1_train_bound, 0, 0, 0]).reshape(-1, 1),
                high=np.array([self.theta_1_train_bound, 0, 0, 0]).reshape(-1, 1),
                size=(self.NX, self.NB//2)
            )
            informative_states = []
            for i in range(self.NB//2):
                x0_i = self.X_train[:, i]
                # print(f"Picking informative states: Solving open-loop MPC for initial state {i + 1}/{self.NB//2}, x0={x0_i}")
                _, x_traj, u_traj = self.furuta_pend.solve_MPC(x0_i, return_traj=True)
                # Pick the state at the middle of the trajectory as an informative initial state, 
                # since the start and end states are often not informative (start is always the same, end is always near the target).
                idx = np.random.randint(1, self.N - 1)
                x_mid = np.array(x_traj[:, idx]).reshape(-1)
                # print(f"Selected informative state from trajectory at time step {idx}: {x_mid}")
                informative_states.append(x_mid)
            self.X_train = np.array(informative_states).T
            # Generate the rest of the initial states randomly
            num_random = self.NB - self.NB//2
            # print(f"Generated {len(informative_states)} informative initial states. Generating {num_random} random initial states.")
            if num_random > 0:
                X_random = np.random.uniform(
                    low=np.array([-self.theta_1_train_bound, 0, 0, 0]).reshape(-1, 1),
                    high=np.array([self.theta_1_train_bound, 0, 0, 0]).reshape(-1, 1),
                    size=(self.NX, num_random)
                )
                self.X_train = np.hstack((self.X_train, X_random))
                
        else:
            self.X_train = np.random.uniform(
                low=np.array([-self.theta_1_train_bound, 0, 0, 0]).reshape(-1, 1),
                high=np.array([self.theta_1_train_bound, 0, 0, 0]).reshape(-1, 1),
                size=(self.NX, self.NB)
            )
        # print(self.X_train[:, :5])
        
        # print(f"Generated initial states for training. Sample states:\n{self.X_train[:, :5]}")
        # raise NotImplementedError("Initial state generation complete. Call setup_optimization() to set up the optimization problem.")
    
    def network_warm_start_with_sgd(self, params_init, learning_rate=1e-3, num_samples=20, iterations=500):
        """Warm start network parameters using simple SGD on a surrogate loss.
        
        Parameters
        ----------
        params_init : np.ndarray
            Initial parameters to warm start from.
        learning_rate : float
            Learning rate for SGD.
        num_samples : int
                Number of training samples to use for the surrogate loss.
        iterations : int
            Number of SGD iterations.
            
        Returns
        -------
        params_warm : np.ndarray
            Warm-started parameters after SGD.
        """
        print("Starting network warm start with SGD...")
        
        # Extarct the first num_samples training data for surrogate loss
        X_train = self.X_train[:, :num_samples]
        
        # Rollout the MPC 
        U_target = []
        for i in range(num_samples):
            u_opt = self.furuta_pend.solve_MPC(X_train[:, i], ret_seq=True)
            U_target.append(u_opt)
        
        U_target = np.array(U_target).reshape(self.NU * self.N, num_samples)
        
        # Forward pass
        F = self.net_fcn.map(num_samples)
        if self.complementarity_constraints:
            U_pred = F(X_train, self.params_flattened, *([0.0] * len(self.cc_vars)))
        else:
            U_pred = F(X_train, self.params_flattened)
        
        # Compute loss (MSE) and its gradient
        loss = ca.sqrt(ca.sumsqr(U_pred - U_target)) / num_samples + self.regularization * ca.dot(self.params_flattened, self.params_flattened)
        loss_fcn = ca.Function('loss_fcn', [self.params_flattened], [loss])
        grad = ca.gradient(loss, self.params_flattened)
        grad_fcn = ca.Function('grad_fcn', [self.params_flattened], [grad])
        
        # SGD loop
        params = params_init.copy()
        for it in range(iterations):
            grad_val = grad_fcn(params.reshape(-1, 1)).full().flatten()
            params -= learning_rate * grad_val
            
            if (it + 1) % 50 == 0 or it == 0:
                if self.complementarity_constraints:
                    u_pred = F(X_train, params, *([0.0] * len(self.cc_vars)))
                else:
                    u_pred = F(X_train, params)
                loss_val = loss_fcn(params.reshape(-1, 1)).full().item()
                print(f"SGD iteration {it + 1}/{iterations}, Loss: {loss_val:.4f}, Grad Norm: {np.linalg.norm(grad_val):.4f}")
                print(f"Sample predictions: {u_pred.full().flatten()[:5]}, Targets: {U_target.flatten()[:5]}")
        
        print("Network warm start complete.")
        return params  
    
    def solve_closed_loop_MPC_for_initial_states(self):
    #     """Solve closed-loop MPC for each initial state to generate initial guess trajectories. """
        # Rebuild the internal MPC with the same horizon used in this training problem.
        self.furuta_pend.define_simple_MPC_control(N=self.N)
        self.initial_trajectories = np.zeros((self.NX, self.N + 1, self.NB))
        self.initial_controls = np.zeros((self.NU, self.N, self.NB))
        for i in range(self.NB):
            # Simulate forward in time
            x0_i = self.X_train[:, i]
            # print(f"Warm-start closed-loop rollout {i + 1}/{self.NB}, x0={x0_i}")
            x_traj, u_traj, _ = self.furuta_pend.close_loop_simulation(
                x0_i,
                Nsim=self.N,
                plot_results=False,
                # on_mpc_failure='skip',
                warm_start=True
            )
            # Store trajectories
            # Check if 
            self.initial_trajectories[:, :, i] = x_traj.T
            self.initial_controls[:, :, i] = u_traj.T
            
    def solve_open_loop_MPC_for_initial_states(self):
        """Solve open-loop MPC for each initial state to generate initial guess trajectories. """
        self.initial_trajectories = np.zeros((self.NX, self.N + 1, self.NB))
        self.initial_controls = np.zeros((self.NU, self.N, self.NB))
        for i in range(self.NB):
            x0_i = self.X_train[:, i]
            print(f"Solving open-loop MPC for initial state {i + 1}/{self.NB}, x0={x0_i}")
            _, x_traj, u_traj = self.furuta_pend.solve_MPC(x0_i, return_traj=True)
            # Store trajectories
            self.initial_trajectories[:, :, i] = x_traj
            self.initial_controls[:, :, i] = u_traj

    def generate_state_warm_start(self, X0):
        """Generate state warm start by forward propagating with zero control.
        
        Parameters
        ----------
        X0 : np.ndarray
            Initial states for each batch element (NX x NB).
            
        Returns
        -------
        state_warm_start : dict
            Dictionary mapping (batch_idx, time_idx) -> state value for interval endpoints
            and (batch_idx, time_idx, colloc_idx) -> state value for collocation points.
        """
        print("Generating state warm start by forward propagating with zero control...")
        
        # Get collocation coefficients
        C_coeff, D_coeff, B_coeff, tau_root = self.get_collocation_coefficients(self.degree)
        
        # Create dynamics function for numerical integration
        state_warm_start = {}
        
        for i in range(self.NB):
            x_current = X0[:, i].reshape(-1, 1)
            
            # Store initial state
            state_warm_start[(i, 0)] = x_current.flatten()
            
            for k in range(self.N):
                # Zero control input
                u_zero = np.zeros((self.NU, 1))
                
                # Use RK4 integration over the interval with collocation points
                # First, compute states at collocation points using simple Euler steps
                for j in range(self.degree):
                    # Time fraction within interval
                    tau = tau_root[j + 1]
                    dt_frac = tau * self.furuta_pend.dt
                    
                    # Simple forward Euler from interval start
                    x_dot_val = self.furuta_pend.dynamics(x_current, u_zero).full()
                    x_colloc = x_current + dt_frac * x_dot_val
                    
                    state_warm_start[(i, k, j)] = x_colloc.flatten()
                
                # Compute state at end of interval using RK4
                x_next = self.rk4_step(x_current, u_zero, self.furuta_pend.dynamics, self.furuta_pend.dt)
                x_current = x_next
                
                # Store state at end of interval
                state_warm_start[(i, k + 1)] = x_current.flatten()
        
        print("State warm start generation complete.")
        return state_warm_start
    
    @staticmethod
    def rk4_step(x, u, f_dyn, dt):
        """Single RK4 integration step.
        
        Parameters
        ----------
        x : np.ndarray
            Current state.
        u : np.ndarray
            Control input.
        f_dyn : ca.Function
            Dynamics function.
        dt : float
            Time step.
            
        Returns
        -------
        x_next : np.ndarray
            Next state.
        """
        k1 = f_dyn(x, u).full()
        k2 = f_dyn(x + 0.5 * dt * k1, u).full()
        k3 = f_dyn(x + 0.5 * dt * k2, u).full()
        k4 = f_dyn(x + dt * k3, u).full()
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    def setup_optimization(self, params_init=None, warm_start='mpc'):
        """Set up the NLP optimization problem.
        
        Parameters
        ----------
        params_init : np.ndarray, optional
            Initial parameters. If None, will use He initialization.
        warm_start : string, optional
            Method for warm starting the optimization. Options are 'mpc' for using closed loop MPC trajectories, 
            'col' for using collocation-based state warm start, or None for no warm start.
        """
        print("Setting up optimization for learning problem...")
        
        # Initialize parameters
        if params_init is None:
            params_init = self.initialize_parameters()
        
        # Generate state warm start if requested
        state_warm_start = None
        control_warm_start = None
        if warm_start == 'col':
            state_warm_start = self.generate_state_warm_start(self.X_train)
        elif warm_start == 'mpc':
            # Generate initial trajectories by solving closed-loop MPC for each initial state
            # Warm starts both state and control actions
            self.solve_closed_loop_MPC_for_initial_states()
            state_warm_start = {}
            control_warm_start = {}
            for i in range(self.NB):
                for k in range(self.N + 1):
                    state_warm_start[(i, k)] = self.initial_trajectories[:, k, i]
                for k in range(self.N):
                    state_warm_start[(i, k, 0)] = self.initial_trajectories[:, k, i]  # colloc point 0
                    state_warm_start[(i, k, 1)] = (self.initial_trajectories[:, k, i] + self.initial_trajectories[:, k + 1, i]) / 2  # colloc point 1
                    state_warm_start[(i, k, 2)] = self.initial_trajectories[:, k + 1, i]  # colloc point 2
                    # Store control warm start
                    control_warm_start[(i, k)] = self.initial_controls[:, k, i]
                    
        # Get collocation coefficients
        C_coeff, D_coeff, B_coeff, tau_root = self.get_collocation_coefficients()
        
        # Define dynamics and stage cost
        x = ca.SX.sym('x', self.NX)
        u = ca.SX.sym('u', self.NU)
        x_dot = self.furuta_pend.dynamics(x, u)
        f = ca.Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])
        
        # Initialize NLP structures
        w = []      # decision variables
        w0 = []     # initial guess
        lbw = []    # lower bounds
        ubw = []    # upper bounds
        J = 0       # objective
        g = []      # constraints
        lbg = []    # lower bounds on constraints
        ubg = []    # upper bounds on constraints
        
        # Complementarity pair types for ccopt
        varcon = 1  # first index into x, second index into g
        ind_cc = []
        cctypes = []
        cc_var_sizes = []
        cc_first_constr_offsets = []
        if self.complementarity_constraints:
            if len(self.cc_vars) == 0:
                raise ValueError("Complementarity mode enabled but no complementarity variables were created.")

            # Each complementarity variable contributes one or more constraint blocks
            # in cc_fcn (e.g., only h-z, or [h-z, h*(h-z)]). Infer this from cc_g.
            if len(self.cc_g) % len(self.cc_vars) != 0:
                raise ValueError(
                    "Unexpected complementarity layout: number of constraints is not a multiple "
                    "of complementarity variables. Update ind_cc construction accordingly."
                )

            cc_blocks_per_var = len(self.cc_g) // len(self.cc_vars)
            cc_offset = 0
            for cc_var in self.cc_vars:
                cc_size = int(cc_var.size1() * cc_var.size2())
                cc_var_sizes.append(cc_size)
                # Pair with the first constraint block for this variable (h-z block).
                cc_first_constr_offsets.append(cc_offset)
                cc_offset += cc_blocks_per_var * cc_size
        
        # Add consensus parameters as decision variables
        z = ca.SX.sym('z', self.n_param)
        w += [z]
        lbw += [-100.0] * self.n_param
        ubw += [100.0] * self.n_param
        w0 += params_init.tolist()
        
        # Define one NLP for each batch element
        for i in range(self.NB):
            x0 = self.X_train[:, i]
            x_target = np.array([np.pi, 0.0, np.pi, 0.0])  # Target state
            
            # Initial state
            Xk = ca.SX.sym('X_' + str(i) + '_0', self.NX)
            w += [Xk]
            lbw += x0.tolist()
            ubw += x0.tolist()
            
            # Add the contribution of the initial state to the cost
            J += self.furuta_pend.init_cost(Xk, x0)
            
            # Use warm start if available
            if state_warm_start is not None:
                w0 += state_warm_start[(i, 0)].tolist()
            else:
                w0 += x0.tolist()

            cc_step_vars = []
            cc_var_x_offsets = []
            if self.complementarity_constraints:
                for layer_idx, cc_var in enumerate(self.cc_vars):
                    nrow, ncol = cc_var.shape
                    y_var = ca.SX.sym(f"Y_{i}_{layer_idx}", nrow, ncol)
                    cc_step_vars.append(y_var)
                    cc_var_x_offsets.append(len(lbw))
                    w += [y_var]
                    lbw += [0.0] * (nrow * ncol)
                    ubw += [ca.inf] * (nrow * ncol)
                    w0 += [0.0] * (nrow * ncol)
        
            control_vec = []
            for k in range(self.N):
                # Control input
                u_k = ca.SX.sym('U_' + str(i) + '_' + str(k), self.NU)
                lbw += [self.u_min] * self.NU
                ubw += [self.u_max] * self.NU
                w += [u_k]
                control_vec.append(u_k)
                # Use control warm start if available
                if control_warm_start is not None:
                    w0 += control_warm_start[(i, k)].tolist()
                else:
                    w0 += [0.0] * self.NU

                
                # State at collocation points
                Xc = []
                for j in range(self.degree):
                    Xkj = ca.SX.sym('X_' + str(i) + '_' + str(k) + '_' + str(j), self.NX)
                    Xc.append(Xkj)
                    w += [Xkj]
                    lbw += [-self.theta_1_bound, -self.omega_1_bound, -self.theta_2_bound, -self.omega_2_bound]
                    ubw += [self.theta_1_bound, self.omega_1_bound, self.theta_2_bound, self.omega_2_bound]
                    # Use warm start if available
                    if state_warm_start is not None:
                        w0 += state_warm_start[(i, k, j)].tolist()
                    else:
                        # Fallback: linear interpolation for collocation points
                        alpha = (k + tau_root[j+1]) / self.N
                        x_warm = (1 - alpha) * x0 + alpha * x_target
                        w0 += x_warm.tolist()
                
                # Loop over collocation points
                Xk_end = D_coeff[0] * Xk
                for j in range(1, self.degree + 1):
                    # Expression for the state derivative at the collocation point
                    xp = C_coeff[0, j] * Xk
                    for r in range(self.degree):
                        xp = xp + C_coeff[r + 1, j] * Xc[r]
                    
                    # Append collocation equations
                    fj = f(Xc[j - 1], u_k)
                    g += [self.furuta_pend.dt * fj - xp]
                    lbg += [0.0] * self.NX
                    ubg += [0.0] * self.NX
                    
                    # Add contribution to the end state
                    Xk_end = Xk_end + D_coeff[j] * Xc[j - 1]
                    
                # Add contribution of stage cost
                J += self.furuta_pend.stage_cost(Xk, u_k)
                
                # Next state
                Xk = ca.SX.sym('X_' + str(i) + '_' + str(k + 1), self.NX)
                w += [Xk]
                lbw += [-self.theta_1_bound, -self.omega_1_bound, -self.theta_2_bound, -self.omega_2_bound]
                ubw += [self.theta_1_bound, self.omega_1_bound, self.theta_2_bound, self.omega_2_bound]
                # Use warm start if available
                if state_warm_start is not None:
                    w0 += state_warm_start[(i, k + 1)].tolist()
                else:
                    # Fallback: linear interpolation for interval end states
                    alpha = (k + 1) / self.N
                    x_warm = (1 - alpha) * x0 + alpha * x_target
                    w0 += x_warm.tolist()
                
                # Add dynamics constraint
                g += [Xk_end - Xk]
                lbg += [0.0] * self.NX
                ubg += [0.0] * self.NX
                
            # Add constraint on control input as function approximator
            u_i = ca.vertcat(*control_vec)
            if self.complementarity_constraints:
                g += [u_i - self.net_fcn(x0, z, *cc_step_vars)]
                lbg += [0.0] * self.NU * self.N 
                ubg += [0.0] * self.NU * self.N
                
                # Pair each Y component with its corresponding (h-z) constraint.
                # ccopt expects one-indexed indices for both x and g.
                cc_g_start = len(lbg)
                for layer_idx, cc_size in enumerate(cc_var_sizes):
                    x_start = cc_var_x_offsets[layer_idx]
                    g_start = cc_g_start + cc_first_constr_offsets[layer_idx]
                    for elem_idx in range(cc_size):
                        ind_cc.append([x_start + elem_idx + 1, g_start + elem_idx + 1])
                        cctypes.append(varcon)
                    
                # Add complementarity constraints for this step
                g += [self.cc_fcn(x0, z, *cc_step_vars)]
                lbg += self.cc_lbg
                ubg += self.cc_ubg
            
            else:
                g += [u_i - self.net_fcn(x0, z)]
                lbg += [0.0] * self.NU * self.N 
                ubg += [0.0] * self.NU * self.N
            
            # Add terminal cost
            J += self.furuta_pend.terminal_cost(Xk)
        
        # Average cost over the batch
        # J *= 1.0 / self.NB
        
        # Add regularization on parameters
        J += self.regularization * ca.dot(z, z)
        
        # Store NLP structures
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        
        # print(w0)
        # raise NotImplementedError("NLP setup complete. Call solve() to solve the optimization problem.")
        
        # Create NLP problem
        nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        print("NLP problem created.")
        
        decision_variables_num = sum([var.size1() * var.size2() for var in w])
        constraints_num = sum([constr.size1() * constr.size2() for constr in g])
        print(f"Number of decision variables: {decision_variables_num}")
        print(f"Number of constraints: {constraints_num}")
        print(f"Number of complementarity pairs: {len(ind_cc)}")
        
        # Create the NLP solver
        print("Creating NLP solver...")
        start_time = time.time()
        opts = {
            "expand": True,
            # ccopt delegates NLP solves to MadNLP; bound_relax_factor=0.0 is
            # important for complementarity formulations.
            "ind_cc": ind_cc,
            "cctypes": cctypes,
            # "madnlp.print_level": 2,
            "madnlp.max_iter": 5000,
            "madnlp.tol": 1e-6,
            "madnlp.bound_relax_factor": 0.0,
            "madnlp.linear_solver": "Ma97Solver",
            "ccopt.relaxation_update.TYPE" : "RolloffRelaxationUpdate",
            "ccopt.relaxation_update.rolloff_point" : 1e-3,
            "ccopt.relaxation_update.rolloff_slope" : 1.5,
            "ccopt.relaxation_update.sigma_min" : 1e-6,
        }
        self.solver = ca.nlpsol('solver', 'ccopt', nlp, opts)
        print(f"Solver creation time: {time.time() - start_time:.2f} seconds")
    
    def solve(self):
        """Solve the NLP optimization problem."""
        if self.solver is None:
            raise RuntimeError("Must call setup_optimization() before solve()")
        
        print("Solving NLP...")
        self.solution = self.solver(
            x0=self.w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        print("NLP solved.")
        print(self.solver.stats()['unified_return_status'])
        # Extract solution
        self.extract_solution()
    
    def extract_solution(self):
        """Extract optimal parameters and trajectories from the solution."""
        w_opt = self.solution['x'].full().flatten()
        
        # Extract consensus parameters (first n_param elements)
        self.optimal_params = w_opt[:self.n_param]
        
        # Storage
        self.x_opt = np.zeros((self.NB, self.N + 1, self.NX))
        self.u_opt = np.zeros((self.NB, self.N, self.NU))
        
        offset = self.n_param
        cc_step_size = 0
        if self.complementarity_constraints:
            cc_step_size = sum(int(var.size1() * var.size2()) for var in self.cc_vars)
        for i in range(self.NB):
            # Initial state X_i_0
            self.x_opt[i, 0, :] = w_opt[offset : offset + self.NX]
            offset += self.NX

            if cc_step_size:
                offset += cc_step_size
            
            # For each time step k
            for k in range(self.N):
                # Control U_i_k
                self.u_opt[i, k, :] = w_opt[offset : offset + self.NU]
                offset += self.NU

                # Skip collocation states
                offset += self.degree * self.NX
                
                # Next state X_i_(k+1)
                self.x_opt[i, k + 1, :] = w_opt[offset : offset + self.NX]
                offset += self.NX
        
        print(f"\nOptimal solution extraction complete:")
        print(f"  Max optimal parameter value: {np.max(np.abs(self.optimal_params)):.4f}")
        print(f"  Min optimal parameter value: {np.min(np.abs(self.optimal_params)):.4f}")
        print(f"  Extracted {self.NB} batch trajectories with {self.N+1} states and {self.N} controls each")
    
    def plot_results(self):
        """Plot the optimized trajectories."""
        if self.x_opt is None or self.u_opt is None:
            raise RuntimeError("Must call solve() before plot_results()")
        
        time_x = np.arange(self.N + 1)
        time_u = np.arange(self.N)
        
        plt.figure(figsize=(12, 10))
        
        # State: Theta 1
        plt.subplot(5, 1, 1)
        for i in range(self.NB):
            plt.plot(time_x, self.x_opt[i, :, 0], color='gray', alpha=0.25)
        mean_angle = self.x_opt[:, :, 0].mean(axis=0)
        plt.plot(time_x, mean_angle, 'g-', linewidth=2, label='mean angle')
        plt.ylabel('Angle (rad)')
        plt.grid()
        plt.legend()

        # State: Omega 1
        plt.subplot(5, 1, 2)
        for i in range(self.NB):
            plt.plot(time_x, self.x_opt[i, :, 1], color='gray', alpha=0.25)
        mean_omega = self.x_opt[:, :, 1].mean(axis=0)
        plt.plot(time_x, mean_omega, 'c-', linewidth=2, label='mean angular velocity')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.grid()
        plt.legend()
        
        # State: Theta 2
        plt.subplot(5, 1, 3)
        for i in range(self.NB):
            plt.plot(time_x, self.x_opt[i, :, 2], color='gray', alpha=0.25)
        mean_angle2 = self.x_opt[:, :, 2].mean(axis=0)
        plt.plot(time_x, mean_angle2, 'm-', linewidth=2, label='mean angle 2')
        plt.ylabel('Angle 2 (rad)')
        plt.grid()
        plt.legend()
        
        # State: Omega 2
        plt.subplot(5, 1, 4)
        for i in range(self.NB):
            plt.plot(time_x, self.x_opt[i, :, 3], color='gray', alpha=0.25)
        mean_omega2 = self.x_opt[:, :, 3].mean(axis=0)
        plt.plot(time_x, mean_omega2, 'y-', linewidth=2, label='mean angular velocity 2')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity 2 (rad/s)')
        plt.grid()
        plt.legend()

        # Control
        plt.subplot(5, 1, 5)
        for i in range(self.NB):
            plt.step(time_u, self.u_opt[i, :, 0], where='post', color='gray', alpha=0.25)
        mean_u = self.u_opt.mean(axis=0)
        plt.step(time_u, mean_u[:, 0], where='post', color='k', linewidth=2, label='mean control')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input')
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self):
        """Save optimal parameters to JSON and YAML files."""
        if self.optimal_params is None:
            raise RuntimeError("Must call solve() before save_results()")
        
        # Get current date for filename
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Prepare data for serialization
        if self.complementarity_constraints:
            params_dict = {
                "optimal_params": np.asarray(self.optimal_params).tolist(),
                "n_param": int(self.n_param),
                "hidden_sizes": self.hidden_sizes,
                "model_name": self.model_name,
                "batch_size": int(self.NB),
                "horizon": int(self.N),
                "date": date_str,
            }
        else:
            params_dict = {
                "optimal_params": np.asarray(self.optimal_params).tolist(),
                "n_param": int(self.n_param),
                "hidden_sizes": self.hidden_sizes,
                "model_name": self.model_name,
                "batch_size": int(self.NB),
                "beta": float(self.beta),
                "horizon": int(self.N),
                "date": date_str,
            }
        
        # Save YAML
        if self.complementarity_constraints:
            yaml_path = self.model_dir / f"optimal_params_fp_cc_{self.model_name}_{date_str}.yaml"
        else:
            yaml_path = self.model_dir / f"optimal_params_fp_{self.model_name}_{date_str}.yaml"
        with yaml_path.open("w") as f:
            yaml.safe_dump(params_dict, f)
        print(f"Saved parameters to {yaml_path}")
    
    @staticmethod
    def get_collocation_coefficients(d=3):
        """Compute collocation coefficients for a given degree d.
        
        Parameters
        ----------
        d : int
            Degree of collocation polynomials.
            
        Returns
        -------
        C : np.ndarray
            Coefficients of the collocation equation.
        D : np.ndarray
            Coefficients of the continuity equation.
        B : np.ndarray
            Coefficients of the quadrature function.
        """
        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
        
        # Coefficients of the collocation equation
        C = np.zeros((d + 1, d + 1))
        
        # Coefficients of the continuity equation
        D = np.zeros(d + 1)
        
        # Coefficients of the quadrature function
        B = np.zeros(d + 1)
        
        # Construct polynomial basis
        for j in range(d + 1):
            # Construct Lagrange polynomials
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
            
            # Evaluate the polynomial at the final time
            D[j] = p(1.0)
            
            # Evaluate the time derivative of the polynomial at all collocation points
            pder = np.polyder(p)
            for r in range(d + 1):
                C[j, r] = pder(tau_root[r])
            
            # Evaluate the integral of the polynomial
            pint = np.polyint(p)
            B[j] = pint(1.0)
        
        return C, D, B, tau_root
    
    @staticmethod
    def load_params(params_file, n_param):
        """Load optimal parameters from a file.
        
        Parameters
        ----------
        params_file : str or Path
            Path to the file containing the optimal parameters (JSON or YAML).
        n_param : int
            Expected number of parameters.
        
        Returns
        -------
        params_init : np.ndarray
            Flattened array of initialized parameters.
        """
        params_path = Path(params_file)
        if not params_path.is_file():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        with params_path.open("r") as f:
            if params_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif params_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError("Unsupported file format. Use JSON or YAML.")
        
        optimal_params = data.get("optimal_params")
        n_param_file = data.get("n_param")
        
        if optimal_params is None or n_param_file is None:
            raise ValueError("Invalid parameters file format. Expected keys: 'optimal_params', 'n_param'.")
        
        params_init_vec = np.array(optimal_params).flatten()
        # safety: if sizes mismatch, pad or truncate to n_param
        if params_init_vec.size < n_param:
            params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
        elif params_init_vec.size > n_param:
            params_init_vec = params_init_vec[:n_param]
        
        return params_init_vec
    
    def parameter_initialization_he(self):
        """Kaiming (He) initialization for neural network parameters.
        
        Returns
        -------
        params_init : np.ndarray
            Flattened array of initialized parameters.
        """
        params_init = np.random.randn(self.n_param) * 0.01
        return params_init
    
    def find_latest_params(self, model_dir, model_name, extension="yaml"):
        """Find the most recent parameter file for a given model.
        
        Parameters
        ----------
        model_dir : Path
            Directory containing parameter files.
        model_name : str
            Model name (e.g., '4x6x6x1').
        extension : str
            File extension ('yaml' or 'json').
        
        Returns
        -------
        latest_file : Path or None
            Path to the most recent parameter file, or None if not found.
        """
        if self.complementarity_constraints:
            pattern = f"optimal_params_fp_cc_{model_name}_*.{extension}"
        else:
            pattern = f"optimal_params_fp_{model_name}_*.{extension}"
        files = list(model_dir.glob(pattern))
        if not files:
            return None
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return latest_file

def main():
    # Configure the problem
    mpc = FurutaPendulumRNN(
        hidden_sizes=[6, 6],
        batch_size=100,
        horizon=12,
        degree=3,
        regularization=1e-4,
        seed=42,
        complementarity_constraints=True,
    )

    # Initialize params with warm start
    # warm_params = mpc.network_warm_start_with_sgd(mpc.initialize_parameters(), num_samples=20)
    warm_params = None
    # if tau_k == tau_init:
    params_file = mpc.find_latest_params(mpc.model_dir, mpc.model_name, extension="yaml")
    # warm_params = mpc.initialize_parameters(params_file)

    # Setup and solve the optimization problem
    mpc.setup_optimization(warm_params, warm_start='mpc')
    mpc.solve()
    
    # Visualize results
    mpc.plot_results()
    
    # Save optimal parameters
    mpc.save_results()


if __name__ == "__main__":
    main()




