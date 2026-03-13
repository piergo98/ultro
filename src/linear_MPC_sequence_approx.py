from datetime import datetime
import json
import os
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

from csnn import set_sym_type, Linear, Sequential, ReLU, Softplus
from models.linear import LinearSystem


class LinearMPCSequenceApproximation:
    """MPC approximation with sequence output for linear discrete-time systems.
    
    This class sets up and solves an optimization problem to learn a neural network
    policy that outputs a full control sequence for a linear discrete-time system
    with complex conjugate pole pairs.
    """
    
    def __init__(
        self,
        A,
        B,
        layer_sizes=None,
        batch_size=100,
        horizon=15,
        Q=None,
        R=None,
        xr=None,
        ur=None,
        u_bounds=[1.0],
        state_bounds=[50.0, 50.0, 10.0, 10.0],
        alpha_train=1.0,
        regularization=1e-4,
        seed=42,
        model_dir=None,
    ):
        """Initialize the linear MPC sequence approximation problem.
        
        Parameters
        ----------
        A : np.ndarray or ca.DM
            State transition matrix (NX x NX).
        B : np.ndarray or ca.DM
            Input matrix (NX x NU).
        layer_sizes : list of int, optional
            Neural network layer sizes. First element must match state dimension,
            last element must match horizon length (number of control outputs).
            If None, uses [NX, 20, horizon].
        batch_size : int
            Number of parallel trajectories to optimize.
        horizon : int
            MPC horizon length.
        Q : np.ndarray or None
            State cost weight matrix. If None, uses identity matrix.
        R : np.ndarray or None
            Control cost weight matrix. If None, uses identity matrix.
        xr : np.ndarray or None
            Reference state. If None, uses zeros.
        ur : np.ndarray or None
            Reference control. If None, uses zeros.
        u_bounds : list of float
            Control input bounds, symmetric for each component.
        state_bounds : list of float
            State bounds, symmetric for each component.
        alpha_train : float
            Scaling factor for random initial state generation.
        regularization : float
            L2 regularization weight on network parameters.
        seed : int
            Random seed for reproducibility.
        model_dir : Path or str, optional
            Directory to save model parameters. Defaults to ./models.
        """
        # Store system matrices
        self.A = ca.DM(A)
        self.B = ca.DM(B)
        self.NX = self.A.size1()
        self.NU = self.B.size2()
        
        # Initialize the linear system model for closed-loop simulations
        self.sys = LinearSystem(A, B, dt=0.1)
        # Set default layer sizes if not provided
        if layer_sizes is None:
            layer_sizes = [self.NX, 20, horizon]
        
        # Store configuration
        self.layer_sizes = layer_sizes
        self.NB = batch_size
        self.N = horizon
        self.regularization = regularization
        self.seed = seed
        self.alpha_train = alpha_train
        self.u_min, self.u_max = -np.array(u_bounds), np.array(u_bounds)
        self.state_min, self.state_max = -np.array(state_bounds), np.array(state_bounds)
        self.train_bound = np.array(state_bounds) * self.alpha_train
        
        # Validate layer sizes
        if layer_sizes[0] != self.NX:
            raise ValueError(f"First layer size ({layer_sizes[0]}) must match state dimension ({self.NX})")
        if layer_sizes[-1] != self.N and layer_sizes[-1] != self.NU:
            raise ValueError(f"Last layer size ({layer_sizes[-1]}) must match horizon ({self.N}) or control dimension ({self.NU})")
        
        # Cost matrices
        if Q is None:
            self.Q = ca.DM.eye(self.NX)
        else:
            self.Q = ca.DM(Q)
        
        if R is None:
            self.R = ca.DM.eye(self.NU)
        else:
            self.R = ca.DM(R)
        
        # Reference trajectory
        if xr is None:
            self.xr = ca.DM.zeros(self.NX, 1)
        else:
            self.xr = ca.DM(xr)
        
        if ur is None:
            self.ur = ca.DM.zeros(self.NU, 1)
        else:
            self.ur = ca.DM(ur)
        
        # Compute terminal cost matrix using discrete-time algebraic Riccati equation
        self.E = ca.DM(solve_discrete_are(
            self.A.full(), self.B.full(), self.Q.full(), self.R.full()
        ))
        
        # Setup model directory
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent / "models_nn" / "linear_mpc"
        else:
            self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.get_model_name()
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Initialize storage for results
        self.solution = None
        self.optimal_params = None
        self.x_opt = None
        self.u_opt = None
        self.solver = None
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.lbg = []
        self.ubg = []
        self.g = []
        self.J = 0
        
        # Setup network
        self.setup_network()
        
        # Setup dynamics function
        self.setup_dynamics()
    
    def get_model_name(self):
        """Generate model name from layer sizes."""
        return 'x'.join(map(str, self.layer_sizes))
    
    def setup_network(self):
        """Set up the neural network approximator."""
        set_sym_type("SX")
        
        # Build network from layer_sizes
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:  # add activation for all but last layer
                layers.append(Softplus(beta=30.0))
                # layers.append(ReLU())
        
        self.net = Sequential[ca.SX](tuple(layers))
        self.n_param = int(self.net.num_parameters)
        print(f"Number of parameters in the NN: {self.n_param}")
        
        # Get flattened network parameters
        params = []
        for _, param in self.net.parameters():
            params.append(ca.reshape(param, -1, 1))
        self.params_flattened = ca.vertcat(*params)
        
        # Create a neural network function that outputs full sequence
        x = ca.SX.sym('x', self.NX)
        self.net_fcn = ca.Function('net_fcn', [x, self.params_flattened], [self.net(x.T).T])
    
    def setup_dynamics(self):
        """Set up the dynamics function."""
        x = ca.SX.sym('x', self.NX)
        u = ca.SX.sym('u', self.NU)
        
        # Next state
        x_next = self.A @ x + self.B @ u
        
        # Stage cost
        l = (x - self.xr).T @ self.Q @ (x - self.xr) + (u - self.ur).T @ self.R @ (u - self.ur)
        
        self.f_dyn = ca.Function('f_dyn', [x, u], [x_next, l], ['x', 'u'], ['x_next', 'l'])
        
    def network_warm_start_with_sgd(self, params_init, learning_rate=1e-3, num_samples=20, iterations=1000):
        """Warm start network parameters using simple SGD on a surrogate loss.
        
        Parameters
        ----------
        params_init : np.ndarray
            Initial parameters to warm start from.
        learning_rate : float
            Learning rate for SGD.
        num_samples : int
            Number of samples to use for training.
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
            u_opt = self.sys.solve_MPC(X_train[:, i], ret_seq=True)
            U_target.append(u_opt)
        
        U_target = np.array(U_target).reshape(self.NU * self.N, num_samples)
        
        # Forward pass
        F = self.net_fcn.map(num_samples)
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
                u_pred = F(X_train, params)
                loss_val = loss_fcn(params.reshape(-1, 1)).full().item()
                print(f"SGD iteration {it + 1}/{iterations}, Loss: {loss_val:.4f}, Grad Norm: {np.linalg.norm(grad_val):.4f}")
                # print(f"Sample predictions: {u_pred.full().flatten()[:5]}, Targets: {U_target.flatten()[:5]}")
        
        print("Network warm start complete.")
        return params
    
    def parameter_initialization_he(self):
        """Kaiming (He) initialization for neural network parameters.
        
        Returns
        -------
        params_init : np.ndarray
            Flattened array of initialized parameters.
        """
        params_init = []

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = np.sqrt(2.0 / float(fan_in))
            W = np.random.randn(fan_out, fan_in) * std  # weights
            b = np.zeros(fan_out)                        # biases initialized to zero
            params_init.append(W.ravel())
            params_init.append(b.ravel())
            
        params_init_vec = np.concatenate(params_init)
        
        # Safety: if sizes mismatch, pad or truncate to n_param
        if params_init_vec.size < self.n_param:
            params_init_vec = np.concatenate([
                params_init_vec, 
                np.zeros(self.n_param - params_init_vec.size)
            ])
        elif params_init_vec.size > self.n_param:
            params_init_vec = params_init_vec[:self.n_param]

        return params_init_vec
    
    def load_params(self, params_file):
        """Load optimal parameters from a file.

        Parameters
        ----------
        params_file : str or Path
            Path to the file containing the optimal parameters (JSON or YAML).

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
            raise ValueError(
                "Invalid parameters file format. Expected keys: 'optimal_params', 'n_param'."
            )
        
        params_init_vec = np.array(optimal_params).flatten()
        
        # Safety: if sizes mismatch, pad or truncate to n_param
        if params_init_vec.size < self.n_param:
            params_init_vec = np.concatenate([
                params_init_vec,
                np.zeros(self.n_param - params_init_vec.size)
            ])
        elif params_init_vec.size > self.n_param:
            params_init_vec = params_init_vec[:self.n_param]
            
        return params_init_vec
    
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
                params_init = self.load_params(params_file)
                print(f"Loaded pre-optimized parameters from {params_file}")
                return params_init
            except (FileNotFoundError, ValueError) as e:
                print(f"Could not load parameters from {params_file}: {e}")
                print("Using Kaiming (He) initialization instead.")
        
        print("Using Kaiming (He) initialization.")
        return self.parameter_initialization_he()
    
    def find_latest_params(self, extension="yaml"):
        """Find the most recent parameter file for this model.
        
        Parameters
        ----------
        extension : str
            File extension ('yaml' or 'json').
        
        Returns
        -------
        latest_file : Path or None
            Path to the most recent parameter file, or None if not found.
        """
        pattern = f"optimal_params_lin_{self.model_name}_*.{extension}"
        matching_files = sorted(self.model_dir.glob(pattern))
        if matching_files:
            return matching_files[-1]  # Return the last one (most recent alphabetically)
        return None
    
    def generate_intial_states(self):
        """Generate initial states for training."""
        self.X_train = np.random.uniform(
            low=-self.train_bound.reshape(-1, 1),
            high=self.train_bound.reshape(-1, 1),
            size=(self.NX, self.NB)
        )
    
    def solve_closed_loop_MPC_for_initial_states(self):
        """Solve closed-loop MPC for each initial state to generate initial guess trajectories. """
        self.initial_trajectories = np.zeros((self.NX, self.N + 1, self.NB))
        self.initial_controls = np.zeros((self.NU, self.N, self.NB))
        for i in range(self.NB):
            # Simulate forward in time
            x_traj, u_traj = self.sys.close_loop_simulation(self.X_train[:, i], Nsim=self.N, plot_results=False)
            # Store trajectories
            self.initial_trajectories[:, :, i] = x_traj.T
            self.initial_controls[:, :, i] = u_traj.T
    
    def setup_optimization(self, params_init=None, warm_start='mpc'):
        """Set up the optimization problem.
        
        Parameters
        ----------
        params_init : np.ndarray, optional
            Initial parameter values. If None, uses He initialization.
        """
        print("Setting up optimization problem...")
        start_time = time.time()
        
        # Initialize parameters if not provided
        if params_init is None:
            params_init = self.initialize_parameters()
            
        # Generate state warm start if requested
        state_warm_start = None
        control_warm_start = None
        if warm_start == 'mpc':
            # Generate initial trajectories by solving closed-loop MPC for each initial state
            # Warm starts both state and control actions
            self.solve_closed_loop_MPC_for_initial_states()
            state_warm_start = {}
            control_warm_start = {}
            for i in range(self.NB):
                state_warm_start[i] = self.initial_trajectories[:, :, i]
                control_warm_start[i] = self.initial_controls[:, :, i]
        
        # Reset decision variables and constraints
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.J = 0
        
        # Add parameters to decision variables
        z = ca.SX.sym('params', self.n_param)
        self.w += [z]
        self.lbw += [-50.0] * self.n_param
        self.ubw += [50.0] * self.n_param
        self.w0 += params_init.tolist()
        
        
        # Define one NLP for each batch element
        for i in range(self.NB):
            # Initial state for this batch element
            x0 = self.X_train[:, i].tolist()

            # Initialize state
            Xk = ca.SX.sym('X_' + str(i) + '_0', self.NX)
            self.w += [Xk]
            self.lbw += x0
            self.ubw += x0
            if state_warm_start is not None:
                self.w0 += state_warm_start[i][:, 0].tolist()
            else:
                self.w0 += x0
        
            # Iterate over horizon
            control_vec = []
            for k in range(self.N):
                # Extract k-th control from sequence
                u_k = ca.SX.sym('u_' + str(i) + '_' + str(k), self.NU)
                self.w += [u_k]
                self.lbw += self.u_min.tolist()
                self.ubw += self.u_max.tolist()
                if control_warm_start is not None:
                    self.w0 += control_warm_start[i][:, k].tolist()
                else:
                    self.w0 += [0.0] * self.NU  # Initial guess for control
                control_vec.append(u_k)
                
                # Compute next state
                Xk_next, l_k = self.f_dyn(Xk, u_k)
                
                # Add state to decision variables
                Xk = ca.SX.sym('X_' + str(i) + '_' + str(k+1), self.NX)
                self.w += [Xk]
                self.lbw += self.state_min.tolist()
                self.ubw += self.state_max.tolist()
                if state_warm_start is not None:
                    self.w0 += state_warm_start[i][:, k+1].tolist()
                else:
                    self.w0 += x0
                
                # Add dynamics constraint
                self.g += [Xk_next - Xk]
                self.lbg += [0.0] * self.NX
                self.ubg += [0.0] * self.NX
                         
                # Accumulate cost
                self.J += l_k
                
            # Add constraint on control input as function approximator
            u_i = ca.vertcat(*control_vec)
            self.g += [u_i - self.net_fcn(x0, z)]
            self.lbg += [0.0] * self.NU * self.N
            self.ubg += [0.0] * self.NU * self.N
            
            # Add terminal cost
            self.J += (Xk - self.xr).T @ self.E @ (Xk - self.xr)
        
        # Optional: add regularization term to the cost
        if self.regularization > 0:
            self.J += self.regularization * ca.dot(z, z)
        
        # Create NLP solver
        nlp = {'f': self.J, 'x': ca.vertcat(*self.w), 'g': ca.vertcat(*self.g)}
        
        decision_variables_num = sum([var.size1() * var.size2() for var in self.w])
        constraints_num = sum([constr.size1() * constr.size2() for constr in self.g])
        
        print(f"NLP problem created in {time.time() - start_time:.2f} seconds.")
        print(f"Number of decision variables: {decision_variables_num}")
        print(f"Number of constraints: {constraints_num}")
        print(f"Batch size: {self.NB}, Horizon: {self.N}")
        
        # Create the NLP solver
        opts = {
            "expand": True, 
            "ipopt": {
                "print_level": 5, 
                "max_iter": 5000, 
                "tol": 1e-6,
                "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
                "linear_solver": "ma86",
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-6,
                "warm_start_bound_frac": 1e-6,
            }
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def solve(self):
        """Solve the optimization problem."""
        if self.solver is None:
            raise RuntimeError("Must call setup_optimization() before solve()")
        
        print("\nSolving NLP...")
        start_time = time.time()
        
        self.solution = self.solver(
            x0=self.w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg
        )
        
        solve_time = time.time() - start_time
        print(f"NLP solved in {solve_time:.2f} seconds.")
        
        # Extract the optimal parameters
        w_opt = self.solution['x'].full().flatten()
        self.optimal_params = w_opt[:self.n_param]
        
        # Extract optimal state and control trajectories for all batches
        state_block_size = (self.N + 1) * self.NX
        self.x_opt = np.zeros((self.NB, self.N + 1, self.NX))
        self.u_opt = np.zeros((self.NB, self.N, self.NU))

        offset = self.n_param
        for i in range(self.NB):
            # Initial state X_i_0
            self.x_opt[i, 0, :] = w_opt[offset : offset + self.NX]
            offset += self.NX
            
            # For each time step k
            for k in range(self.N):
                # Control U_i_k
                self.u_opt[i, k, :] = w_opt[offset : offset + self.NU]
                offset += self.NU
                
                # Next state X_i_(k+1)
                self.x_opt[i, k + 1, :] = w_opt[offset : offset + self.NX]
                offset += self.NX
        
        print("Optimization complete.")
        return self.optimal_params
    
    def plot_results(self):
        """Plot the trajectories across the batch."""
        if self.x_opt is None or self.u_opt is None:
            raise RuntimeError("Must call solve() before plot_results()")
        
        time_x = np.arange(self.N + 1)
        time_u = np.arange(self.N)

        plt.figure(figsize=(12, 6))

        # States subplot
        plt.subplot(1, 2, 1)
        # Individual traces
        for i in range(self.NB):
            for j in range(self.NX):
                plt.plot(time_x, self.x_opt[i, :, j], color='gray', alpha=0.25)
        
        # Batch mean
        for j in range(self.NX):
            mean_xj = self.x_opt[:, :, j].mean(axis=0)
            plt.plot(time_x, mean_xj, linewidth=2, label=f'mean x{j+1}')
        
        plt.title('Batch State Trajectories (individual & mean)')
        plt.xlabel('Time Step')
        plt.ylabel('States')
        plt.grid()
        plt.legend()

        # Control subplot
        plt.subplot(1, 2, 2)
        for i in range(self.NB):
            plt.step(time_u, self.u_opt[i, :], where='post', color='gray', alpha=0.25)
        
        mean_u = self.u_opt.mean(axis=0)
        plt.step(time_u, mean_u, where='post', color='k', linewidth=2, label='mean u')
        plt.title('Batch Control Inputs (individual & mean)')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def save_results(self):
        """Save the optimal parameters to file."""
        if self.optimal_params is None:
            raise RuntimeError("Must call solve() before save_results()")
        
        # Get current date for filename
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Prepare data for serialization
        params_dict = {
            "optimal_params": np.asarray(self.optimal_params).tolist(),
            "n_param": int(self.n_param),
            "layer_sizes": self.layer_sizes,
            "model_name": self.model_name,
            "date": date_str,
            "batch_size": self.NB,
            "horizon": self.N,
            "regularization": self.regularization,
        }

        # Save YAML
        yaml_path = self.model_dir / f"optimal_params_lin_{self.model_name}_{date_str}.yaml"
        with yaml_path.open("w") as f:
            yaml.safe_dump(params_dict, f)
        print(f"Saved parameters to {yaml_path}")
    
    def test_policy_closed_loop(self, x0, params=None, Nsim=100, plot_results=True):
        """Test the trained neural network policy in closed loop.
        
        Parameters
        ----------
        x0 : array_like
            Initial state.
        params : np.ndarray, optional
            Network parameters to use. If None, uses self.optimal_params.
        Nsim : int, optional
            Number of simulation steps (default: 100).
        plot_results : bool, optional
            Whether to plot the results (default: True).
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (Nsim+1, NX).
        u_traj : np.ndarray
            Control trajectory of shape (Nsim, NU).
        """
        if params is None:
            if self.optimal_params is None:
                raise RuntimeError("No parameters available. Either provide params or call solve() first.")
            params = self.optimal_params
        
        # Initialize trajectories
        x_traj = np.zeros((Nsim + 1, self.NX))
        u_traj = np.zeros((Nsim, self.NU))
        
        x_traj[0] = np.array(x0).flatten()
        
        # Simulate forward in time using the LinearSystem's simulation method
        for k in range(Nsim):
            # Get control action from the neural network policy
            u_k = self.net_fcn(x_traj[k], params).full().flatten()[0:self.NU]
            
            # Store control action
            u_traj[k] = u_k
            
            # Compute next state using system dynamics
            x_next, _ = self.f_dyn(x_traj[k], u_k)
            x_traj[k + 1] = x_next.full().flatten()
            
        # Plot results
        time = np.arange(Nsim + 1) * self.sys.dt
        time_u = np.arange(Nsim) * self.sys.dt
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 10))
        title = "Cart-Pole Simulation"
        
        # Position
        axes[0].plot(time, x_traj[:, 1], 'b-', linewidth=2)
        axes[0].set_ylabel('Position (m)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        
        # Velocity
        axes[1].plot(time, x_traj[:, 3], 'r-', linewidth=2)
        axes[1].set_ylabel('Velocity (m/s)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Angle
        axes[2].plot(time, x_traj[:, 0], 'g-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (rad)', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Angular velocity
        axes[3].plot(time, x_traj[:, 2], 'c-', linewidth=2)
        axes[3].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Control
        axes[4].step(time_u, u_traj[:, 0], 'k-', where='post', linewidth=2)
        axes[4].set_ylabel('Control Force (N)', fontsize=10)
        axes[4].set_xlabel('Time (s)', fontsize=10)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj
    
    def test_policy_vs_mpc(self, x0, params=None, Nsim=100, plot_results=True):
        """Compare trained neural network policy against MPC in closed loop.
        
        Parameters
        ----------
        x0 : array_like
            Initial state.
        params : np.ndarray, optional
            Network parameters to use. If None, uses self.optimal_params.
        Nsim : int, optional
            Number of simulation steps (default: 100).
        plot_results : bool, optional
            Whether to plot the comparison results (default: True).
        
        Returns
        -------
        x_traj_nn : np.ndarray
            State trajectory with NN policy of shape (Nsim+1, NX).
        u_traj_nn : np.ndarray
            Control trajectory with NN policy of shape (Nsim, NU).
        x_traj_mpc : np.ndarray
            State trajectory with MPC of shape (Nsim+1, NX).
        u_traj_mpc : np.ndarray
            Control trajectory with MPC of shape (Nsim, NU).
        """
        if params is None:
            if self.optimal_params is None:
                raise RuntimeError("No parameters available. Either provide params or call solve() first.")
            params = self.optimal_params
        
        # Test neural network policy
        x_traj_nn, u_traj_nn = self.test_policy_closed_loop(
            x0, params=params, Nsim=Nsim, plot_results=False
        )
        
        # Test MPC controller
        x_traj_mpc, u_traj_mpc = self.sys.close_loop_simulation(
            x0, Nsim=Nsim, control_policy=None, plot_results=False
        )
        
        if plot_results:
            time = np.arange(Nsim + 1) * self.sys.dt
            time_u = np.arange(Nsim) * self.sys.dt
            
            fig, axes = plt.subplots(self.NX + 1, 1, figsize=(12, 3 * (self.NX + 1)))
            
            # Plot states
            for i in range(self.NX):
                axes[i].plot(time, x_traj_nn[:, i], 'b-', linewidth=2, label='NN Policy')
                axes[i].plot(time, x_traj_mpc[:, i], 'r--', linewidth=2, label='MPC')
                axes[i].set_ylabel(f'State x{i+1}', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                if i == 0:
                    axes[i].set_title('Neural Network Policy vs MPC', fontsize=12, fontweight='bold')
            
            # Plot control
            axes[self.NX].step(time_u, u_traj_nn[:, 0], 'b-', where='post', linewidth=2, label='NN Policy')
            axes[self.NX].step(time_u, u_traj_mpc[:, 0], 'r--', where='post', linewidth=2, label='MPC')
            axes[self.NX].set_ylabel('Control u', fontsize=10)
            axes[self.NX].set_xlabel('Time (s)', fontsize=10)
            axes[self.NX].grid(True, alpha=0.3)
            axes[self.NX].legend()
            
            plt.tight_layout()
            plt.show()
        
        return x_traj_nn, u_traj_nn, x_traj_mpc, u_traj_mpc


def main():
    """Main function demonstrating usage of the LinearMPCSequenceApproximation class."""
    A = [[1.0, 0.1, 0.0, 0.0],
         [0.0, 0.9818, 0.2673, 0.0],
         [0.0, 0.0, 1.0, 0.1],
         [0.0, -0.0455, 3.1182, 1.0]]
    
    B = [[0.0], [0.1818], [0.0], [0.4546]]   
    # Configure the problem
    mpc = LinearMPCSequenceApproximation(
        A=A,
        B=B,
        layer_sizes=[4, 20, 20, 10],
        batch_size=120,
        horizon=10,
        Q=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        R=[[1.0]],
        u_bounds=[1.0],
        state_bounds=[1, 1.5, 0.35, 1.0],
        alpha_train=0.3,
        regularization=1e-4,
        seed=42
    )
    # Generate random initial states for batch
    print(f"Generating {mpc.NB} random initial states...")
    mpc.generate_intial_states()
    
    # Initialize params with warm start
    # warm_params = mpc.network_warm_start_with_sgd(mpc.initialize_parameters(), num_samples=20)
    # warm_params = None
    
    # Load parameters from file
    params_file = mpc.find_latest_params()
    warm_params = mpc.initialize_parameters(params_file)
    
    # Setup and solve the optimization problem
    mpc.setup_optimization(warm_params, warm_start='mpc')
    mpc.solve()
    
    # Visualize results
    mpc.plot_results()
    
    # Save optimal parameters
    mpc.save_results()
    
    # Test the trained policy in closed loop
    print("\nTesting trained policy in closed loop...")
    # Sample test initial state from the bounded region
    np.random.seed(43)  # Different seed for test set
    x0_test = np.random.uniform(-mpc.train_bound, mpc.train_bound)
    print(f"Test initial state: {x0_test}")
    # x_traj, u_traj = mpc.test_policy_closed_loop(x0_test, Nsim=100, plot_results=True)
    
    # Compare NN policy vs MPC
    print("\nComparing NN policy vs MPC...")
    mpc.test_policy_vs_mpc(x0_test, params=warm_params, Nsim=100, plot_results=True)


if __name__ == "__main__":
    main()
