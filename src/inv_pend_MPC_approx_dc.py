from datetime import datetime
import json
import os
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

from csnn import set_sym_type, Linear, Sequential, ReLU, Softplus
from models.inverted_pendulum import InvertedPendulum


class InvertedPendulumMPCInputCollocation:
    """MPC approximation with input collocation for inverted pendulum system.
    
    This class sets up and solves an optimization problem to learn a neural network
    policy for inverted pendulum control using multiple parallel shooting with collocation,
    where the control input is directly approximated by the neural network.
    """
    
    def __init__(
        self,
        layer_sizes=[2, 6, 6, 1],
        batch_size=80,
        horizon=20,
        degree=3,
        beta=0.5,
        q_weights=[100, 1, 30, 1],
        r_weight=0.01,
        regularization=1e-4,
        seed=42,
        model_dir=None,
    ):
        """Initialize the MPC approximation problem.
        
        Parameters
        ----------
        layer_sizes : list of int
            Neural network layer sizes (input must be 4 for inverted pendulum state).
        batch_size : int
            Number of parallel trajectories to optimize.
        horizon : int
            MPC horizon length.
        degree : int
            Degree of collocation polynomials.
        beta : float
            Softplus activation beta parameter.
        q_weights : list of float
            State cost weights [position, velocity, angle, angular_velocity].
        r_weight : float
            Control cost weight.
        regularization : float
            L2 regularization weight on network parameters.
        seed : int
            Random seed for reproducibility.
        model_dir : Path or str, optional
            Directory to save model parameters. Defaults to ./models.
        """
        # Store configuration
        self.layer_sizes = layer_sizes
        self.NX = 2  # state dimension
        self.NU = 1  # control dimension
        self.NB = batch_size
        self.N = horizon
        self.degree = degree
        self.beta = beta
        self.regularization = regularization
        self.seed = seed
        
        # Cost weights
        self.Q = ca.diag(ca.DM(q_weights))
        self.R = ca.diag(ca.DM([r_weight]))
        
        # Reference trajectory
        self.xr = ca.DM([0.0, 0.0])
        self.ur = ca.DM([0.0])
        
        # Training bounds
        self.theta_train_bound = np.pi / 6
        self.omega_train_bound = 1.0
        
        # State bounds
        self.theta_bound = np.pi / 3
        self.omega_bound = 2.0
        
        # Control bounds
        self.u_min = -100.0
        self.u_max = 100.0
        
        # Setup model directory
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models_nn"
        else:
            self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.get_model_name()
        
        # Initialize inverted pendulum system
        self.inverted_pend = InvertedPendulum(sym_type='SX')
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Initialize storage for results
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
        self.setup_network()
        
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
                layers.append(Softplus(beta=self.beta))
        
        self.net = Sequential[ca.SX](tuple(layers))
        self.n_param = self.net.num_parameters
        print(f"Number of parameters in the NN: {self.n_param}")
        
        # Get flattened network parameters
        params = []
        for _, param in self.net.parameters():
            params.append(ca.reshape(param, -1, 1))
        self.params_flattened = ca.vertcat(*params)
        
        # Create a neural network function
        x = ca.SX.sym('x', self.NX)
        self.net_fcn = ca.Function('net_fcn', [x, self.params_flattened], [self.net(x.T)])
    
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
                return params_init
            except (FileNotFoundError, ValueError) as e:
                print(f"Could not load parameters from {params_file}: {e}")
                print("Using Kaiming (He) initialization instead.")
        
        print("Using Kaiming (He) initialization.")
        return self.parameter_initialization_he()
    
    def network_warm_start_with_sgd(self, params_init, learning_rate=1e-2, iterations=500):
        """Warm start network parameters using simple SGD on a surrogate loss.
        
        Parameters
        ----------
        params_init : np.ndarray
            Initial parameters to warm start from.
        learning_rate : float
            Learning rate for SGD.
        iterations : int
            Number of SGD iterations.
            
        Returns
        -------
        params_warm : np.ndarray
            Warm-started parameters after SGD.
        """
        print("Starting network warm start with SGD...")
        
        # Generate random training data for surrogate loss
        num_samples = 100
        X_train = np.random.uniform(
            low=[-self.theta_train_bound, -self.omega_train_bound],
            high=[self.theta_train_bound, self.omega_train_bound],
            size=(num_samples, self.NX)
        )
        
        # Rollout the MPC 
        U_target = []
        for i in range(num_samples):
            u_opt = self.inverted_pend.solve_MPC(X_train[i, :].T)
            U_target.append(u_opt)
        
        U_target = np.array(U_target).reshape(self.NU, num_samples)
        
        # Forward pass
        F = self.net_fcn.map(num_samples)
        U_pred = F(X_train.T, self.params_flattened)
        
        # Compute loss (MSE) and its gradient
        loss = ca.sqrt(ca.sumsqr(U_pred - U_target)) / num_samples
        loss_fcn = ca.Function('loss_fcn', [self.params_flattened], [loss])
        grad = ca.gradient(loss, self.params_flattened)
        grad_fcn = ca.Function('grad_fcn', [self.params_flattened], [grad])
        
        # SGD loop
        params = params_init.copy()
        for it in range(iterations):
            # X_train = np.random.uniform(
            #     low=[-self.theta_train_bound, -self.omega_train_bound],
            #     high=[self.theta_train_bound, self.omega_train_bound],
            #     size=(num_samples, self.NX)
            # )
            u_pred = F(X_train.T, params)
            
            grad_val = grad_fcn(params.reshape(-1, 1)).full().flatten()
            params -= learning_rate * grad_val
            
            if (it + 1) % 50 == 0 or it == 0:
                loss_val = loss_fcn(params.reshape(-1, 1)).full().item()
                print(f"SGD iteration {it + 1}/{iterations}, Loss: {loss_val:.4f}")
        
        print("Network warm start complete.")
        return params
    
    def setup_optimization(self, params_init=None):
        """Set up the NLP optimization problem.
        
        Parameters
        ----------
        params_init : np.ndarray, optional
            Initial parameters. If None, will use He initialization.
        """
        print("Setting up optimization for learning problem...")
        
        # Initialize parameters
        if params_init is None:
            params_init = self.initialize_parameters()
        
        # Get collocation coefficients
        C_coeff, D_coeff, B_coeff = self.get_collocation_coefficients()
        
        # Define dynamics and stage cost
        x = ca.SX.sym('x', self.NX)
        u = ca.SX.sym('u', self.NU)
        x_dot = self.inverted_pend.dynamics(x, u)
        l = (x - self.xr).T @ self.Q @ (x - self.xr) + (u - self.ur).T @ self.R @ (u - self.ur)
        f = ca.Function('f', [x, u], [x_dot, l], ['x', 'u'], ['x_dot', 'l'])
        
        # Compute LQR terminal cost matrix
        A, B = self.inverted_pend.lin_dyn(self.xr, self.ur)
        E = solve_continuous_are(A.full(), B.full(), self.Q.full(), self.R.full())
        E_dm = ca.DM(E)
        
        # Initialize NLP structures
        w = []      # decision variables
        w0 = []     # initial guess
        lbw = []    # lower bounds
        ubw = []    # upper bounds
        J = 0       # objective
        g = []      # constraints
        lbg = []    # lower bounds on constraints
        ubg = []    # upper bounds on constraints
        
        # Add consensus parameters as decision variables
        z = ca.SX.sym('z', self.n_param)
        w += [z]
        lbw += [-100.0] * self.n_param
        ubw += [100.0] * self.n_param
        w0 += params_init.tolist()
        
        # Define one NLP for each batch element
        for i in range(self.NB):
            # Generate random initial state for this batch element
            theta0 = np.random.uniform(-self.theta_train_bound, self.theta_train_bound)
            omega0 = np.random.uniform(-self.omega_train_bound, self.omega_train_bound)
            x0 = [theta0, omega0]
            
            # Initial state
            Xk = ca.SX.sym('X_' + str(i) + '_0', self.NX)
            w += [Xk]
            lbw += x0
            ubw += x0
            w0 += x0
            
            for k in range(self.N):
                # Control input
                u_k = ca.SX.sym('U_' + str(i) + '_' + str(k), self.NU)
                lbw += [self.u_min] * self.NU
                ubw += [self.u_max] * self.NU
                w += [u_k]
                w0 += [0.0] * self.NU
                
                # Add constraint on control input as function approximator
                g += [u_k - self.net_fcn(Xk, z)]
                lbg += [0.0] * self.NU
                ubg += [0.0] * self.NU
                
                # State at collocation points
                Xc = []
                for j in range(self.degree):
                    Xkj = ca.SX.sym('X_' + str(i) + '_' + str(k) + '_' + str(j), self.NX)
                    Xc.append(Xkj)
                    w += [Xkj]
                    lbw += [-self.theta_bound, -self.omega_bound]
                    ubw += [self.theta_bound, self.omega_bound]
                    w0 += [0.0] * self.NX
                
                # Loop over collocation points
                Xk_end = D_coeff[0] * Xk
                for j in range(1, self.degree + 1):
                    # Expression for the state derivative at the collocation point
                    xp = C_coeff[0, j] * Xk
                    for r in range(self.degree):
                        xp = xp + C_coeff[r + 1, j] * Xc[r]
                    
                    # Append collocation equations
                    fj, qj = f(Xc[j - 1], u_k)
                    g += [self.inverted_pend.dt * fj - xp]
                    lbg += [0.0] * self.NX
                    ubg += [0.0] * self.NX
                    
                    # Add contribution to the end state
                    Xk_end = Xk_end + D_coeff[j] * Xc[j - 1]
                    
                    # Add contribution to quadrature function
                    J += B_coeff[j] * qj * self.inverted_pend.dt
                
                # Next state
                Xk = ca.SX.sym('X_' + str(i) + '_' + str(k + 1), self.NX)
                w += [Xk]
                lbw += [-self.theta_bound, -self.omega_bound]
                ubw += [self.theta_bound, self.omega_bound]
                w0 += [0.0] * self.NX
                
                # Add dynamics constraint
                g += [Xk_end - Xk]
                lbg += [0.0] * self.NX
                ubg += [0.0] * self.NX
            
            # Add terminal cost
            J += (Xk - self.xr).T @ E_dm @ (Xk - self.xr)
        
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
        
        # Create NLP problem
        nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        print("NLP problem created.")
        
        decision_variables_num = sum([var.size1() * var.size2() for var in w])
        constraints_num = sum([constr.size1() * constr.size2() for constr in g])
        print(f"Number of decision variables: {decision_variables_num}")
        print(f"Number of constraints: {constraints_num}")
        
        # Create the NLP solver
        print("Creating NLP solver...")
        start_time = time.time()
        opts = {
            "expand": True,
            "ipopt": {
                "print_level": 5,
                "max_iter": 5000,
                "tol": 1e-6,
                "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
                "linear_solver": "ma86",
            }
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
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
            ubg=self.ubg
        )
        print("NLP solved.")
        
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
        for i in range(self.NB):
            # Initial state X_i_0
            self.x_opt[i, 0, :] = w_opt[offset : offset + self.NX]
            offset += self.NX
            
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
        print(f"  Consensus param (z) L2 norm: {np.linalg.norm(self.optimal_params):.4f}")
        print(f"  Extracted {self.NB} batch trajectories with {self.N+1} states and {self.N} controls each")
    
    def plot_results(self):
        """Plot the optimized trajectories."""
        if self.x_opt is None or self.u_opt is None:
            raise RuntimeError("Must call solve() before plot_results()")
        
        time_x = np.arange(self.N + 1)
        time_u = np.arange(self.N)
        
        plt.figure(figsize=(12, 10))
        
        # State 1: Angle
        plt.subplot(3, 1, 1)
        for i in range(self.NB):
            plt.plot(time_x, self.x_opt[i, :, 0], color='gray', alpha=0.25)
        mean_x1 = self.x_opt[:, :, 0].mean(axis=0)
        plt.plot(time_x, mean_x1, 'g-', linewidth=2, label='mean angle')
        plt.ylabel('Angle (rad)')
        plt.grid()
        plt.legend()
        
        # State 4: Angular Velocity
        plt.subplot(3, 1, 2)
        for i in range(self.NB):
            plt.plot(time_x, self.x_opt[i, :, 1], color='gray', alpha=0.25)
        mean_x4 = self.x_opt[:, :, 1].mean(axis=0)
        plt.plot(time_x, mean_x4, 'c-', linewidth=2, label='mean angular velocity')
        plt.ylabel('Angular Velocity')
        plt.grid()
        plt.legend()
        
        # Control
        plt.subplot(3, 1, 3)
        for i in range(self.NB):
            plt.step(time_u, self.u_opt[i, :], where='post', color='gray', alpha=0.25)
        mean_u = self.u_opt.mean(axis=0)
        plt.step(time_u, mean_u, where='post', color='k', linewidth=2, label='mean control')
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
        params_dict = {
            "optimal_params": np.asarray(self.optimal_params).tolist(),
            "n_param": int(self.n_param),
            "layer_sizes": self.layer_sizes,
            "model_name": self.model_name,
            "batch_size": int(self.NB),
            "horizon": int(self.N),
            "date": date_str
        }
        
        # Save JSON
        json_path = self.model_dir / f"optimal_params_inv_dc_{self.model_name}_{date_str}.json"
        with json_path.open("w") as f:
            json.dump(params_dict, f, indent=2)
        print(f"Saved parameters to {json_path}")
        
        # Save YAML
        yaml_path = self.model_dir / f"optimal_params_inv_dc_{self.model_name}_{date_str}.yaml"
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
        
        return C, D, B
    
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
        params_init = []
        
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = np.sqrt(2.0 / float(fan_in))
            W = np.random.randn(fan_out, fan_in) * std  # weights
            b = np.zeros(fan_out)  # biases initialized to zero
            params_init.append(W.ravel())
            params_init.append(b.ravel())
        
        params_init_vec = np.concatenate(params_init)
        
        # safety: if sizes mismatch, pad or truncate to n_param
        if params_init_vec.size < self.n_param:
            params_init_vec = np.concatenate([params_init_vec, np.zeros(self.n_param - params_init_vec.size)])
        elif params_init_vec.size > self.n_param:
            params_init_vec = params_init_vec[:self.n_param]
        
        return params_init_vec
    
    @staticmethod
    def find_latest_params(model_dir, model_name, extension="yaml"):
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
        pattern = f"optimal_params_cp_input{model_name}_*.{extension}"
        files = list(model_dir.glob(pattern))
        if not files:
            return None
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return latest_file

def main():
    # Configure the problem
    mpc = InvertedPendulumMPCInputCollocation(
        layer_sizes=[2, 20, 1],
        batch_size=50,
        horizon=10,
        degree=3,
        beta=2.0,
        q_weights=[30, 1],
        r_weight=1.0,
        regularization=1e-4,
        seed=42
    )
    
    # Initialize params with warm start
    warm_params = mpc.network_warm_start_with_sgd(mpc.initialize_parameters())
    
    # params_file = mpc.find_latest_params(mpc.model_dir, mpc.model_name, extension="yaml")
    # warm_params = mpc.initialize_parameters(params_file)

    # Setup and solve the optimization problem
    mpc.setup_optimization(warm_params)
    mpc.solve()
    
    # Visualize results
    mpc.plot_results()
    
    # Save optimal parameters
    mpc.save_results()


if __name__ == "__main__":
    main()




