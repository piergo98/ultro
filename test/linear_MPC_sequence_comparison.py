from datetime import datetime
import json
from pathlib import Path
import time
import yaml

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are

try:
    from csnn import set_sym_type, Linear, Sequential, ReLU, Softplus
    HAS_CSNN = True
except ImportError:
    HAS_CSNN = False


def get_model_name(layer_sizes):
    """Generate model name from layer sizes.
    
    Parameters
    ----------
    layer_sizes : list of int
        List containing the sizes of each layer in the network.
    
    Returns
    -------
    model_name : str
        Model name in format like '4x20x1'
    """
    return 'x'.join(map(str, layer_sizes))


def find_latest_params(model_dir, model_name, extension="yaml"):
    """Find the most recent parameter file for a given model.
    
    Parameters
    ----------
    model_dir : Path
        Directory containing parameter files.
    model_name : str
        Model name (e.g., '4x20x1').
    extension : str
        File extension ('yaml' or 'json').
    
    Returns
    -------
    latest_file : Path or None
        Path to the most recent parameter file, or None if not found.
    """
    pattern = f"optimal_params_lin_{model_name}_*.{extension}"
    files = list(model_dir.glob(pattern))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file


def load_params(params_file, return_metadata=False):
    """Load optimal parameters from a file.

    Parameters
    ----------
    params_file : str or Path
        Path to the file containing the optimal parameters (JSON or YAML).
    return_metadata : bool
        If True, return a dict with all metadata. If False, return only parameters.

    Returns
    -------
    params_init : np.ndarray or dict
        If return_metadata is False: Flattened array of initialized parameters.
        If return_metadata is True: Dict with 'params', 'q_weights', 'r_weight', and other metadata.
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
    n_param = data.get("n_param")
    
    if optimal_params is None or n_param is None:
        raise ValueError("Invalid parameters file format. Expected keys: 'optimal_params', 'n_param'.")
    
    params_init_vec = np.array(optimal_params).flatten()
    # safety: if sizes mismatch, pad or truncate to n_param
    if params_init_vec.size < n_param:
        params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
    elif params_init_vec.size > n_param:
        params_init_vec = params_init_vec[:n_param]
    
    if return_metadata:
        return {
            'params': params_init_vec,
            'q_weights': data.get('q_weights'),
            'r_weight': data.get('r_weight'),
            'beta': data.get('beta'),
            'horizon': data.get('horizon'),
            'layer_sizes': data.get('layer_sizes'),
            'batch_size': data.get('batch_size')
        }
    else:
        return params_init_vec


# Class implementing a linear system with MPC control and optional neural network policy comparison
class LinearMPCComparison:
    
    def __init__(self, A, B, dt=0.1, N=10, Q=None, R=None, layer_sizes=None, beta=0.5, 
                 model_dir=None, wait_for_input=False):
        """Initialize the linear MPC system with optional neural network policy comparison.
        
        Parameters
        ----------
        A : np.ndarray
            State transition matrix of shape (nx, nx)
        B : np.ndarray
            Control input matrix of shape (nx, nu)
        dt : float, optional
            Time step (default: 0.1)
        N : int, optional
            MPC horizon length (default: 10)
        Q : np.ndarray, optional
            State cost weight matrix
        R : np.ndarray, optional
            Control cost weight matrix
        layer_sizes : list of int, optional
            Neural network architecture for learned policy
        beta : float, optional
            Softplus beta parameter for neural network
        model_dir : Path or None, optional
            Directory containing trained model parameters
        wait_for_input : bool, optional
            If True, wait for user input before starting testing
        """
        print("Setting up the linear MPC problem for testing...")
        start_time = time.time()
        
        # Check the instance of A and B
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        if not isinstance(B, np.ndarray):
            B = np.array(B)
        
        self.A = A
        self.B = B
        self.dt = dt
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
        # Set default Q and R if not provided
        if Q is None:
            Q = np.eye(self.nx)
        if R is None:
            R = np.eye(self.nu)
        
        self.Q = ca.DM(Q)
        self.R = ca.DM(R)
        
        # Store parameters
        self.N = N  # MPC horizon length
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.model_dir = model_dir if model_dir is not None else Path(__file__).parent.parent / "models_nn" / "linear_mpc"
        
        # Setup neural network if layer_sizes provided
        if layer_sizes is not None and HAS_CSNN:
            self._setup_network()
        else:
            self.net_fcn = None
        
        # Storage for results
        self.initial_states = None
        self.x_opt_batch = []
        self.u_opt_batch = []
        self.x_sim_batch = []
        self.u_sim_batch = []
        self.rmse_states_batch = []
        self.rmse_u_batch = []
        self.cost_opt_batch = []
        self.cost_sim_batch = []
        self.valid_indices = []
        
        if wait_for_input:
            input("Press Enter to start testing...")
        
        # Set up MPC problem
        self._setup_mpc()
        
        print(f"Setup completed in {time.time() - start_time:.2f} seconds")
    
    def _setup_network(self):
        """Set up the neural network approximator."""
        if not HAS_CSNN:
            raise ImportError("csnn module not found. Install it to use neural network policies.")
        
        set_sym_type("SX")
        
        # Build network from layer_sizes
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:  # add activation for all but last layer
                layers.append(Softplus(beta=self.beta))
        self.net = Sequential[ca.SX](tuple(layers))
        
        self.n_param = self.net.num_parameters
        
        # Get flattened network parameters
        params = []
        for _, param in self.net.parameters():
            params.append(ca.reshape(param, -1, 1))
        self.params_flattened = ca.vertcat(*params)
        
        # Create network function
        x = ca.SX.sym('x', self.nx)
        self.net_fcn = ca.Function('net_fcn', [x, self.params_flattened], [self.net(x.T)])
        
    def _build_integrator(self, f):
        """Build RK4 integrator for dynamics.
        
        Parameters
        ----------
        f : ca.Function
            Dynamics function.
        
        Returns
        -------
        f_dyn : ca.Function
            Integrated dynamics function.
        """
        X0 = ca.MX.sym('X0', self.nx)
        U = ca.MX.sym('U', self.nu)
        M = 4  # RK4 steps per interval
        DT = self.dt / M
        Xk = X0
        Q = 0
        for j in range(M):
            k1_x_dot, k1_l = f(Xk, U)
            k2_x_dot, k2_l = f(Xk + DT/2 * k1_x_dot, U)
            k3_x_dot, k3_l = f(Xk + DT/2 * k2_x_dot, U)
            k4_x_dot, k4_l = f(Xk + DT * k3_x_dot, U)
            Xk = Xk + (DT/6) * (k1_x_dot + 2*k2_x_dot + 2*k3_x_dot + k4_x_dot)
            Q += (DT/6) * (k1_l + 2*k2_l + 2*k3_l + k4_l)
        f_dyn = ca.Function('f_dyn', [X0, U], [Xk, Q], ['x', 'u'], ['x_next', 'l'])
        
        return f_dyn
    
    def _setup_mpc(self):
        """Set up the MPC optimization problem."""
        # Define dynamics and stage cost
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        x_dot = self.A @ x + self.B @ u
        self.xr = ca.DM(np.zeros(self.nx))  # reference state (origin)
        self.ur = ca.DM(np.zeros(self.nu))  # reference control input
        l = (x - self.xr).T @ self.Q @ (x - self.xr) + (u - self.ur).T @ self.R @ (u - self.ur)
        
        f = ca.Function('f', [x, u], [x_dot, l], ['x', 'u'], ['x_dot', 'l'])
        self.dyn = self._build_integrator(f)
        
        # Trajectory extraction functions
        X = ca.SX.sym('X', self.nx, self.N + 1)  # state trajectory
        U = ca.SX.sym('U', self.nu, self.N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.w0 = []     # initial guess
        lbw = []    # lower bounds
        ubw = []    # upper bounds
        J = 0       # objective
        g = []      # constraints
        lbg = []    # lower bounds on constraints
        ubg = []    # upper bounds on constraints
        
        # State bounds
        state_bounds = np.array([1.0, 1.5, 0.35, 1.0])
        # Control bounds
        control_bounds = np.array([1.0])
        
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds
        self.x_min = -state_bounds
        self.x_max = state_bounds
        self.u_min = -control_bounds
        self.u_max = control_bounds
        
        # Initialize state as symbolic parameter
        Xk = ca.SX.sym('X_0', self.nx)
        w += [Xk]
        lbw += self.x_min.tolist()
        ubw += self.x_max.tolist()
        self.w0 += [0.0]*self.nx
        
        for k in range(self.N):
            u_k = ca.SX.sym('u_' + str(k), self.nu)
            w += [u_k]
            lbw += self.u_min.tolist()
            ubw += self.u_max.tolist()
            self.w0 += [0.0]*self.nu
            
            # Compute next state and stage cost
            x_next, l_k = self.dyn(Xk, u_k)
            
            # Add state to decision variables
            Xk = ca.SX.sym('X_' + str(k+1), self.nx)
            w += [Xk]
            lbw += self.x_min.tolist()
            ubw += self.x_max.tolist()
            self.w0 += [0.0]*self.nx
            
            # Add dynamics constraint
            g += [x_next - Xk]
            lbg += [0.0]*self.nx
            ubg += [0.0]*self.nx
                        
            # Accumulate cost
            J += l_k
        
        # Add final cost using discrete ARE
        E = solve_continuous_are(self.A, self.B, self.Q.full(), self.R.full())
        # Compute LQR gain matrix: K = R^{-1} * B^T * E
        self.K = np.linalg.solve(self.R, self.B.T @ E)
        J += (Xk - self.xr).T @ ca.DM(E) @ (Xk - self.xr)
        
        # Store bounds
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        
        # Create NLP solver
        nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        opts = {"expand": True, "print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000, "tol": 1e-8}}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def generate_test_states(self, n_test=20, seed=42):
        """Generate random initial states for testing.
        
        Parameters
        ----------
        n_test : int
            Number of test cases.
        seed : int
            Random seed for reproducibility.
        """
        np.random.seed(seed)
        alpha = 0.5  # scaling factor to ensure test states are within training distribution
        self.test_bounds = self.state_bounds * alpha
        
        self.initial_states = np.zeros((n_test, self.nx))
        for i in range(self.nx):
            self.initial_states[:, i] = np.random.uniform(-self.test_bounds[i], self.test_bounds[i], n_test)
        
        print(f"Generated {n_test} uniformly sampled initial states")
    
    def load_learned_params(self, params_file=None):
        """Load parameters for the learned policy.
        
        Parameters
        ----------
        params_file : str or Path, optional
            Path to parameters file. If None, searches for latest file in model_dir.
        """
        if not HAS_CSNN or not self.net_fcn:
            raise RuntimeError("Neural network not initialized. Provide layer_sizes at initialization.")
        
        if params_file is None:
            model_name = get_model_name(self.layer_sizes)
            params_file = find_latest_params(self.model_dir, model_name, "yaml")
            params_file = "/home/pietro/data-driven/freiburg_stuff/ultro/models_nn/linear_mpc/optimal_params_lin_4x20x20x10_2026-03-13_11-41-08.yaml"
            if params_file is None:
                raise FileNotFoundError(f"No parameter files found for model {model_name} in {self.model_dir}")
        
        # Load parameters and metadata
        param_data = load_params(params_file, return_metadata=True)
        self.params_init_vec = param_data['params']
        
        # Load Q and R weights from file if available
        if param_data['q_weights'] is not None:
            self.Q = ca.DM(param_data['q_weights']).reshape((self.nx, self.nx))
            print(f"Loaded Q weights from file")
        
        if param_data['r_weight'] is not None:
            self.R = ca.DM([param_data['r_weight']]).reshape((self.nu, self.nu))
            print(f"Loaded R weight from file")
        
        print(f"Loaded parameters from {params_file}")
    
    def run_comparison(self):
        """Run the comparison between optimal MPC and learned policy."""
        
        if self.initial_states is None:
            raise ValueError("Generate test states first using generate_test_states()")
        
        if self.net_fcn is None:
            raise RuntimeError("Neural network policy not initialized. Provide layer_sizes at initialization.")
        
        N_TEST = len(self.initial_states)
        print(f"Testing {N_TEST} initial states...")
        
        # Clear previous results
        self.x_opt_batch = []
        self.u_opt_batch = []
        self.x_sim_batch = []
        self.u_sim_batch = []
        self.rmse_states_batch = []
        self.rmse_u_batch = []
        self.cost_opt_batch = []
        self.cost_sim_batch = []
        self.valid_indices = []
        
        # Solve MPC and simulate learned policy for each initial state
        for i, x0 in enumerate(self.initial_states):
            x0_list = x0.tolist()
            
            # Set bounds for initial state
            lbw_i = self.lbw.copy()
            ubw_i = self.ubw.copy()
            lbw_i[:self.nx] = x0_list
            ubw_i[:self.nx] = x0_list
            
            # Solve the MPC NLP
            solution = self.solver(x0=self.w0, lbx=lbw_i, ubx=ubw_i, lbg=self.lbg, ubg=self.ubg)
            
            # Extract the optimal solution
            w_opt = solution['x'].full().flatten()
            
            # Extract optimal state and control trajectories
            x_opt, u_opt = self.extract_traj(w_opt)
            x_opt = np.asarray(x_opt)
            u_opt = np.asarray(u_opt)

            # Simulate the learned policy for the same initial state
            x_sim = np.zeros((self.nx, self.N + 1))
            u_sim = np.zeros((self.nu, self.N))
            x_sim[:, 0] = x0
            for k in range(self.N):
                u_next = self.net_fcn(x_sim[:, k], self.params_init_vec).full().flatten()
                if self.nu == 1:
                    u_sim[:, k] = u_next[0]
                else:
                    u_sim[:, k] = u_next
                x_next = self.A @ x_sim[:, k] + self.B @ u_sim[:, k]
                x_sim[:, k + 1] = x_next.flatten()
            
            # Check for NaN values in the simulation
            has_nan = np.isnan(x_sim).any() or np.isnan(u_sim).any()
            if has_nan:
                print(f"  Warning: Test case {i} produced NaN values, skipping...")
                continue
            
            # Store results for valid test case
            self.x_opt_batch.append(x_opt)
            self.u_opt_batch.append(u_opt)
            self.x_sim_batch.append(x_sim)
            self.u_sim_batch.append(u_sim)
            self.valid_indices.append(i)
            
            # Compute RMSE for states and control
            rmse_states = np.sqrt(np.mean((x_opt - x_sim) ** 2, axis=1))
            rmse_u = np.sqrt(np.mean((u_opt - u_sim) ** 2))
            self.rmse_states_batch.append(rmse_states)
            self.rmse_u_batch.append(rmse_u)
            
            # Compute cost for optimal trajectory
            cost_opt = self._compute_trajectory_cost(x_opt, u_opt)
            self.cost_opt_batch.append(cost_opt)
            
            # Compute cost for simulated trajectory
            cost_sim = self._compute_trajectory_cost(x_sim, u_sim)
            self.cost_sim_batch.append(cost_sim)
            
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{N_TEST} test cases")
        
        # Report on valid test cases
        N_VALID = len(self.valid_indices)
        N_INVALID = N_TEST - N_VALID
        if N_INVALID > 0:
            print(f"\n{N_INVALID} test case(s) were skipped due to NaN values.")
        print(f"Analyzing {N_VALID} valid test cases...\n")
    
    def _compute_trajectory_cost(self, x_traj, u_traj):
        """Compute the total cost for a trajectory.
        
        Parameters
        ----------
        x_traj : np.ndarray
            State trajectory (nx x N+1).
        u_traj : np.ndarray
            Control trajectory (nu x N).
        
        Returns
        -------
        cost : float
            Total trajectory cost.
        """
        E = solve_discrete_are(self.A, self.B, self.Q.full(), self.R.full())
        cost = 0.0
        for k in range(self.N):
            x_k = x_traj[:, k]
            u_k = u_traj[:, k]
            stage_cost = float(((x_k - self.xr.full().flatten()).T @ self.Q.full() @ 
                               (x_k - self.xr.full().flatten()) + 
                               (u_k - self.ur.full().flatten()).T @ self.R.full() @ 
                               (u_k - self.ur.full().flatten())).item())
            cost += stage_cost
        # Add terminal cost
        x_N = x_traj[:, self.N]
        terminal_cost = float(((x_N - self.xr.full().flatten()).T @ E @ 
                                (x_N - self.xr.full().flatten())).item())
        cost += terminal_cost
        return cost
    
    def print_results(self, threshold_rmse=0.01):
        """Print comparison results.
        
        Parameters
        ----------
        threshold_rmse : float
            RMSE threshold for reporting.
        """
        N_VALID = len(self.valid_indices)
        
        if N_VALID == 0:
            print("No valid test cases to analyze.")
            return
        
        # Count how many cases have RMSE above threshold
        num_above_threshold_states = np.sum(np.array(self.rmse_states_batch) > threshold_rmse, axis=0)
        num_above_threshold_u = np.sum(np.array(self.rmse_u_batch) > threshold_rmse)
        print(f"\nNumber of test cases with RMSE above {threshold_rmse}:")
        for i in range(self.nx):
            print(f"  State {i}: {num_above_threshold_states[i]}")
        print(f"  Control: {num_above_threshold_u}")
        
        # Compute average RMSE over all test cases
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        print(f"\nAverage RMSE over {N_VALID} valid test cases:")
        for i in range(self.nx):
            print(f"  State {i}: {avg_rmse_states[i]:.4f}")
        print(f"  Control: {avg_rmse_u:.4f}")
        
        # Compute cost statistics
        cost_opt_array = np.array(self.cost_opt_batch)
        cost_sim_array = np.array(self.cost_sim_batch)
        cost_diff = cost_sim_array - cost_opt_array
        cost_ratio = cost_sim_array / cost_opt_array
        suboptimality_pct = 100.0 * cost_diff / cost_opt_array
        
        print(f"\nCost Function Comparison over {N_VALID} valid test cases:")
        print(f"  Optimal MPC cost (avg): {cost_opt_array.mean():.4f} ± {cost_opt_array.std():.4f}")
        print(f"  Learned policy cost (avg): {cost_sim_array.mean():.4f} ± {cost_sim_array.std():.4f}")
        print(f"  Cost difference (avg): {cost_diff.mean():.4f} ± {cost_diff.std():.4f}")
        print(f"  Cost ratio (learned/optimal, avg): {cost_ratio.mean():.4f}")
        print(f"  Suboptimality percentage (avg): {suboptimality_pct.mean():.2f}%")
        print(f"  Max suboptimality: {suboptimality_pct.max():.2f}%")
        print(f"  Min suboptimality: {suboptimality_pct.min():.2f}%")
    
    def plot_trajectories(self):
        """Plot optimal vs learned trajectories."""
        N_VALID = len(self.valid_indices)
        if N_VALID == 0:
            print("No valid test cases to plot.")
            return
        
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_x = np.arange(self.N + 1)
        t_u = np.arange(self.N)
        
        plt.figure(figsize=(12, 3*self.nx + 3))
        
        # Plot states
        for state_idx in range(self.nx):
            plt.subplot(self.nx + 1, 1, state_idx + 1)
            for i in range(N_VALID):
                plt.plot(t_x, self.x_opt_batch[i][state_idx, :], '-', alpha=0.3, color='C0')
                plt.plot(t_x, self.x_sim_batch[i][state_idx, :], '-', alpha=0.3, color='C1')
            plt.plot([], [], '-', label=f'Optimal (RMSE={avg_rmse_states[state_idx]:.3f})', color='C0')
            plt.plot([], [], '-', label='Learned', color='C1')
            plt.ylabel(f'State x[{state_idx}]')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Control over time
        plt.subplot(self.nx + 1, 1, self.nx + 1)
        for i in range(N_VALID):
            plt.plot(t_u, self.u_opt_batch[i].flatten(), alpha=0.3, color='C2')
            plt.plot(t_u, self.u_sim_batch[i].flatten(), alpha=0.3, color='C3')
        plt.plot([], [], label=f'Optimal (RMSE={avg_rmse_u:.3f})', color='C2')
        plt.plot([], [], label='Learned', color='C3')
        plt.xlabel('Time Step')
        plt.ylabel('Control')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle(f'Optimal vs Learned Policy ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def plot_errors(self):
        """Plot trajectory errors."""
        N_VALID = len(self.valid_indices)
        if N_VALID == 0:
            print("No valid test cases to plot.")
            return
        
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_x = np.arange(self.N + 1)
        t_u = np.arange(self.N)
        
        plt.figure(figsize=(12, 3*self.nx + 3))
        
        # Error in states
        for state_idx in range(self.nx):
            plt.subplot(self.nx + 1, 1, state_idx + 1)
            for i in range(N_VALID):
                error = self.x_opt_batch[i][state_idx, :] - self.x_sim_batch[i][state_idx, :]
                plt.plot(t_x, error, '-', alpha=0.3, color='C0')
            plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[state_idx]:.3f}', color='C0')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.ylabel(f'State x[{state_idx}] Error')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Error in control
        plt.subplot(self.nx + 1, 1, self.nx + 1)
        for i in range(N_VALID):
            error_u = self.u_opt_batch[i].flatten() - self.u_sim_batch[i].flatten()
            plt.plot(t_u, error_u, alpha=0.3, color='C1')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_u:.3f}', color='C1')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Control Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle(f'Trajectory Errors (Optimal - Learned) ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def plot_costs(self):
        """Plot cost comparison."""
        N_VALID = len(self.valid_indices)
        if N_VALID == 0:
            print("No valid test cases to plot.")
            return
        
        cost_opt_array = np.array(self.cost_opt_batch)
        cost_sim_array = np.array(self.cost_sim_batch)
        cost_diff = cost_sim_array - cost_opt_array
        suboptimality_pct = 100.0 * cost_diff / cost_opt_array
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Cost values
        plt.subplot(2, 2, 1)
        plt.bar(range(N_VALID), cost_opt_array, alpha=0.7, label='Optimal MPC', color='C0')
        plt.bar(range(N_VALID), cost_sim_array, alpha=0.7, label='Learned Policy', color='C1')
        plt.xlabel('Test Case')
        plt.ylabel('Total Cost')
        plt.title('Cost Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Cost difference
        plt.subplot(2, 2, 2)
        plt.bar(range(N_VALID), cost_diff, color='C2', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Test Case')
        plt.ylabel('Cost Difference (Learned - Optimal)')
        plt.title(f'Cost Difference (Avg: {cost_diff.mean():.2f})')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Suboptimality percentage
        plt.subplot(2, 2, 3)
        plt.bar(range(N_VALID), suboptimality_pct, color='C3', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Test Case')
        plt.ylabel('Suboptimality (%)')
        plt.title(f'Suboptimality Percentage (Avg: {suboptimality_pct.mean():.2f}%)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Scatter plot
        plt.subplot(2, 2, 4)
        plt.scatter(cost_opt_array, cost_sim_array, alpha=0.6, s=50, color='C4')
        min_cost = min(cost_opt_array.min(), cost_sim_array.min())
        max_cost = max(cost_opt_array.max(), cost_sim_array.max())
        plt.plot([min_cost, max_cost], [min_cost, max_cost], 'k--', alpha=0.5, label='Perfect match')
        plt.xlabel('Optimal MPC Cost')
        plt.ylabel('Learned Policy Cost')
        plt.title('Cost Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.suptitle(f'Cost Function Analysis ({N_VALID} Valid Test Cases)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def solve_MPC(self, x0, ret_seq=False):
        """Solve the MPC problem for a given initial state x0.
        
        Parameters
        ----------
        x0 : array_like
            Initial state
        ret_seq : bool, optional
            If True, return the full control sequence; otherwise return only first control
            
        Returns
        -------
        float or np.ndarray
            First control input or full control sequence
        """
        x0_list = np.array(x0).flatten().tolist()
        
        # Set bounds for initial state
        lbw_i = self.lbw.copy()
        ubw_i = self.ubw.copy()
        lbw_i[:self.nx] = x0_list
        ubw_i[:self.nx] = x0_list
        
        # Solve the MPC NLP
        solution = self.solver(x0=self.w0, lbx=lbw_i, ubx=ubw_i, lbg=self.lbg, ubg=self.ubg)
        
        w_opt = solution['x'].full().flatten()
        x_opt, u_opt = self.extract_traj(w_opt)
        u_opt = u_opt.full().flatten()
        
        if ret_seq:
            return u_opt
        return u_opt[0]
    
    def close_loop_simulation(self, x0, Nsim=100):
        """Test the learned policy in closed-loop simulation from a given initial state."""
        if self.net_fcn is None:
            raise RuntimeError("Neural network not initialized.")
        
        x_sim = np.zeros((self.nx, Nsim + 1))
        u_sim = np.zeros((self.nu, Nsim))
        x_sim[:, 0] = np.array(x0).flatten()
        
        for k in range(Nsim):
            u_next = self.net_fcn(x_sim[:, k], self.params_init_vec).full().flatten()
            if self.nu == 1:
                u_sim[:, k] = u_next[0]
            else:
                u_sim[:, k] = u_next
            x_next = self.A @ x_sim[:, k] + self.B @ u_sim[:, k]
            x_sim[:, k + 1] = x_next.flatten()
        
        # Check for NaN values
        has_nan = np.isnan(x_sim).any() or np.isnan(u_sim).any()
        if has_nan:
            print(f"Warning: Closed-loop simulation produced NaN values.")
            return None, None
        
        # Plot results
        time = np.arange(Nsim + 1) * self.dt
        time_u = np.arange(Nsim) * self.dt
        
        fig, axes = plt.subplots(self.nx + 1, 1, figsize=(10, 3*self.nx + 3))
        title = "Closed-Loop Simulation with Learned Policy"
        
        for i in range(self.nx):
            axes[i].plot(time, x_sim[i, :], linewidth=2)
            axes[i].set_ylabel(f'State x[{i}]')
            axes[i].grid(True)
            if i == 0:
                axes[i].set_title(title, fontsize=12, fontweight='bold')
        
        axes[self.nx].step(time_u, u_sim[0, :], where='post', linewidth=2)
        axes[self.nx].set_ylabel('Control')
        axes[self.nx].set_xlabel('Time (s)')
        axes[self.nx].grid(True)
        
        plt.tight_layout()
        return x_sim, u_sim
    
    def plot_policy(self):
        """Plot the learned policy as a function of state."""
        if self.net_fcn is None:
            raise RuntimeError("Neural network not initialized.")
        
        # Create a grid of states for plotting
        i = 2
        n_points = 50
        state_ranges = [(-self.test_bounds[i], self.test_bounds[i])]
        state_grids = [np.linspace(r[0], r[1], n_points) for r in state_ranges]
        grid = state_grids[0]
        # mesh = np.meshgrid(*state_grids)
        # state_grid_flat = np.array([m.flatten() for m in mesh])
        
        # Compute optimal control, learned control, and LQR control on the grid
        U_opt_grid = np.zeros(grid.shape[0])
        U_learned_grid = np.zeros(grid.shape[0])
        U_lqr_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            # State: [position, velocity, angle, angular_velocity]
            x_i = np.array([0.0, 0.0, grid[i], 0.0])
            
            # Solve MPC for this state to get optimal control
            lbw_i = self.lbw.copy()
            ubw_i = self.ubw.copy()
            lbw_i[:self.nx] = x_i.tolist()
            ubw_i[:self.nx] = x_i.tolist()
            solution = self.solver(x0=self.w0, lbx=lbw_i, ubx=ubw_i, lbg=self.lbg, ubg=self.ubg)
            w_opt = solution['x'].full().flatten()
            _, u_opt_traj = self.extract_traj(w_opt)
            U_opt_grid[i] = u_opt_traj[0, 0]  # first control input

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i.tolist(), self.params_init_vec).full().flatten()[0]
            
            # Compute LQR control: u = -K * (x - x_ref)
            x_ref = self.xr.full().flatten()
            U_lqr_grid[i] = (-self.K @ (x_i - x_ref))[0]

        # Reshape for plotting
        U_opt_grid = U_opt_grid.reshape(grid.shape)
        U_learned_grid = U_learned_grid.reshape(grid.shape)
        U_lqr_grid = U_lqr_grid.reshape(grid.shape)
        
        # Compute errors
        U_error_learned = np.abs(U_opt_grid - U_learned_grid)
        print(f"Max error between optimal and learned policy: {U_error_learned.max():.4f} at angle {grid[U_error_learned.argmax()]:.4f} rad")
        U_error_lqr = np.abs(U_lqr_grid - U_learned_grid)

        fig = plt.figure(figsize=(16, 10))
        
        # First row: Optimal, Learned, and LQR policies
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(grid, U_opt_grid.flatten(), alpha=0.7, color='C0', linewidth=2, label='Optimal MPC')
        ax1.set_title('Optimal Control Policy')
        ax1.set_xlabel('Angle (rad)')
        ax1.set_ylabel('Control Force (N)')
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(grid, U_learned_grid.flatten(), alpha=0.7, color='C1', linewidth=2, label='Learned NN')
        ax2.set_title(f'Learned Control Policy')
        ax2.set_xlabel('Angle (rad)')
        ax2.set_ylabel('Control Force (N)')
        ax2.grid(True)
        ax2.legend()
        
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(grid, U_lqr_grid.flatten(), alpha=0.7, color='C3', linewidth=2, label='LQR')
        ax3.set_title('LQR Control Policy')
        ax3.set_xlabel('Angle (rad)')
        ax3.set_ylabel('Control Force (N)')
        ax3.grid(True)
        ax3.legend()
        
        # Second row: Comparison and errors
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(grid, U_opt_grid.flatten(), alpha=0.7, color='C0', linewidth=2, label='Optimal MPC')
        ax4.plot(grid, U_learned_grid.flatten(), alpha=0.7, color='C1', linewidth=2, linestyle='--', label='Learned NN')
        # ax4.plot(grid, U_lqr_grid.flatten(), alpha=0.7, color='C3', linewidth=2, linestyle=':', label='LQR')
        ax4.set_title('All Policies Comparison')
        ax4.set_xlabel('Angle (rad)')
        ax4.set_ylabel('Control Force (N)')
        ax4.grid(True)
        ax4.legend()
        
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.semilogy(grid, U_error_learned.flatten(), alpha=0.7, color='C2', linewidth=2, label='Learned error')
        ax5.set_title('|Optimal - Learned| (log scale)')
        ax5.set_xlabel('Angle (rad)')
        ax5.set_ylabel('|Error| (N)')
        ax5.grid(True, which='both')
        ax5.legend()
        
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.semilogy(grid, U_error_lqr.flatten(), alpha=0.7, color='C4', linewidth=2, label='LQR error')
        ax6.set_title('|LQR - Learned| (log scale)')
        ax6.set_xlabel('Angle (rad)')
        ax6.set_ylabel('|Error| (N)')
        ax6.grid(True, which='both')
        ax6.legend()

        plt.tight_layout()
    
    def show_plots(self):
        """Display all plots."""
        plt.show()


if __name__ == "__main__":
    # Define a simple linear system: x_{k+1} = A x_k + B u_k
    # This is a 4D system similar to the discretized cart-pole linearization
    A = np.array([
        [1.0, 0.1, 0.0, 0.0],
        [0.0, 0.9818, 0.2673, 0.0],
        [0.0, 0.0, 1.0, 0.1],
        [0.0, -0.0455, 3.1182, 1.0]
    ])
    
    B = np.array([[0.0], [0.1818], [0.0], [0.4546]])
    
    # Define cost matrices
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # state cost
    R = np.diag([1.0])                  # control cost
    
    # Run closed-loop simulation with MPC
    print("\nRunning closed-loop MPC simulation...")
    # Note: This requires net_fcn which isn't available without neural network
    # So we'll skip this part
    
    print("\n" + "=" * 60)
    print("Example: Linear MPC With Neural Network Policy Comparison")
    print("=" * 60)
    
    # Create comparison object with neural network
    try:
        mpc_nn_system = LinearMPCComparison(
            A=A,
            B=B,
            dt=0.1,
            N=10,
            Q=Q,
            R=R,
            layer_sizes=[4, 20, 20, 10],  # 4 inputs, hidden layer, 1 output
            beta=100.0,
            # model_dir=Path(__file__).parent.parent / "models_nn",
            wait_for_input=False
        )
        
        # Try to load learned parameters (if available)
        try:
            mpc_nn_system.load_learned_params()
            
            # Generate test states
            mpc_nn_system.generate_test_states(n_test=200, seed=36)
            
            # Run comparison
            print("Running comparison between optimal MPC and learned policy...")
            mpc_nn_system.run_comparison()
            
            # Print results
            mpc_nn_system.print_results()
            
            # Generate plots
            print("Generating plots...")
            mpc_nn_system.plot_trajectories()
            mpc_nn_system.plot_errors()
            np.random.seed(37)
            x0 = np.random.uniform(-mpc_nn_system.test_bounds, mpc_nn_system.test_bounds)
            mpc_nn_system.close_loop_simulation(x0, Nsim=100)
            # mpc_nn_system.plot_costs()
            mpc_nn_system.plot_policy()
            mpc_nn_system.show_plots()
            
        except FileNotFoundError as e:
            print(f"Learned parameters not found: {e}")
            print("Skipping comparison. Make sure to train a network first.")
    
    except ImportError as e:
        print(f"Neural network module not available: {e}")
        print("To use neural network features, install the csnn module.")