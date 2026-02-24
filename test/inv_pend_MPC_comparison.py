from datetime import datetime
import json
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are

from csnn import set_sym_type, Linear, Sequential, ReLU, Softplus

from models.inverted_pendulum import InvertedPendulum


def get_model_name(layer_sizes):
    """Generate model name from layer sizes.
    
    Parameters
    ----------
    layer_sizes : list of int
        List containing the sizes of each layer in the network.
    
    Returns
    -------
    model_name : str
        Model name in format like '2x6x6x1'
    """
    return 'x'.join(map(str, layer_sizes))

def find_latest_params(model_dir, model_name, extension="yaml"):
    """Find the most recent parameter file for a given model.
    
    Parameters
    ----------
    model_dir : Path
        Directory containing parameter files.
    model_name : str
        Model name (e.g., '2x6x6x1').
    extension : str
        File extension ('yaml' or 'json').
    
    Returns
    -------
    latest_file : Path or None
        Path to the most recent parameter file, or None if not found.
    """
    pattern = f"optimal_params_ip_{model_name}_*.{extension}"
    files = list(model_dir.glob(pattern))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file

def load_params(params_file):
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
    n_param = data.get("n_param")
    
    if optimal_params is None or n_param is None:
        raise ValueError("Invalid parameters file format. Expected keys: 'optimal_params', 'n_param'.")
    
    params_init_vec = np.array(optimal_params).flatten()
    # safety: if sizes mismatch, pad or truncate to n_param
    if params_init_vec.size < n_param:
        params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
    elif params_init_vec.size > n_param:
        params_init_vec = params_init_vec[:n_param]
        
    return params_init_vec

class InvertedPendulumMPCComparison:
    """Class for comparing optimal MPC with learned neural network policy for inverted pendulum system."""
    
    def __init__(self, layer_sizes=[2, 20, 1], beta=0.5, horizon=20, model_dir=None):
        """Initialize the MPC comparison.
        
        Parameters
        ----------
        layer_sizes : list of int
            Neural network architecture (including input and output layers).
        beta : float
            Softplus beta parameter.
        horizon : int
            MPC prediction horizon length.
        model_dir : Path or None
            Directory containing trained model parameters.
        """
        print("Setting up the MPC problem for testing...")
        start_time = time.time()
        
        # Store parameters
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.N = horizon
        self.model_dir = model_dir if model_dir is not None else Path(__file__).parent.parent / "models_nn"
        
        # Initialize inverted pendulum model
        self.inverted_pendulum = InvertedPendulum(sym_type='SX')
        
        # Declare variables
        self.NX = 2      # state dimension (position, velocity, angle, angular velocity)
        self.NU = 1      # control dimension
        
        # Set up neural network
        self._setup_network()
        
        # Set up MPC problem
        self._setup_mpc()
        
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
        
        print(f"Setup completed in {time.time() - start_time:.2f} seconds")
    
    def _setup_network(self):
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
        
        # Get flattened network parameters
        params = []
        for _, param in self.net.parameters():
            params.append(ca.reshape(param, -1, 1))
        self.params_flattened = ca.vertcat(*params)
        
        # Create network function
        x = ca.SX.sym('x', self.NX)
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
        X0 = ca.MX.sym('X0', self.NX)
        U = ca.MX.sym('U', self.NU)
        M = 4  # RK4 steps per interval
        DT = self.inverted_pendulum.dt / M
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
        x = ca.SX.sym('x', self.NX)
        u = ca.SX.sym('u', self.NU)
        self.Q = ca.diag(ca.DM([100, 1]))  # state cost weights
        self.R = ca.diag(ca.DM([1.0]))  # control cost weight
        x_dot = self.inverted_pendulum.dynamics(x, u)
        self.xr = ca.DM([0.0, 0.0])  # reference state (upright pendulum)
        self.ur = ca.DM([0.0])                  # reference control input
        l = (x - self.xr).T @ self.Q @ (x - self.xr) + (u - self.ur).T @ self.R @ (u - self.ur)
        f = ca.Function('f', [x, u], [x_dot, l], ['x', 'u'], ['x_dot', 'l'])
        self.f_dyn = self._build_integrator(f)
        
        # Trajectory extraction functions
        X = ca.SX.sym('X', self.NX, self.N + 1)  # state trajectory
        U = ca.SX.sym('U', self.NU, self.N)    # control trajectory
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
        self.theta_bound = np.pi/3
        self.omega_bound = 2.0
        
        # Control bounds
        self.u_bound = 10.0
        
        # Initialize state as symbolic parameter
        Xk = ca.SX.sym('X_0', self.NX)
        w += [Xk]
        lbw += [-self.theta_bound, -self.omega_bound]
        ubw += [self.theta_bound, self.omega_bound]
        self.w0 += [0.0]*self.NX
        
        for k in range(self.N):
            u_k = ca.SX.sym('u_' + str(k), self.NU)
            w += [u_k]
            lbw += [-self.u_bound]*self.NU
            ubw += [self.u_bound]*self.NU
            self.w0 += [0.0]*self.NU
            
            # Compute next state
            Xk_next, l_k = self.f_dyn(Xk, u_k)
            # Add state to decision variables
            Xk = ca.SX.sym('X_' + str(k+1), self.NX)
            w += [Xk]
            lbw += [-self.theta_bound, -self.omega_bound]
            ubw += [self.theta_bound, self.omega_bound]
            self.w0 += [0.0]*self.NX
            
            # Add dynamics constraint
            g += [Xk_next - Xk]
            lbg += [0.0]*self.NX
            ubg += [0.0]*self.NX
                        
            # Accumulate cost
            J += l_k
        
        # Add final cost using linearized dynamics
        x_eq = ca.DM([0.0, 0.0])  # equilibrium state (upright pendulum)
        u_eq = ca.DM([0.0])                 # equilibrium control input
        A, B = self.inverted_pendulum.lin_dyn(x_eq, u_eq)
        self.E = solve_continuous_are(A.full(), B.full(), self.Q.full(), self.R.full())
        J += (Xk - self.xr).T @ ca.DM(self.E) @ (Xk - self.xr)
        
        # Store bounds
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        
        # Create NLP solver
        nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        opts = {"expand": False, "print_time": False, "ipopt": {"print_level": 0, "max_iter": 3000, "tol": 1e-8}}
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
        theta_test_bound = np.pi/6
        omega_test_bound = 1.0
        
        self.initial_states = np.zeros((n_test, self.NX))
        self.initial_states[:, 0] = np.random.uniform(-theta_test_bound, theta_test_bound, n_test)
        self.initial_states[:, 1] = np.random.uniform(-omega_test_bound, omega_test_bound, n_test)
        
        print(f"Generated {n_test} uniformly sampled initial states")
    
    def load_learned_params(self):
        """Load parameters for the learned policy."""
        model_name = get_model_name(self.layer_sizes)
        params_file = find_latest_params(self.model_dir, model_name, "yaml")
        if params_file is None:
            raise FileNotFoundError(f"No parameter files found for model {model_name} in {self.model_dir}")
        self.params_init_vec = load_params(params_file)
        print(f"Loaded parameters from {params_file}")
    
    def run_comparison(self, wait_for_input=True):
        """Run the comparison between optimal MPC and learned policy.
        
        Parameters
        ----------
        wait_for_input : bool
            Whether to wait for user input before starting.
        """
        if self.initial_states is None:
            raise ValueError("Generate test states first using generate_test_states()")
        
        if not hasattr(self, 'params_init_vec'):
            self.load_learned_params()
        
        if wait_for_input:
            input("Press Enter to start testing...")
        
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
            lbw_i[:self.NX] = x0_list
            ubw_i[:self.NX] = x0_list
            
            # Solve the MPC NLP
            solution = self.solver(x0=self.w0, lbx=lbw_i, ubx=ubw_i, lbg=self.lbg, ubg=self.ubg)
            
            # Extract the optimal solution
            w_opt = solution['x'].full().flatten()
            
            # Extract optimal state and control trajectories
            x_opt, u_opt = self.extract_traj(w_opt)
            x_opt = np.asarray(x_opt)
            u_opt = np.asarray(u_opt)
            self.x_opt_batch.append(x_opt)
            self.u_opt_batch.append(u_opt)

            # Simulate the learned policy for the same initial state
            x_sim = np.zeros((self.NX, self.N + 1))
            u_sim = np.zeros((self.NU, self.N))
            x_sim[:, 0] = x0
            for k in range(self.N):
                u_next = self.net_fcn(x_sim[:, k], self.params_init_vec).full().flatten()
                u_sim[:, k] = u_next
                x_next, _ = self.f_dyn(x_sim[:, k], u_next)
                x_sim[:, k + 1] = x_next.full().flatten()
            
            # Check for NaN values in the simulation
            has_nan = np.isnan(x_sim).any() or np.isnan(u_sim).any()
            if has_nan:
                print(f"  Warning: Test case {i} produced NaN values, skipping...")
                print(f"    Initial state: {x0}")
                continue
            
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
            State trajectory (NX x N+1).
        u_traj : np.ndarray
            Control trajectory (NU x N).
        
        Returns
        -------
        cost : float
            Total trajectory cost.
        """
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
        terminal_cost = float(((x_N - self.xr.full().flatten()).T @ self.E @ 
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
        
        # Count how many cases have RMSE above threshold
        num_above_threshold_states = np.sum(np.array(self.rmse_states_batch) > threshold_rmse, axis=0)
        num_above_threshold_u = np.sum(np.array(self.rmse_u_batch) > threshold_rmse)
        print(f"\nNumber of test cases with RMSE above {threshold_rmse}:")
        print(f"  State 1 (angle): {num_above_threshold_states[0]}")
        print(f"  State 2 (angular velocity): {num_above_threshold_states[1]}")
        print(f"  Control: {num_above_threshold_u}")
        
        # Compute average RMSE over all test cases
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        print(f"\nAverage RMSE over {N_VALID} valid test cases:")
        print(f"  State 1 (angle): {avg_rmse_states[0]:.4f}")
        print(f"  State 2 (angular velocity): {avg_rmse_states[1]:.4f}")
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
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_x = self.inverted_pendulum.dt * np.arange(self.N + 1)
        t_u = self.inverted_pendulum.dt * np.arange(self.N)
        
        plt.figure(figsize=(12, 10))
        
        # State 1 (angle) over time
        plt.subplot(3, 1, 1)
        for i in range(N_VALID):
            plt.plot(t_x, self.x_opt_batch[i][0, :], '-', alpha=0.3, color='C0')
            plt.plot(t_x, self.x_sim_batch[i][0, :], '-', alpha=0.3, color='C1')
        plt.plot([], [], '-', label=f'Optimal (Avg RMSE={avg_rmse_states[0]:.3f})', color='C0')
        plt.plot([], [], '-', label='Learned', color='C1')
        plt.ylim(-np.pi/3, np.pi/3)
        plt.ylabel('Angle')
        plt.grid(True)
        plt.legend()
        
        # State 2 (angular velocity) over time
        plt.subplot(3, 1, 2)
        for i in range(N_VALID):
            plt.plot(t_x, self.x_opt_batch[i][1, :], '-', alpha=0.3, color='C0')
            plt.plot(t_x, self.x_sim_batch[i][1, :], '-', alpha=0.3, color='C1')
        plt.plot([], [], '-', label=f'Optimal (Avg RMSE={avg_rmse_states[1]:.3f})', color='C0')
        plt.plot([], [], '-', label='Learned', color='C1')
        plt.ylim(-2.0, 2.0)
        plt.ylabel('Angular Velocity')
        plt.grid(True)
        plt.legend()
        
        # Control over time
        plt.subplot(3, 1, 3)
        for i in range(N_VALID):
            plt.step(t_u, self.u_opt_batch[i].flatten(), where='post', alpha=0.3, color='C2')
            plt.step(t_u, self.u_sim_batch[i].flatten(), where='post', alpha=0.3, color='C3')
        plt.step([], [], where='post', label=f'Optimal (Avg RMSE={avg_rmse_u:.3f})', color='C2')
        plt.step([], [], where='post', label='Learned', color='C3')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'Optimal vs Learned Policy ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
    def plot_policy(self):
        """Plot learned policy vs optimal control."""
        
        # Create a grid of states for plotting the policy
        theta_grid = np.linspace(-self.theta_bound, self.theta_bound, 50)
        # omega_grid = np.linspace(-self.omega_bound, self.omega_bound, 50)

        # Compute optimal control and learned control on the grid
        U_opt_grid = np.zeros(theta_grid.shape[0])
        U_learned_grid = np.zeros(theta_grid.shape[0])

        for i in range(theta_grid.shape[0]):
            x_i = [theta_grid[i], 0.0]  # zero angular velocity slice
            # Solve MPC for this state to get optimal control
            lbw_i = self.lbw.copy()
            ubw_i = self.ubw.copy()
            lbw_i[:self.NX] = x_i
            ubw_i[:self.NX] = x_i
            solution = self.solver(x0=self.w0, lbx=lbw_i, ubx=ubw_i, lbg=self.lbg, ubg=self.ubg)
            w_opt = solution['x'].full().flatten()
            _, u_opt_traj = self.extract_traj(w_opt)
            U_opt_grid[i] = u_opt_traj[0, 0]  # first control input

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i, self.params_init_vec).full().flatten()[0]

        # Reshape for plotting
        U_opt_grid = U_opt_grid.reshape(theta_grid.shape)
        U_learned_grid = U_learned_grid.reshape(theta_grid.shape)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(theta_grid, U_opt_grid.flatten(), alpha=0.5, color='C0', linewidth=2)
        ax1.set_title('Optimal Control Policy')
        ax1.set_xlabel('Angle (rad)')
        ax1.set_ylabel('Control')
        ax1.grid(True)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(theta_grid, U_learned_grid.flatten(), alpha=0.5, color='C1', linewidth=2)
        ax2.set_title(f'Learned Control Policy')
        ax2.set_xlabel('Angle (rad)')
        ax2.set_ylabel('Control')
        ax2.grid(True)

        plt.tight_layout()
    
    def plot_policy_3d(self, elev=30, azim=-60):
        """3D surface plot of optimal and learned control policies.

        Parameters
        ----------
        elev : float
            Elevation angle for 3D view.
        azim : float
            Azimuth angle for 3D view.
        """
        avg_rmse_u = np.mean(self.rmse_u_batch)

        # Create a grid of states for plotting the policy
        theta_grid = np.linspace(-self.theta_bound, self.theta_bound, 50)
        omega_grid = np.linspace(-self.omega_bound, self.omega_bound, 50)
        Theta, Omega = np.meshgrid(theta_grid, omega_grid)
        X_grid = np.vstack([Theta.flatten(), Omega.flatten()])

        # Compute optimal control and learned control on the grid
        U_opt_grid = np.zeros(X_grid.shape[1])
        U_learned_grid = np.zeros(X_grid.shape[1])

        for i in range(X_grid.shape[1]):
            x_i = X_grid[:, i]
            # Solve MPC for this state to get optimal control
            lbw_i = self.lbw.copy()
            ubw_i = self.ubw.copy()
            lbw_i[:self.NX] = x_i.tolist()
            ubw_i[:self.NX] = x_i.tolist()
            solution = self.solver(x0=self.w0, lbx=lbw_i, ubx=ubw_i, lbg=self.lbg, ubg=self.ubg)
            w_opt = solution['x'].full().flatten()
            _, u_opt_traj = self.extract_traj(w_opt)
            U_opt_grid[i] = u_opt_traj[0, 0]  # first control input

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i, self.params_init_vec).full().flatten()[0]

        # Reshape for plotting
        U_opt_grid = U_opt_grid.reshape(Theta.shape)
        U_learned_grid = U_learned_grid.reshape(Theta.shape)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(Theta, Omega, U_opt_grid, cmap='viridis', rcount=50, ccount=50, linewidth=0, antialiased=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.6)
        ax1.set_title('Optimal Control Policy (3D)')
        ax1.set_xlabel('Angle (rad)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.set_zlabel('Control')
        ax1.view_init(elev=elev, azim=azim)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(Theta, Omega, U_learned_grid, cmap='viridis', rcount=50, ccount=50, linewidth=0, antialiased=True)
        fig.colorbar(surf2, ax=ax2, shrink=0.6)
        ax2.set_title(f'Learned Policy (3D)')
        ax2.set_xlabel('Angle (rad)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_zlabel('Control')
        ax2.view_init(elev=elev, azim=azim)

        plt.tight_layout()

    def plot_errors(self):
        """Plot trajectory errors."""
        N_VALID = len(self.valid_indices)
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_x = np.arange(self.N + 1)
        t_u = np.arange(self.N)
        
        plt.figure(figsize=(12, 10))
        
        # Error in state 1 (angle) over time
        plt.subplot(3, 1, 1)
        for i in range(N_VALID):
            error_x1 = self.x_opt_batch[i][0, :] - self.x_sim_batch[i][0, :]
            plt.plot(t_x, error_x1, '-', alpha=0.3, color='C0')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[0]:.3f}', color='C0')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Angle Error')
        plt.grid(True)
        plt.legend()
        
        # Error in state 2 (angular velocity) over time
        plt.subplot(3, 1, 2)
        for i in range(N_VALID):
            error_x2 = self.x_opt_batch[i][1, :] - self.x_sim_batch[i][1, :]
            plt.plot(t_x, error_x2, '-', alpha=0.3, color='C1')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[1]:.3f}', color='C1')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Angular Velocity Error')
        plt.grid(True)
        plt.legend()
        
        # Error in control over time
        plt.subplot(3, 1, 3)
        for i in range(N_VALID):
            error_u = self.u_opt_batch[i].flatten() - self.u_sim_batch[i].flatten()
            plt.step(t_u, error_u, where='post', alpha=0.3, color='C4')
        plt.step([], [], where='post', label=f'Avg RMSE={avg_rmse_u:.3f}', color='C4')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Control Error')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'Trajectory Errors (Optimal - Learned) ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def plot_costs(self):
        """Plot cost comparison."""
        N_VALID = len(self.valid_indices)
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
        # Add diagonal line (perfect match)
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
    
    def show_plots(self):
        """Display all plots."""
        plt.show()


if __name__ == "__main__":
    # Create comparison object
    comparison = InvertedPendulumMPCComparison(
        layer_sizes=[2, 20, 1],
        beta=15.0,
        horizon=10
    )
    
    # Generate test states
    comparison.generate_test_states(n_test=200, seed=42)
    
    # Run comparison
    comparison.run_comparison(wait_for_input=True)
    
    # Print results
    comparison.print_results(threshold_rmse=0.01)
    
    # Generate plots
    comparison.plot_trajectories()
    comparison.plot_policy()
    # comparison.plot_policy_3d()
    comparison.plot_errors()
    comparison.plot_costs()
    comparison.show_plots()
