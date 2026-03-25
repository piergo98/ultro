from datetime import datetime
import json
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are

from csnn import set_sym_type, Linear, Sequential, ReLU, Softplus
from models.cart_pole import CartPole


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
    pattern = f"optimal_params_cp_{model_name}_*.{extension}"
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


class CartPoleMPCComparison:
    """Class for comparing optimal MPC with learned neural network policy for cart-pole system."""
    
    def __init__(self, layer_sizes=[4, 20, 1], beta=0.5, horizon=20, model_dir=None, wait_for_input=True):
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
        wait_for_input : bool
            If True, wait for user input before starting testing.
        """
        print("Setting up the MPC problem for testing...")
        start_time = time.time()
        
        # Store parameters
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.N = horizon
        self.model_dir = model_dir if model_dir is not None else Path(__file__).parent.parent / "models_nn" / "cart_pole"
        
        # Initialize cart-pole model
        self.cart_pole = CartPole(sym_type='SX')
        
        # Declare variables
        self.NX = 4      # state dimension (position, velocity, angle, angular velocity)
        self.NU = 1      # control dimension
        
        # Set up neural network
        self._setup_network()
        
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
        
        # if self.initial_states is None:
        #     raise ValueError("Generate test states first using generate_test_states()")
        
        if not hasattr(self, 'params_init_vec'):
            self.load_learned_params()
        
        if wait_for_input:
            input("Press Enter to start testing...")
        
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
        self.p_test_bound = self.cart_pole.p_bound * alpha
        self.v_test_bound = self.cart_pole.v_bound * alpha
        self.theta_test_bound = self.cart_pole.theta_bound * alpha
        self.omega_test_bound = self.cart_pole.omega_bound * alpha
        
        self.initial_states = np.zeros((n_test, self.NX))
        self.initial_states[:, 0] = np.random.uniform(-self.p_test_bound, self.p_test_bound, n_test)
        self.initial_states[:, 1] = np.random.uniform(-self.v_test_bound, self.v_test_bound, n_test)
        self.initial_states[:, 2] = np.random.uniform(-self.theta_test_bound, self.theta_test_bound, n_test)
        self.initial_states[:, 3] = np.random.uniform(-self.omega_test_bound, self.omega_test_bound, n_test)
        
        print(f"Generated {n_test} uniformly sampled initial states")
    
    def load_learned_params(self):
        """Load parameters for the learned policy."""
        model_name = get_model_name(self.layer_sizes)
        params_file = find_latest_params(self.model_dir, model_name, "yaml")
        params_file = "/home/pietro/data-driven/freiburg_stuff/ultro/models_nn/cart_pole/optimal_params_cp_4x30x30x20_2026-03-10_13-50-47.yaml"
        if params_file is None:
            raise FileNotFoundError(f"No parameter files found for model {model_name} in {self.model_dir}")
        
        # Load parameters and metadata
        param_data = load_params(params_file, return_metadata=True)
        self.params_init_vec = param_data['params']
        
        # Load Q and R weights from file
        if param_data['q_weights'] is not None:
            print(ca.DM(param_data['q_weights']).shape)
            self.Q = ca.DM(param_data['q_weights']).reshape((self.NX, self.NX))
            print(f"Loaded Q weights from file: {param_data['q_weights']}")
        
        if param_data['r_weight'] is not None:
            self.R = ca.DM([param_data['r_weight']]).reshape((self.NU, self.NU))
            print(f"Loaded R weight from file: {param_data['r_weight']}")
        
        print(f"Loaded parameters from {params_file}")
    
    def run_open_loop_comparison(self):
        """Run the comparison between optimal MPC and learned policy."""
        
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
            
            # Solve the MPC NLP
            u_opt = self.cart_pole.solve_MPC(x0, ret_seq=True)
            
            # Extract optimal state and control trajectories
            u_opt = np.asarray(u_opt)

            # Simulate the learned policy for the same initial state
            u_sim = np.zeros((self.NU, self.N))
            u_sim = self.net_fcn(x0, self.params_init_vec).full().flatten().reshape(self.NU, self.N)
            
            # Store results for valid test case
            self.u_opt_batch.append(u_opt)
            self.u_sim_batch.append(u_sim)
            self.valid_indices.append(i)
            
            # Compute RMSE for states and control
            rmse_u = np.sqrt(np.mean((u_opt - u_sim) ** 2))
            self.rmse_u_batch.append(rmse_u)
            
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{N_TEST} test cases")
        
        # Report on valid test cases
        N_VALID = len(self.valid_indices)
        N_INVALID = N_TEST - N_VALID
        if N_INVALID > 0:
            print(f"\n{N_INVALID} test case(s) were skipped due to NaN values.")
        print(f"Analyzing {N_VALID} valid test cases...\n")
        
    def run_closed_loop_comparison(self, Nsim=60):
        """Run a closed-loop comparison between optimal MPC and learned policy.
        
        Parameters
        ----------
        Nsim : int
            Number of simulation steps.
        wait_for_input : bool
            Whether to wait for user input before starting.
        """
        if self.initial_states is None:
            raise ValueError("Generate test states first using generate_test_states()")
        
        if not hasattr(self, 'params_init_vec'):
            self.load_learned_params()
        
        N_TEST = len(self.initial_states)
        print(f"Running closed-loop comparison for {N_TEST} initial states...")
        
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
        
        # Run closed-loop simulation for each initial state
        for i, x0 in enumerate(self.initial_states):
            x0_list = x0.tolist()

            # Simulate the learned policy in closed loop for the same initial state
            x_sim = np.zeros((self.NX, Nsim + 1))
            u_sim = np.zeros((self.NU, Nsim))
            x_sim[:, 0] = x0
            for k in range(Nsim):
                u_next = self.net_fcn(x_sim[:, k], self.params_init_vec).full().flatten()[0]
                u_sim[:, k] = u_next
                x_next = self.cart_pole.step(x_sim[:, k], u_next)
                x_sim[:, k + 1] = np.array(x_next.full()).flatten()
                
            # Simulate closed-loop optimal MPC trajectory for the same initial state
            x_opt_sim, u_opt_sim = self.cart_pole.close_loop_simulation(
                x0,
                Nsim,
                plot_results=False
            )
            
            # Check for NaN values in trajectories
            if np.isnan(x_sim).any() or np.isnan(u_sim).any():
                print(f"Test case {i} contains NaN values. Skipping this case.")
                continue
            
            # Store results
            self.x_opt_batch.append(x_opt_sim.T)
            self.u_opt_batch.append(u_opt_sim.T)
            self.x_sim_batch.append(x_sim)
            self.u_sim_batch.append(u_sim)
            self.valid_indices.append(i)
            
        # Report on valid test cases
        N_VALID = len(self.valid_indices)
        N_INVALID = N_TEST - N_VALID
        if N_INVALID > 0:
            print(f"\n{N_INVALID} test case(s) were skipped due to NaN values.")
        print(f"Analyzing {N_VALID} valid test cases...\n")
            
        self.plot_closed_loop_trajectories(Nsim)
    
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
        print(f"  State 1 (position): {num_above_threshold_states[0]}")
        print(f"  State 2 (velocity): {num_above_threshold_states[1]}")
        print(f"  State 3 (angle): {num_above_threshold_states[2]}")
        print(f"  State 4 (angular velocity): {num_above_threshold_states[3]}")
        print(f"  Control: {num_above_threshold_u}")
        
        # Compute average RMSE over all test cases
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        print(f"\nAverage RMSE over {N_VALID} valid test cases:")
        print(f"  State 1 (position): {avg_rmse_states[0]:.4f}")
        print(f"  State 2 (velocity): {avg_rmse_states[1]:.4f}")
        print(f"  State 3 (angle): {avg_rmse_states[2]:.4f}")
        print(f"  State 4 (angular velocity): {avg_rmse_states[3]:.4f}")
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
        
    def plot_closed_loop_trajectories(self, Nsim):
        """Plot closed-loop trajectories for optimal MPC vs learned policy."""
        N_VALID = len(self.valid_indices)
        print(f"Plotting closed-loop trajectories for {N_VALID} valid test cases...")
        t = self.cart_pole.dt * np.arange(Nsim + 1)
        
        fig = plt.figure(figsize=(12, 8))
        
        # Plot position trajectory
        ax1 = plt.subplot(3, 2, 1)
        for i in range(N_VALID):
            ax1.plot(t, self.x_opt_batch[i][0, :], color='C0', alpha=0.3)
            ax1.plot(t, self.x_sim_batch[i][0, :], color='C1', linestyle='--', alpha=0.3)
        ax1.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
        ax1.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
        ax1.set_ylabel('Position (m)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot velocity trajectory
        ax2 = plt.subplot(3, 2, 2)
        for i in range(N_VALID):
            ax2.plot(t, self.x_opt_batch[i][1, :], color='C0', alpha=0.3)
            ax2.plot(t, self.x_sim_batch[i][1, :], color='C1', linestyle='--', alpha=0.3)
        ax2.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
        ax2.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True)
        ax2.legend()
        
        # Plot angle trajectory
        ax3 = plt.subplot(3, 2, 3)
        for i in range(N_VALID):
            ax3.plot(t, self.x_opt_batch[i][2, :], color='C0', alpha=0.3)
            ax3.plot(t, self.x_sim_batch[i][2, :], color='C1', linestyle='--', alpha=0.3)
        ax3.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
        ax3.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
        ax3.set_ylabel('Angle (rad)')
        ax3.grid(True)
        ax3.legend()
        
        # Plot angular velocity trajectory
        ax4 = plt.subplot(3, 2, 4)
        for i in range(N_VALID):
            ax4.plot(t, self.x_opt_batch[i][3, :], color='C0', alpha=0.3)
            ax4.plot(t, self.x_sim_batch[i][3, :], color='C1', linestyle='--', alpha=0.3)
        ax4.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
        ax4.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
        ax4.set_ylabel('Angular Velocity (rad/s)')
        ax4.grid(True)
        ax4.legend()
        
        # Plot control trajectories across the entire row
        ax5 = plt.subplot(3, 1, 3)
        for i in range(N_VALID):
            ax5.plot(t[:-1], self.u_opt_batch[i].flatten(), color='C0', alpha=0.3)
            ax5.plot(t[:-1], self.u_sim_batch[i].flatten(), color='C1', linestyle='--', alpha=0.3)
        ax5.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
        ax5.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Control Input (Nm)')
        ax5.grid(True)
        ax5.legend()
        
        fig.suptitle(f'Closed-Loop Trajectories: Optimal MPC vs Learned Policy')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def plot_controls(self):
        """Plot optimal vs learned trajectories."""
        N_VALID = len(self.valid_indices)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_u = self.cart_pole.dt * np.arange(self.N)
        
        # plt.figure(figsize=(12, 10))
        
        # Control over time
        for i in range(N_VALID):
            plt.step(t_u, self.u_opt_batch[i].flatten(), where='post', alpha=0.3, color='C2')
            plt.step(t_u, self.u_sim_batch[i].flatten(), where='post', alpha=0.3, color='C3')
        plt.step([], [], where='post', label=f'Optimal (Avg RMSE={avg_rmse_u:.3f})', color='C2')
        plt.step([], [], where='post', label='Learned', color='C3')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'Open Loop Optimal vs Learned Policy ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def plot_errors(self):
        """Plot trajectory errors."""
        N_VALID = len(self.valid_indices)
        avg_rmse_states = np.mean(self.rmse_states_batch, axis=0)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_x = np.arange(self.N + 1)
        t_u = np.arange(self.N)
        
        plt.figure(figsize=(12, 10))
        
        # Error in state 1 (position) over time
        plt.subplot(5, 1, 1)
        for i in range(N_VALID):
            error_x1 = self.x_opt_batch[i][0, :] - self.x_sim_batch[i][0, :]
            plt.plot(t_x, error_x1, '-', alpha=0.3, color='C0')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[0]:.3f}', color='C0')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Position Error')
        plt.grid(True)
        plt.legend()
        
        # Error in state 2 (velocity) over time
        plt.subplot(5, 1, 2)
        for i in range(N_VALID):
            error_x2 = self.x_opt_batch[i][1, :] - self.x_sim_batch[i][1, :]
            plt.plot(t_x, error_x2, '-', alpha=0.3, color='C1')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[1]:.3f}', color='C1')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Velocity Error')
        plt.grid(True)
        plt.legend()
        
        # Error in state 3 (angle) over time
        plt.subplot(5, 1, 3)
        for i in range(N_VALID):
            error_x3 = self.x_opt_batch[i][2, :] - self.x_sim_batch[i][2, :]
            plt.plot(t_x, error_x3, '-', alpha=0.3, color='C2')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[2]:.3f}', color='C2')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Angle Error')
        plt.grid(True)
        plt.legend()
        
        # Error in state 4 (angular velocity) over time
        plt.subplot(5, 1, 4)
        for i in range(N_VALID):
            error_x4 = self.x_opt_batch[i][3, :] - self.x_sim_batch[i][3, :]
            plt.plot(t_x, error_x4, '-', alpha=0.3, color='C3')
        plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[3]:.3f}', color='C3')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Angular Velocity Error')
        plt.grid(True)
        plt.legend()
        
        # Error in control over time
        plt.subplot(5, 1, 5)
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
    
    def plot_policy(self):
        """Plot learned policy vs optimal control vs LQR.
        
        Plots control as a function of angle, with other states fixed at zero
        (position=0, velocity=0, angular_velocity=0).
        """
        # Create a grid of position for plotting the policy
        grid = np.linspace(-self.cart_pole.theta_bound, self.cart_pole.theta_bound, 50)

        # Compute optimal control, learned control, and LQR control on the grid
        U_opt_grid = np.zeros(grid.shape[0])
        U_learned_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            # State: [position, velocity, angle, angular_velocity]
            x_i = np.array([0.0, 0.0, grid[i], 0.0])
            
            # Solve MPC for this state to get optimal control
            U_opt_grid[i] = self.cart_pole.solve_MPC(x_i, ret_seq=False)

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i.tolist(), self.params_init_vec).full().flatten()[0]

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i.tolist(), self.params_init_vec).full().flatten()[0]

        # Reshape for plotting
        U_opt_grid = U_opt_grid.reshape(grid.shape)
        U_learned_grid = U_learned_grid.reshape(grid.shape)
        
        # Compute errors
        U_error_learned = np.abs(U_opt_grid - U_learned_grid)

        fig = plt.figure(figsize=(12, 8))
        
        # First row: Optimal, Learned, and LQR policies
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(grid, U_opt_grid.flatten(), alpha=0.7, color='C0', linewidth=2, label='Optimal MPC')
        ax1.set_title('Optimal Control Policy')
        ax1.set_xlabel('Angle (rad)')
        ax1.set_ylabel('Control Torque (Nm)')
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(grid, U_learned_grid.flatten(), alpha=0.7, color='C1', linewidth=2, label='Learned NN')
        ax2.set_title(f'Learned Control Policy')
        ax2.set_xlabel('Angle (rad)')
        ax2.set_ylabel('Control Torque (Nm)')
        ax2.grid(True)
        ax2.legend()
        
        # Second row: Comparison and errors
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(grid, U_opt_grid.flatten(), alpha=0.7, color='C0', linewidth=2, label='Optimal MPC')
        ax3.plot(grid, U_learned_grid.flatten(), alpha=0.7, color='C1', linewidth=2, linestyle='--', label='Learned NN')
        ax3.set_title('All Policies Comparison')
        ax3.set_xlabel('Angle (rad)')
        ax3.set_ylabel('Control Torque (Nm)')
        ax3.grid(True)
        ax3.legend()
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.semilogy(grid, U_error_learned.flatten(), alpha=0.7, color='C2', linewidth=2, label='Learned error')
        ax4.set_title('|Optimal - Learned| (log scale)')
        ax4.set_xlabel('Angle (rad)')
        ax4.set_ylabel('|Error| (Nm)')
        ax4.grid(True, which='both')
        ax4.legend()

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
        theta_grid = np.linspace(-self.cart_pole.theta_bound, self.cart_pole.theta_bound, 50)
        omega_grid = np.linspace(-self.cart_pole.omega_bound, self.cart_pole.omega_bound, 50)
        Theta, Omega = np.meshgrid(theta_grid, omega_grid)
        X_grid = np.vstack([Theta.flatten(), Omega.flatten()])

        # Compute optimal control and learned control on the grid
        U_opt_grid = np.zeros(X_grid.shape[1])
        U_learned_grid = np.zeros(X_grid.shape[1])

        for i in range(X_grid.shape[1]):
            x_i = np.array([0.0, 0.0, X_grid[0, i], X_grid[1, i]])
            # Solve MPC for this state to get optimal control
            U_opt_grid[i] = self.cart_pole.solve_MPC(x_i, ret_seq=False)

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
    
    def show_plots(self):
        """Display all plots."""
        plt.show()


if __name__ == "__main__":
    # Create comparison object
    comparison = CartPoleMPCComparison(
        layer_sizes=[4, 30, 30, 20],
        beta=20.0,
        horizon=20
    )
    
    # Generate test states
    comparison.generate_test_states(n_test=200, seed=36)
    
    # Run comparison
    comparison.run_open_loop_comparison()

    
    # Generate plots
    comparison.plot_controls()
    comparison.plot_policy()
    comparison.run_closed_loop_comparison(Nsim=30)

    comparison.show_plots()
