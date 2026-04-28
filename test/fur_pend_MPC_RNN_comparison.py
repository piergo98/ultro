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

from models.furuta_pendulum import FurutaPendulum
from models.complementarity_RNN import ComplementarityRNN


def get_model_name(hidden_sizes):
    """Generate model name from layer sizes.
    
    Parameters
    ----------
    hidden_sizes : list of int
        List containing the sizes of each layer in the network.
    
    Returns
    -------
    model_name : str
        Model name in format like '2x6x6x1'
    """
    return 'x'.join(map(str, hidden_sizes))


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

class FurutaPendulumMPC_RNNComparison:
    """Class for comparing optimal MPC with learned neural network policy for furuta pendulum system."""
    
    def __init__(self, hidden_sizes=[10], beta=0.5, horizon=20, complementarity=False, model_dir=None):
        """Initialize the MPC comparison.
        
        Parameters
        ----------
        hidden_sizes : list of int
            List of hidden layer sizes for the RNN approximator.
        beta : float
            Softplus beta parameter.
        horizon : int
            MPC prediction horizon length.
        complementarity : bool
            Whether to include complementarity constraints in the MPC formulation.
        model_dir : Path or None
            Directory containing trained model parameters.
        """
        print("Setting up the MPC problem for testing...")
        start_time = time.time()
        
        # Store parameters
        self.hidden_sizes = hidden_sizes
        self.beta = beta
        self.N = horizon
        self.complementarity = complementarity
        self.model_dir = model_dir if model_dir is not None else Path(__file__).parent.parent / "models_nn" / "rnn_furuta_pendulum"
        
        # Initialize furuta pendulum model
        self.furuta_pendulum = FurutaPendulum(dt=0.05, sym_type='SX')
        
        # Declare variables
        self.NX = 4      # state dimension (theta_1, omega_1, theta_2, omega_2)
        self.NU = 1      # control dimension
        
        # Set up neural network
        self._setup_network()
        
        # Storage for results
        self.X_test = None
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
        
        # Generate test states
        self.generate_test_states(n_test=100, seed=42, alpha=0.5)
        
        if self.X_test is None:
            raise ValueError("Generate test states first using generate_test_states()")
        
        if not hasattr(self, 'params_init_vec'):
            self.load_learned_params()
            input("Press Enter to start testing...")
    
    def _setup_network(self):
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
    
    def generate_test_states(self, n_test=20, seed=42, alpha=0.5, add_extreme=False):
        """Generate random initial states for testing.
        
        Parameters
        ----------
        n_test : int
            Number of test cases.
        seed : int
            Random seed for reproducibility.
        alpha : float
            Scaling factor to ensure test states are within training distribution.
        """
        # self.X_test = np.zeros((self.NX, n_test))
        if add_extreme:
            extreme_x0 = self.furuta_pendulum.solve_extreme_x0(N=self.N, plot_results=False)
            # Consider also the mirrored initial states to get more coverage
            extreme_x0 += [np.array([-x[0], -x[1]]) for x in extreme_x0]
            self.X_test = np.array(extreme_x0).T
            # Generate the rest of the initial states randomly
            num_random = n_test - len(extreme_x0)
        else:
            self.X_test = np.zeros((self.NX, 0))
            num_random = n_test
        np.random.seed(seed)
        self.theta_1_test_bound = np.pi
        
        X_random = np.random.uniform(
            low=np.array([-self.theta_1_test_bound, 0, 0, 0]).reshape(-1, 1),
            high=np.array([self.theta_1_test_bound, 0, 0, 0]).reshape(-1, 1),
            size=(self.NX, num_random)
        )
        self.X_test = np.hstack((self.X_test, X_random))
        # self.X_test[:, 0] = np.random.uniform(-self.theta_test_bound, self.theta_test_bound, n_test)
        # self.X_test[:, 1] = np.random.uniform(-self.omega_test_bound, self.omega_test_bound, n_test)
        # print(self.X_test[:, :5])
        # raise NotImplementedError("Initial state generation complete. Call setup_optimization() to set up the optimization problem.")
        print(f"Generated {n_test} uniformly sampled initial states")
        
    def find_latest_params(self, model_dir, model_name, extension="yaml"):
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
        if self.complementarity:
            pattern = f"optimal_params_fp_cc_{model_name}_*.{extension}"
        else:
            pattern = f"optimal_params_fp_{model_name}_*.{extension}"
        files = list(model_dir.glob(pattern))
        if not files:
            return None
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return latest_file
    
    def load_learned_params(self):
        """Load parameters for the learned policy."""
        model_name = get_model_name(self.hidden_sizes)
        params_file = self.find_latest_params(self.model_dir, model_name, "yaml")
        # params_file = "/home/pietro/data-driven/freiburg_stuff/ultro/models_nn/rnn_inverted_pendulum/optimal_params_ip_cc_6x6_2026-03-27_18-42-41.yaml"
        if params_file is None:
            raise FileNotFoundError(f"No parameter files found for model {model_name} in {self.model_dir}")
        self.params_init_vec = load_params(params_file)
        print(f"Loaded parameters from {params_file}")
    
    def run_open_loop_comparison(self):
        """Run the comparison between optimal MPC and learned policy.
        
        Parameters
        ----------
        wait_for_input : bool
            Whether to wait for user input before starting.
        """
        
        N_TEST = self.X_test.shape[1]
        print(f"Testing {N_TEST} initial states...")
        
        # Clear previous results
        self.u_opt_batch = []
        self.u_sim_batch = []
        self.rmse_u_batch = []
        self.cost_opt_batch = []
        self.cost_sim_batch = []
        self.valid_indices = []
        
        # Solve MPC and simulate learned policy for each initial state
        for i in range(N_TEST):
            x0 = self.X_test[:, i]
            
            # Solve the MPC NLP
            u_opt = self.furuta_pendulum.solve_MPC(x0, ret_seq=True)
            
            # # Extract the optimal solution
            # w_opt = solution['x'].full().flatten()
            if u_opt is None:
                print(f"Test case {i} failed to solve MPC. Skipping this case.")
                continue
            
            # # Extract optimal state and control trajectories
            # _, u_opt = self.extract_traj(w_opt)
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
        
    def run_closed_loop_comparison(self, Nsim=60):
        """Run a closed-loop comparison between optimal MPC and learned policy.
        
        Parameters
        ----------
        Nsim : int
            Number of simulation steps.
        wait_for_input : bool
            Whether to wait for user input before starting.
        """
        if self.X_test is None:
            raise ValueError("Generate test states first using generate_test_states()")
        
        if not hasattr(self, 'params_init_vec'):
            self.load_learned_params()
        
        N_TEST = self.X_test.shape[1]
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
        for i in range(N_TEST):
            x0 = self.X_test[:, i]

            # Simulate the learned policy in closed loop for the same initial state
            x_sim = np.zeros((self.NX, Nsim + 1))
            u_sim = np.zeros((self.NU, Nsim))
            x_sim[:, 0] = x0
            for k in range(Nsim):
                u_next = self.net_fcn(x_sim[:, k], self.params_init_vec).full().flatten()[0]
                u_sim[:, k] = u_next
                x_next = self.furuta_pendulum.step(x_sim[:, k], u_next)
                x_sim[:, k + 1] = np.array(x_next.full()).flatten()
                
            # Simulate closed-loop optimal MPC trajectory for the same initial state
            print(f"x0: {x0}")
            x_opt_sim, u_opt_sim, skip = self.furuta_pendulum.close_loop_simulation(
                x0,
                Nsim,
                plot_results=False,
                warm_start=True
            )
            if skip:
                print(f"Skipping trajectory for initial state {x0} due to MPC failure.")
                continue
            
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
        
    def plot_closed_loop_trajectories(self, Nsim):
        """Plot closed-loop trajectories for optimal MPC vs learned policy."""
        N_VALID = len(self.valid_indices)
        print(f"Plotting closed-loop trajectories for {N_VALID} valid test cases...")
        t = self.furuta_pendulum.dt * np.arange(Nsim + 1)
        
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

        state_labels = [
            r'$\theta_1$ (rad)',
            r'$\omega_1$ (rad/s)',
            r'$\theta_2$ (rad)',
            r'$\omega_2$ (rad/s)',
        ]
        state_refs = [np.pi, 0.0, np.pi, 0.0]

        for s in range(self.NX):
            ax = axes[s]
            for i in range(N_VALID):
                ax.plot(t, self.x_opt_batch[i][s, :], color='C0', alpha=0.3)
                ax.plot(t, self.x_sim_batch[i][s, :], color='C1', linestyle='--', alpha=0.3)
            ax.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
            ax.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
            ax.axhline(state_refs[s], color='k', linestyle=':', alpha=0.4)
            ax.set_ylabel(state_labels[s])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        ax_u = axes[4]
        for i in range(N_VALID):
            ax_u.plot(t[:-1], self.u_opt_batch[i].flatten(), color='C0', alpha=0.3)
            ax_u.plot(t[:-1], self.u_sim_batch[i].flatten(), color='C1', linestyle='--', alpha=0.3)
        ax_u.plot([], [], color='C0', label='Optimal MPC', linewidth=2)
        ax_u.plot([], [], color='C1', linestyle='--', label='Learned Policy', linewidth=2)
        ax_u.axhline(0.0, color='k', linestyle=':', alpha=0.4)
        ax_u.set_xlabel('Time (s)')
        ax_u.set_ylabel('Torque (Nm)')
        ax_u.grid(True, alpha=0.3)
        ax_u.legend(loc='upper right')

        fig.suptitle(f'Furuta Closed-Loop: MPC vs Learned Policy ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    def plot_controls(self):
        """Plot optimal vs learned trajectories."""
        N_VALID = len(self.valid_indices)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_u = self.furuta_pendulum.dt * np.arange(self.N)
        
        # plt.figure(figsize=(12, 10))
        
        # Control over time
        for i in range(N_VALID):
            plt.plot(t_u, self.u_opt_batch[i].flatten(), alpha=0.3, color='C2')
            plt.plot(t_u, self.u_sim_batch[i].flatten(), alpha=0.3, color='C3')
        plt.step([], [], where='post', label=f'Optimal (Avg RMSE={avg_rmse_u:.3f})', color='C2')
        plt.step([], [], where='post', label='Learned', color='C3')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input')
        plt.grid(True)
        plt.legend()
        
        plt.suptitle(f'Optimal vs Learned Policy ({N_VALID} Valid Test Cases)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def plot_policy(self):
        """Plot Furuta policy slice over pendulum angle theta_2.

        The slice is evaluated around the upright equilibrium:
        x = [theta_1=pi, omega_1=0, theta_2 in grid, omega_2=0].
        """
        theta_1_bound = np.pi * 1.5
        grid = np.linspace(-theta_1_bound, theta_1_bound, 80)

        # Compute optimal and learned control on the slice
        U_opt_grid = np.zeros(grid.shape[0])
        U_learned_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            x_i = np.array([grid[i], 0.0, 0.0, 0.0])
            
            # Solve MPC for this state to get optimal control
            U_opt_grid[i] = self.furuta_pendulum.solve_MPC(x_i, ret_seq=False)

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i, self.params_init_vec).full().flatten()[0]
            

        # Reshape for plotting
        U_opt_grid = U_opt_grid.reshape(grid.shape)
        U_learned_grid = U_learned_grid.reshape(grid.shape)
        
        # Compute errors
        U_error_learned = np.abs(U_opt_grid - U_learned_grid)

        fig = plt.figure(figsize=(12, 8))
        
        # First row: Optimal and learned policies
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(grid, U_opt_grid.flatten(), alpha=0.7, color='C0', linewidth=2, label='Optimal MPC')
        ax1.set_title('Optimal Policy Slice')
        ax1.set_xlabel(r'$\theta_1$ (rad)')
        ax1.set_ylabel('Control Torque (Nm)')
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(grid, U_learned_grid.flatten(), alpha=0.7, color='C1', linewidth=2, label='Learned NN')
        ax2.set_title('Learned Policy Slice')
        ax2.set_xlabel(r'$\theta_1$ (rad)')
        ax2.set_ylabel('Control Torque (Nm)')
        ax2.grid(True)
        ax2.legend()
        
        # Second row: Comparison and errors
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(grid, U_opt_grid.flatten(), alpha=0.7, color='C0', linewidth=2, label='Optimal MPC')
        ax3.plot(grid, U_learned_grid.flatten(), alpha=0.7, color='C1', linewidth=2, linestyle='--', label='Learned NN')
        ax3.set_title('Policy Comparison')
        ax3.set_xlabel(r'$\theta_1$ (rad)')
        ax3.set_ylabel('Control Torque (Nm)')
        ax3.grid(True)
        ax3.legend()
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.semilogy(grid, U_error_learned.flatten(), alpha=0.7, color='C2', linewidth=2, label='Learned error')
        ax4.set_title('|Optimal - Learned| (log scale)')
        ax4.set_xlabel(r'$\theta_1$ (rad)')
        ax4.set_ylabel('|Error| (Nm)')
        ax4.grid(True, which='both')
        ax4.legend()

        fig.suptitle(r'Furuta Policy Slice at $\theta_2=0,\omega_1=0,\omega_2=0$')
        plt.tight_layout()
    
    def plot_policy_3d(self, elev=30, azim=-60):
        """3D surface plot of Furuta control policy slice.

        Parameters
        ----------
        elev : float
            Elevation angle for 3D view.
        azim : float
            Azimuth angle for 3D view.
        """
        # Create a 2D slice over (theta_2, omega_2) around upright equilibrium
        theta_grid = np.linspace(-np.pi, np.pi, 50)
        omega_grid = np.linspace(-12.0, 12.0, 50)
        Theta, Omega = np.meshgrid(theta_grid, omega_grid)
        X_grid = np.vstack([
            np.full(Theta.size, np.pi),
            np.zeros(Theta.size),
            Theta.flatten(),
            Omega.flatten(),
        ])

        # Compute optimal control and learned control on the grid
        U_opt_grid = np.zeros(X_grid.shape[1])
        U_learned_grid = np.zeros(X_grid.shape[1])

        for i in range(X_grid.shape[1]):
            x_i = X_grid[:, i]
            # Solve MPC for this state to get optimal control
            U_opt_grid[i] = self.furuta_pendulum.solve_MPC(x_i, ret_seq=False)

            # Compute learned control
            U_learned_grid[i] = self.net_fcn(x_i, self.params_init_vec).full().flatten()[0]

        # Reshape for plotting
        U_opt_grid = U_opt_grid.reshape(Theta.shape)
        U_learned_grid = U_learned_grid.reshape(Theta.shape)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(Theta, Omega, U_opt_grid, cmap='viridis', rcount=50, ccount=50, linewidth=0, antialiased=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.6)
        ax1.set_title(r'Optimal MPC Slice ($\theta_1=\pi,\omega_1=0$)')
        ax1.set_xlabel(r'$\theta_2$ (rad)')
        ax1.set_ylabel(r'$\omega_2$ (rad/s)')
        ax1.set_zlabel('Torque (Nm)')
        ax1.view_init(elev=elev, azim=azim)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(Theta, Omega, U_learned_grid, cmap='viridis', rcount=50, ccount=50, linewidth=0, antialiased=True)
        fig.colorbar(surf2, ax=ax2, shrink=0.6)
        ax2.set_title(r'Learned Policy Slice ($\theta_1=\pi,\omega_1=0$)')
        ax2.set_xlabel(r'$\theta_2$ (rad)')
        ax2.set_ylabel(r'$\omega_2$ (rad/s)')
        ax2.set_zlabel('Torque (Nm)')
        ax2.view_init(elev=elev, azim=azim)

        plt.tight_layout()

    def plot_errors(self):
        """Plot trajectory errors."""
        N_VALID = len(self.valid_indices)
        avg_rmse_u = np.mean(self.rmse_u_batch)
        
        t_u = np.arange(self.N)
        
        plt.figure()
        
        # Error in control over time
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
    comparison = FurutaPendulumMPC_RNNComparison(
        hidden_sizes=[6, 6],
        horizon=12,
        complementarity=True
    )
    
    # Run comparison
    comparison.run_open_loop_comparison()
    
    # Print results
    # comparison.print_results(threshold_rmse=0.01)
    
    # Generate plots
    comparison.plot_controls()
    # comparison.plot_policy()
    # # comparison.plot_policy_3d()
    # comparison.plot_errors()
    # # comparison.plot_costs()
    comparison.run_closed_loop_comparison(Nsim=30)
    comparison.show_plots()
