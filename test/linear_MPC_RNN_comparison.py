from datetime import datetime
import json
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

from csnn import set_sym_type
from models.complementarity_RNN import ComplementarityRNN
from models.linear import LinearSystem


def get_model_name(hidden_sizes):
	"""Generate model name from layer sizes."""
	return "x".join(map(str, hidden_sizes))


def load_params(params_file):
	"""Load optimal parameters from a file."""
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
	if params_init_vec.size < n_param:
		params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
	elif params_init_vec.size > n_param:
		params_init_vec = params_init_vec[:n_param]

	return params_init_vec


class LinearMPC_RNNComparison:
	"""Compare optimal MPC with learned RNN policy for a linear system."""

	def __init__(
		self,
		A,
		B,
		dt=0.1,
		hidden_sizes=[6, 6],
		horizon=10,
		q_weights=None,
		r_weight=1.0,
		complementarity=True,
		model_dir=None,
		x_bounds=None,
		u_bounds=None,
	):
		print("Setting up the MPC comparison for linear system...")
		start_time = time.time()

		self.A = np.array(A)
		self.B = np.array(B)
		self.dt = float(dt)
		self.hidden_sizes = hidden_sizes
		self.N = int(horizon)
		self.complementarity = bool(complementarity)
		self.model_dir = (
			Path(model_dir)
			if model_dir is not None
			else Path(__file__).parent.parent / "models_nn" / "rnn_linear_mpc"
		)

		self.NX = int(self.A.shape[0])
		self.NU = int(self.B.shape[1])

		if q_weights is None:
			q_weights = [1.0] * self.NX
		self.Q = ca.diag(ca.DM(q_weights))
		self.R = ca.diag(ca.DM([r_weight]))
		self.Q_np = np.diag(np.array(q_weights, dtype=float))
		self.R_np = np.diag(np.array([r_weight], dtype=float))
		self.E = solve_discrete_are(self.A, self.B, self.Q_np, self.R_np)

		self.xr = ca.DM.zeros(self.NX, 1)
		self.ur = ca.DM.zeros(self.NU, 1)

		if x_bounds is None:
			x_bounds = np.ones(self.NX)
		if u_bounds is None:
			u_bounds = np.ones(self.NU)
		self.x_bounds = np.array(x_bounds, dtype=float).reshape(-1)
		self.u_bounds = np.array(u_bounds, dtype=float).reshape(-1)
		self.x_min = -self.x_bounds
		self.x_max = self.x_bounds
		self.u_min = -self.u_bounds
		self.u_max = self.u_bounds

		self.linear_sys = LinearSystem(self.A, self.B, dt=self.dt, N=self.N)
		self.linear_sys.define_simple_MPC_control(self.N, self.Q_np, self.R_np)

		self._setup_network()

		self.initial_states = None
		self.x_opt_batch = []
		self.u_opt_batch = []
		self.x_sim_batch = []
		self.u_sim_batch = []
		self.rmse_u_batch = []
		self.cost_opt_batch = []
		self.cost_sim_batch = []
		self.valid_indices = []
  
		# if self.initial_states is None:
		# 	raise ValueError("Generate test states first using generate_test_states()")
		if not hasattr(self, "params_init_vec"):
			self.load_learned_params()

		input("Press Enter to start testing...")

		print(f"Setup completed in {time.time() - start_time:.2f} seconds")

	def _setup_network(self):
		set_sym_type("SX")

		x = ca.SX.sym("x", self.NX)
		x_seq = ca.repmat(x, 1, self.N)

		rnn = ComplementarityRNN(
			input_size=self.NX,
			hidden_size=self.hidden_sizes,
			output_size=self.NU,
			complementarity=False,
			output_bias=True,
		)
		h0 = np.zeros((self.hidden_sizes[0], 1))
		result = rnn.build(x_seq, h0)

		self.params_flattened = result["params_flat"]
		self.n_param = rnn.n_params
		self.cc_vars = []
		self.cc_fcn = None

		output_seq = result["output"]
		output_vec = ca.reshape(output_seq, self.NU * self.N, 1)
		self.net_fcn = ca.Function("net_fcn", [x, self.params_flattened], [output_vec])

		print(f"Number of parameters in the RNN network: {self.n_param}")

	def _eval_net_sequence(self, x0):
		if self.complementarity:
			cc_zero = [np.zeros(var.shape) for var in self.cc_vars]
			u_seq = self.net_fcn(x0, self.params_init_vec, *cc_zero).full().flatten()
		else:
			u_seq = self.net_fcn(x0, self.params_init_vec).full().flatten()
		return u_seq.reshape(self.NU, self.N)

	def generate_test_states(self, n_test=30, seed=42, alpha=0.3, plot_distribution=False, bins=20):
		"""Generate random initial states for testing."""
		np.random.seed(seed)
		bounds = alpha * self.x_bounds
		self.initial_states = np.random.uniform(
			low=-bounds.reshape(1, -1),
			high=bounds.reshape(1, -1),
			size=(n_test, self.NX),
		)
		print(f"Generated {n_test} uniformly sampled initial states")

		if plot_distribution:
			fig, axes = plt.subplots(self.NX, 1, figsize=(8, 2.5 * self.NX), sharex=False)
			if self.NX == 1:
				axes = [axes]
			for i in range(self.NX):
				axes[i].hist(self.initial_states[:, i], bins=bins, color="C0", alpha=0.75)
				axes[i].set_title(f"Train Set Distribution: x[{i}]")
				axes[i].set_xlabel(f"x[{i}]")
				axes[i].set_ylabel("Count")
				axes[i].grid(True, alpha=0.3)
			plt.tight_layout()

	def find_latest_params(self, model_dir, model_name, extension="yaml"):
		if self.complementarity:
			pattern = f"optimal_params_linear_cc_{model_name}_*.{extension}"
		else:
			pattern = f"optimal_params_linear_{model_name}_*.{extension}"
		files = list(Path(model_dir).glob(pattern))
		if not files:
			return None
		latest_file = max(files, key=lambda f: f.stat().st_mtime)
		return latest_file

	def load_learned_params(self):
		model_name = get_model_name(self.hidden_sizes)
		params_file = self.find_latest_params(self.model_dir, model_name, "yaml")
		if params_file is None:
			raise FileNotFoundError(
				f"No parameter files found for model {model_name} in {self.model_dir}"
			)
		self.params_init_vec = load_params(params_file)
		print(f"Loaded parameters from {params_file}")

	def _simulate_open_loop(self, x0, u_seq):
		x_traj = np.zeros((self.NX, self.N + 1))
		x_traj[:, 0] = np.array(x0).flatten()
		for k in range(self.N):
			x_next = self.linear_sys.step(x_traj[:, k], u_seq[:, k]).full().flatten()
			x_traj[:, k + 1] = x_next
		return x_traj

	def _compute_trajectory_cost(self, x_traj, u_traj):
		"""Compute the total cost for a trajectory."""
		cost = 0.0
		for k in range(self.N):
			x_k = x_traj[:, k]
			u_k = u_traj[:, k]
			stage_cost = float(
				((x_k - self.xr.full().flatten()).T @ self.Q.full() @ (x_k - self.xr.full().flatten())
				 + (u_k - self.ur.full().flatten()).T @ self.R.full() @ (u_k - self.ur.full().flatten())).item()
			)
			cost += stage_cost
		x_N = x_traj[:, self.N]
		terminal_cost = float(
			((x_N - self.xr.full().flatten()).T @ self.E @ (x_N - self.xr.full().flatten())).item()
		)
		cost += terminal_cost
		return cost

	def run_open_loop_comparison(self):

		N_TEST = len(self.initial_states)
		print(f"Testing {N_TEST} initial states...")

		self.u_opt_batch = []
		self.u_sim_batch = []
		self.rmse_u_batch = []
		self.cost_opt_batch = []
		self.cost_sim_batch = []
		self.valid_indices = []

		for i, x0 in enumerate(self.initial_states):
			u_opt = np.asarray(self.linear_sys.solve_MPC(x0, ret_seq=True))
			u_opt = u_opt.reshape(self.NU, self.N)

			u_sim = self._eval_net_sequence(x0)
			x_opt = self._simulate_open_loop(x0, u_opt)
			x_sim = self._simulate_open_loop(x0, u_sim)

			self.u_opt_batch.append(u_opt)
			self.u_sim_batch.append(u_sim)
			self.valid_indices.append(i)

			rmse_u = np.sqrt(np.mean((u_opt - u_sim) ** 2))
			self.rmse_u_batch.append(rmse_u)

			self.cost_opt_batch.append(self._compute_trajectory_cost(x_opt, u_opt))
			self.cost_sim_batch.append(self._compute_trajectory_cost(x_sim, u_sim))

			if (i + 1) % 5 == 0:
				print(f"Completed {i + 1}/{N_TEST} test cases")

	def run_closed_loop_comparison(self, Nsim=40):
		if self.initial_states is None:
			raise ValueError("Generate test states first using generate_test_states()")
		if not hasattr(self, "params_init_vec"):
			self.load_learned_params()

		N_TEST = len(self.initial_states)
		print(f"Running closed-loop comparison for {N_TEST} initial states...")

		self.x_opt_batch = []
		self.u_opt_batch = []
		self.x_sim_batch = []
		self.u_sim_batch = []
		self.valid_indices = []

		for i, x0 in enumerate(self.initial_states):
			x_opt_sim, u_opt_sim = self.linear_sys.close_loop_simulation(
						x0,
						Nsim=Nsim,
						plot_results=False,
					)
			if self.linear_sys.skip:
				print(f"Optimal MPC failed for test case {i+1}. Skipping this case.")
				self.linear_sys.skip = False
				continue
			x_sim = np.zeros((self.NX, Nsim + 1))
			u_sim = np.zeros((self.NU, Nsim))
			x_sim[:, 0] = x0
			for k in range(Nsim):
				u_next = self._eval_net_sequence(x_sim[:, k])[:, 0]
				u_sim[:, k] = u_next
				x_next = self.linear_sys.step(x_sim[:, k], u_next)
				x_sim[:, k + 1] = np.array(x_next.full()).flatten()

			if np.isnan(x_sim).any() or np.isnan(u_sim).any():
				print(f"Test case {i} contains NaN values. Skipping this case.")
				continue

			self.x_opt_batch.append(x_opt_sim.T)
			self.u_opt_batch.append(u_opt_sim.T)
			self.x_sim_batch.append(x_sim)
			self.u_sim_batch.append(u_sim)
			self.valid_indices.append(i)

		self.plot_closed_loop_trajectories(Nsim)

	def print_results(self, threshold_rmse=0.01):
		"""Print comparison results."""
		N_VALID = len(self.valid_indices)
		if N_VALID == 0:
			print("No valid test cases to report.")
			return

		num_above_threshold_u = np.sum(np.array(self.rmse_u_batch) > threshold_rmse)
		print(f"\nNumber of test cases with RMSE above {threshold_rmse}:")
		print(f"  Control: {num_above_threshold_u}")

		avg_rmse_u = np.mean(self.rmse_u_batch)
		print(f"\nAverage RMSE over {N_VALID} valid test cases:")
		print(f"  Control: {avg_rmse_u:.4f}")

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

	def plot_controls(self):
		N_VALID = len(self.valid_indices)
		avg_rmse_u = np.mean(self.rmse_u_batch) if self.rmse_u_batch else 0.0
		t_u = np.arange(self.N) * self.dt

		for i in range(N_VALID):
			plt.plot(t_u, self.u_opt_batch[i].flatten(), alpha=0.3, color="C2")
			plt.plot(t_u, self.u_sim_batch[i].flatten(), alpha=0.3, color="C3")
		plt.plot([], [], label=f"Optimal (Avg RMSE={avg_rmse_u:.3f})", color="C2")
		plt.plot([], [], label="Learned", color="C3")
		plt.xlabel("Time (s)")
		plt.ylabel("Control Input")
		plt.grid(True)
		plt.legend()
		plt.suptitle(f"Optimal vs Learned Policy ({N_VALID} Valid Test Cases)")
		plt.tight_layout(rect=[0, 0, 1, 0.96])

	def plot_closed_loop_trajectories(self, Nsim):
		N_VALID = len(self.valid_indices)
		print(f"Plotting closed-loop trajectories for {N_VALID} valid test cases...")
		t = self.dt * np.arange(Nsim + 1)

		fig = plt.figure(figsize=(12, 8))
		for s in range(self.NX):
			ax = plt.subplot(self.NX + 1, 1, s + 1)
			for i in range(N_VALID):
				ax.plot(t, self.x_opt_batch[i][s, :], color="C0", alpha=0.3)
				ax.plot(t, self.x_sim_batch[i][s, :], color="C1", linestyle="--", alpha=0.3)
			ax.plot([], [], color="C0", label="Optimal MPC", linewidth=2)
			ax.plot([], [], color="C1", linestyle="--", label="Learned Policy", linewidth=2)
			ax.set_ylabel(f"x[{s}]")
			ax.grid(True)
			ax.legend()

		ax_u = plt.subplot(self.NX + 1, 1, self.NX + 1)
		for i in range(N_VALID):
			ax_u.plot(t[:-1], self.u_opt_batch[i].flatten(), color="C0", alpha=0.3)
			ax_u.plot(t[:-1], self.u_sim_batch[i].flatten(), color="C1", linestyle="--", alpha=0.3)
		ax_u.plot([], [], color="C0", label="Optimal MPC", linewidth=2)
		ax_u.plot([], [], color="C1", linestyle="--", label="Learned Policy", linewidth=2)
		ax_u.set_xlabel("Time (s)")
		ax_u.set_ylabel("u")
		ax_u.grid(True)
		ax_u.legend()

		fig.suptitle(f"Closed-Loop Trajectories ({N_VALID} Valid Test Cases)")
		plt.tight_layout(rect=[0, 0, 1, 0.96])

	def plot_costs(self):
		N_VALID = len(self.valid_indices)
		cost_opt_array = np.array(self.cost_opt_batch)
		cost_sim_array = np.array(self.cost_sim_batch)
		cost_diff = cost_sim_array - cost_opt_array
		suboptimality_pct = 100.0 * cost_diff / cost_opt_array

		plt.figure(figsize=(12, 8))
		plt.subplot(2, 2, 1)
		plt.bar(range(N_VALID), cost_opt_array, alpha=0.7, label="Optimal MPC", color="C0")
		plt.bar(range(N_VALID), cost_sim_array, alpha=0.7, label="Learned Policy", color="C1")
		plt.xlabel("Test Case")
		plt.ylabel("Total Cost")
		plt.title("Cost Comparison")
		plt.legend()
		plt.grid(True, alpha=0.3)

		plt.subplot(2, 2, 2)
		plt.bar(range(N_VALID), cost_diff, color="C2", alpha=0.7)
		plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
		plt.xlabel("Test Case")
		plt.ylabel("Cost Difference (Learned - Optimal)")
		plt.title(f"Cost Difference (Avg: {cost_diff.mean():.2f})")
		plt.grid(True, alpha=0.3)

		plt.subplot(2, 2, 3)
		plt.bar(range(N_VALID), suboptimality_pct, color="C3", alpha=0.7)
		plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
		plt.xlabel("Test Case")
		plt.ylabel("Suboptimality (%)")
		plt.title(f"Suboptimality Percentage (Avg: {suboptimality_pct.mean():.2f}%)")
		plt.grid(True, alpha=0.3)

		plt.subplot(2, 2, 4)
		plt.scatter(cost_opt_array, cost_sim_array, alpha=0.6, s=50, color="C4")
		min_cost = min(cost_opt_array.min(), cost_sim_array.min())
		max_cost = max(cost_opt_array.max(), cost_sim_array.max())
		plt.plot([min_cost, max_cost], [min_cost, max_cost], "k--", alpha=0.5)
		plt.xlabel("Optimal MPC Cost")
		plt.ylabel("Learned Policy Cost")
		plt.title("Cost Correlation")
		plt.grid(True, alpha=0.3)
		plt.axis("equal")

		plt.suptitle(f"Cost Function Analysis ({N_VALID} Valid Test Cases)")
		plt.tight_layout(rect=[0, 0, 1, 0.96])

	def show_plots(self):
		plt.show()

	def plot_policy(self, state_index=0, grid_size=50):
		"""Plot learned policy vs optimal MPC as a function of one state dimension."""
		grid = np.linspace(-self.x_bounds[state_index], self.x_bounds[state_index], grid_size)
		U_opt_grid = np.zeros(grid.shape[0])
		U_learned_grid = np.zeros(grid.shape[0])

		for i in range(grid.shape[0]):
			x_i = np.zeros(self.NX)
			x_i[state_index] = grid[i]
			U_opt_grid[i] = self.linear_sys.solve_MPC(x_i, ret_seq=False)
			U_learned_grid[i] = self.net_fcn(x_i, self.params_init_vec).full().flatten()[0]

		U_error_learned = np.abs(U_opt_grid - U_learned_grid)

		fig = plt.figure(figsize=(12, 8))
		ax1 = fig.add_subplot(2, 2, 1)
		ax1.plot(grid, U_opt_grid.flatten(), alpha=0.7, color="C0", linewidth=2, label="Optimal MPC")
		ax1.set_title("Optimal Control Policy")
		ax1.set_xlabel(f"x[{state_index}]")
		ax1.set_ylabel("Control")
		ax1.grid(True)
		ax1.legend()

		ax2 = fig.add_subplot(2, 2, 2)
		ax2.plot(grid, U_learned_grid.flatten(), alpha=0.7, color="C1", linewidth=2, label="Learned NN")
		ax2.set_title("Learned Control Policy")
		ax2.set_xlabel(f"x[{state_index}]")
		ax2.set_ylabel("Control")
		ax2.grid(True)
		ax2.legend()

		ax3 = fig.add_subplot(2, 2, 3)
		ax3.plot(grid, U_opt_grid.flatten(), alpha=0.7, color="C0", linewidth=2, label="Optimal MPC")
		ax3.plot(grid, U_learned_grid.flatten(), alpha=0.7, color="C1", linestyle="--", label="Learned NN")
		ax3.set_title("Policy Comparison")
		ax3.set_xlabel(f"x[{state_index}]")
		ax3.set_ylabel("Control")
		ax3.grid(True)
		ax3.legend()

		ax4 = fig.add_subplot(2, 2, 4)
		ax4.semilogy(grid, U_error_learned.flatten(), alpha=0.7, color="C2", linewidth=2, label="Learned error")
		ax4.set_title("|Optimal - Learned| (log scale)")
		ax4.set_xlabel(f"x[{state_index}]")
		ax4.set_ylabel("|Error|")
		ax4.grid(True, which="both")
		ax4.legend()

		plt.tight_layout()

	def plot_policy_3d(self, i=0, j=1, grid_size=40, elev=30, azim=-60):
		"""3D surface plot of optimal and learned control policies."""
		x_i = np.linspace(-self.x_bounds[i], self.x_bounds[i], grid_size)
		x_j = np.linspace(-self.x_bounds[j], self.x_bounds[j], grid_size)
		X1, X2 = np.meshgrid(x_i, x_j)
		X_grid = np.vstack([X1.flatten(), X2.flatten()])

		U_opt_grid = np.zeros(X_grid.shape[1])
		U_learned_grid = np.zeros(X_grid.shape[1])

		for idx in range(X_grid.shape[1]):
			x_k = np.zeros(self.NX)
			x_k[i] = X_grid[0, idx]
			x_k[j] = X_grid[1, idx]
			U_opt_grid[idx] = self.linear_sys.solve_MPC(x_k, ret_seq=False)
			U_learned_grid[idx] = self.net_fcn(x_k, self.params_init_vec).full().flatten()[0]

		U_opt_grid = U_opt_grid.reshape(X1.shape)
		U_learned_grid = U_learned_grid.reshape(X1.shape)

		fig = plt.figure(figsize=(14, 6))
		ax1 = fig.add_subplot(1, 2, 1, projection="3d")
		surf1 = ax1.plot_surface(X1, X2, U_opt_grid, cmap="viridis", rcount=50, ccount=50, linewidth=0, antialiased=True)
		fig.colorbar(surf1, ax=ax1, shrink=0.6)
		ax1.set_title("Optimal Control Policy (3D)")
		ax1.set_xlabel(f"x[{i}]")
		ax1.set_ylabel(f"x[{j}]")
		ax1.set_zlabel("Control")
		ax1.view_init(elev=elev, azim=azim)

		ax2 = fig.add_subplot(1, 2, 2, projection="3d")
		surf2 = ax2.plot_surface(X1, X2, U_learned_grid, cmap="viridis", rcount=50, ccount=50, linewidth=0, antialiased=True)
		fig.colorbar(surf2, ax=ax2, shrink=0.6)
		ax2.set_title("Learned Policy (3D)")
		ax2.set_xlabel(f"x[{i}]")
		ax2.set_ylabel(f"x[{j}]")
		ax2.set_zlabel("Control")
		ax2.view_init(elev=elev, azim=azim)

		plt.tight_layout()

	def plot_errors(self):
		"""Plot trajectory errors."""
		N_VALID = len(self.valid_indices)
		avg_rmse_u = np.mean(self.rmse_u_batch) if self.rmse_u_batch else 0.0
		t_u = np.arange(self.N) * self.dt

		plt.figure()
		for i in range(N_VALID):
			error_u = self.u_opt_batch[i].flatten() - self.u_sim_batch[i].flatten()
			plt.plot(t_u, error_u, alpha=0.3, color="C4")
		plt.step([], [], where="post", label=f"Avg RMSE={avg_rmse_u:.3f}", color="C4")
		plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
		plt.xlabel("Time (s)")
		plt.ylabel("Control Error")
		plt.grid(True)
		plt.legend()
		plt.suptitle(f"Trajectory Errors (Optimal - Learned) ({N_VALID} Valid Test Cases)")
		plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == "__main__":
	A = [[1.0, 0.1, 0.0, 0.0],
		 [0.0, 0.9818, 0.2673, 0.0],
		 [0.0, 0.0, 1.0, 0.1],
		 [0.0, -0.0455, 3.1182, 1.0]]
	B = [[0.0], [0.1818], [0.0], [0.4546]]

	comparison = LinearMPC_RNNComparison(
		A,
		B,
		dt=0.1,
		hidden_sizes=[6, 6],
		horizon=10,
		complementarity=True,
		x_bounds=[1.0, 1.5, 0.35, 1.0],
		u_bounds=[1.0],
	)

	comparison.generate_test_states(n_test=200, seed=36, alpha=0.3, plot_distribution=False)
	# comparison.run_open_loop_comparison()
	# comparison.plot_controls()
	comparison.plot_policy(state_index=2)
	comparison.run_closed_loop_comparison(Nsim=20)
	comparison.show_plots()
