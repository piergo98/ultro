from datetime import datetime
import gc
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


class LinearMPCSequenceRNN:
	"""MPC sequence approximation with RNN for a linear system.

	Learns a policy that maps the initial state to the full MPC control
	sequence using a RNN parameterization.
	"""

	def __init__(
		self,
		A,
		B,
		dt=0.1,
		hidden_sizes=[6, 6, 6],
		batch_size=60,
		horizon=10,
		q_weights=None,
		r_weight=1.0,
		regularization=1e-4,
		seed=42,
		complementarity_constraints=True,
		tau=1.0,
		model_dir=None,
		x_bounds=None,
		u_bounds=None,
	):
		self.A = np.array(A)
		self.B = np.array(B)
		self.dt = float(dt)

		self.NX = int(self.A.shape[0])
		self.NU = int(self.B.shape[1])
		self.NB = int(batch_size)
		self.N = int(horizon)

		self.hidden_sizes = hidden_sizes
		self.regularization = float(regularization)
		self.seed = int(seed)
		self.complementarity_constraints = bool(complementarity_constraints)
		self.tau = float(tau)

		if q_weights is None:
			q_weights = [1.0] * self.NX
		self.Q = ca.diag(ca.DM(q_weights))
		self.R = ca.diag(ca.DM([r_weight]))

		self.Q_np = np.diag(np.array(q_weights, dtype=float))
		self.R_np = np.diag(np.array([r_weight], dtype=float))

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

		self.x_train_bounds = 0.3 * self.x_bounds

		if model_dir is None:
			self.model_dir = Path(__file__).parent.parent / "models_nn" / "rnn_linear_mpc"
		else:
			self.model_dir = Path(model_dir)
		self.model_dir.mkdir(parents=True, exist_ok=True)
		self.model_name = self.get_model_name()

		self.linear_sys = LinearSystem(self.A, self.B, dt=self.dt, N=self.N)
		self.linear_sys.define_simple_MPC_control(self.N, self.Q_np, self.R_np)

		np.random.seed(self.seed)

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

		if not self.hidden_sizes:
			raise ValueError("hidden_sizes must contain at least one RNN layer size")
		for size in self.hidden_sizes:
			if size <= 0:
				raise ValueError(f"Hidden layer sizes must be positive. Got {self.hidden_sizes}")

		if self.complementarity_constraints:
			print("Setting up complementarity RNN...")
			self.setup_complementarity_network()
		else:
			print("Setting up standard RNN...")
			self.setup_network()

	def get_model_name(self):
		return "x".join(map(str, self.hidden_sizes))

	def setup_network(self):
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
		result = rnn.build(x_seq, h0, tau=self.tau)

		self.params_flattened = result["params_flat"]
		self.n_param = rnn.n_params

		output_seq = result["output"]
		output_vec = ca.reshape(output_seq, self.NU * self.N, 1)
		self.net_fcn = ca.Function("net_fcn", [x, self.params_flattened], [output_vec])

		print(f"Number of parameters in the RNN network: {self.n_param}")

	def setup_complementarity_network(self):
		x = ca.SX.sym("x", self.NX)
		x_seq = ca.repmat(x, 1, self.N)

		rnn = ComplementarityRNN(
			input_size=self.NX,
			hidden_size=self.hidden_sizes,
			output_size=self.NU,
			complementarity=True,
			output_bias=True,
		)
		h0 = np.zeros((self.hidden_sizes[0], 1))
		result = rnn.build(x_seq, h0, tau=self.tau)

		self.params_flattened = result["params_flat"]
		self.cc_vars = result["vars"]
		self.cc_vars_lbw = result["lbw"]
		self.cc_vars_ubw = result["ubw"]
		self.cc_g = result["g"]
		self.cc_lbg = result["lbg"]
		self.cc_ubg = result["ubg"]
		self.n_param = rnn.n_params

		output_seq = result["output"]
		output_vec = ca.reshape(output_seq, self.NU * self.N, 1)
		self.net_fcn = ca.Function(
			"net_fcn", [x, self.params_flattened, *self.cc_vars], [output_vec]
		)
		self.cc_fcn = ca.Function(
			"cc_fcn", [x, self.params_flattened, *self.cc_vars], [ca.vertcat(*self.cc_g)]
		)

		print(f"Number of parameters in the RNN network: {self.n_param}")

	def initialize_parameters(self, params_file=None):
		if params_file is not None:
			try:
				params_init = self.load_params(params_file, self.n_param)
				print(f"Loaded parameters from {params_file}")
				input("Press Enter to continue with these parameters, or Ctrl+C to abort ")
				return params_init
			except (FileNotFoundError, ValueError) as exc:
				print(f"Could not load parameters from {params_file}: {exc}")
				print("Using Kaiming (He) initialization instead.")

		print("Using Kaiming (He) initialization.")
		return self.parameter_initialization_he()

	def generate_initial_states(self):
		low = -self.x_train_bounds.reshape(-1, 1)
		high = self.x_train_bounds.reshape(-1, 1)
		self.X_train = np.random.uniform(low=low, high=high, size=(self.NX, self.NB))

	def compute_open_loop_mpc_trajectory(self, x0):
		u_seq = self.linear_sys.solve_MPC(x0, ret_seq=True)
		u_seq = np.array(u_seq).reshape(self.N, self.NU)
		x_traj = np.zeros((self.N + 1, self.NX))
		x_traj[0] = np.array(x0).flatten()
		for k in range(self.N):
			x_next = self.linear_sys.step(x_traj[k], u_seq[k]).full().flatten()
			x_traj[k + 1] = x_next
		return x_traj, u_seq

	def solve_open_loop_MPC_for_initial_states(self):
		self.initial_trajectories = np.zeros((self.NX, self.N + 1, self.NB))
		self.initial_controls = np.zeros((self.NU, self.N, self.NB))
		for i in range(self.NB):
			x_traj, u_traj = self.compute_open_loop_mpc_trajectory(self.X_train[:, i])
			self.initial_trajectories[:, :, i] = x_traj.T
			self.initial_controls[:, :, i] = u_traj.T

	def setup_optimization(self, params_init=None, warm_start="mpc"):
		print("Setting up optimization for linear MPC approximation...")

		if params_init is None:
			params_init = self.initialize_parameters()

		state_warm_start = None
		control_warm_start = None
		if warm_start == "mpc":
			self.solve_open_loop_MPC_for_initial_states()
			state_warm_start = {}
			control_warm_start = {}
			for i in range(self.NB):
				for k in range(self.N + 1):
					state_warm_start[(i, k)] = self.initial_trajectories[:, k, i]
				for k in range(self.N):
					control_warm_start[(i, k)] = self.initial_controls[:, k, i]

		x = ca.SX.sym("x", self.NX)
		u = ca.SX.sym("u", self.NU)
		x_next = ca.DM(self.A) @ x + ca.DM(self.B) @ u
		l = (x - self.xr).T @ self.Q @ (x - self.xr) + (u - self.ur).T @ self.R @ (u - self.ur)
		f = ca.Function("f", [x, u], [x_next, l], ["x", "u"], ["x_next", "l"])

		E = solve_discrete_are(self.A, self.B, self.Q_np, self.R_np)
		E_dm = ca.DM(E)

		w = []
		w0 = []
		lbw = []
		ubw = []
		J = 0
		g = []
		lbg = []
		ubg = []

		z = ca.SX.sym("z", self.n_param)
		w += [z]
		lbw += [-100.0] * self.n_param
		ubw += [100.0] * self.n_param
		w0 += params_init.tolist()

		for i in range(self.NB):
			x0 = self.X_train[:, i]

			Xk = ca.SX.sym(f"X_{i}_0", self.NX)
			w += [Xk]
			lbw += x0.tolist()
			ubw += x0.tolist()
			if state_warm_start is not None:
				w0 += state_warm_start[(i, 0)].tolist()
			else:
				w0 += x0.tolist()

			cc_step_vars = []
			if self.complementarity_constraints:
				for layer_idx, cc_var in enumerate(self.cc_vars):
					nrow, ncol = cc_var.shape
					y_var = ca.SX.sym(f"Y_{i}_{layer_idx}", nrow, ncol)
					cc_step_vars.append(y_var)
					w += [y_var]
					lbw += [0.0] * (nrow * ncol)
					ubw += [ca.inf] * (nrow * ncol)
					w0 += [0.0] * (nrow * ncol)

			control_vec = []
			for k in range(self.N):
				uk = ca.SX.sym(f"U_{i}_{k}", self.NU)
				w += [uk]
				lbw += self.u_min.tolist()
				ubw += self.u_max.tolist()
				if control_warm_start is not None:
					w0 += control_warm_start[(i, k)].tolist()
				else:
					w0 += [0.0] * self.NU

				x_next_k, qk = f(Xk, uk)

				Xk = ca.SX.sym(f"X_{i}_{k + 1}", self.NX)
				w += [Xk]
				lbw += self.x_min.tolist()
				ubw += self.x_max.tolist()
				if state_warm_start is not None:
					w0 += state_warm_start[(i, k + 1)].tolist()
				else:
					w0 += [0.0] * self.NX

				g += [x_next_k - Xk]
				lbg += [0.0] * self.NX
				ubg += [0.0] * self.NX

				J += qk
				control_vec.append(uk)

			u_i = ca.vertcat(*control_vec)
			if self.complementarity_constraints:
				g += [u_i - self.net_fcn(x0, z, *cc_step_vars)]
				lbg += [0.0] * self.NU * self.N
				ubg += [0.0] * self.NU * self.N

				g += [self.cc_fcn(x0, z, *cc_step_vars)]
				lbg += self.cc_lbg
				ubg += self.cc_ubg
			else:
				g += [u_i - self.net_fcn(x0, z)]
				lbg += [0.0] * self.NU * self.N
				ubg += [0.0] * self.NU * self.N

			J += (Xk - self.xr).T @ E_dm @ (Xk - self.xr)

		J += self.regularization * ca.dot(z, z)

		self.w0 = w0
		self.lbw = lbw
		self.ubw = ubw
		self.lbg = lbg
		self.ubg = ubg

		nlp = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
		print("NLP problem created.")

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
                "mu_strategy": "adaptive",
            }
        }
		self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

	def solve(self):
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
		self.extract_solution()

	def extract_solution(self):
		w_opt = self.solution["x"].full().flatten()

		self.optimal_params = w_opt[: self.n_param]
		self.x_opt = np.zeros((self.NB, self.N + 1, self.NX))
		self.u_opt = np.zeros((self.NB, self.N, self.NU))

		offset = self.n_param
		cc_step_size = 0
		if self.complementarity_constraints:
			cc_step_size = sum(int(var.size1() * var.size2()) for var in self.cc_vars)

		for i in range(self.NB):
			self.x_opt[i, 0, :] = w_opt[offset : offset + self.NX]
			offset += self.NX
			if cc_step_size:
				offset += cc_step_size
			for k in range(self.N):
				self.u_opt[i, k, :] = w_opt[offset : offset + self.NU]
				offset += self.NU
				self.x_opt[i, k + 1, :] = w_opt[offset : offset + self.NX]
				offset += self.NX

		print("\nOptimal solution extraction complete:")
		print(f"  Max parameter magnitude: {np.max(np.abs(self.optimal_params)):.4f}")
		print(f"  Min parameter magnitude: {np.min(np.abs(self.optimal_params)):.4f}")

	def plot_results(self):
		if self.x_opt is None or self.u_opt is None:
			raise RuntimeError("Must call solve() before plot_results()")

		time_x = np.arange(self.N + 1)
		time_u = np.arange(self.N)

		plt.figure(figsize=(12, 8))
		for s in range(self.NX):
			plt.subplot(self.NX + 1, 1, s + 1)
			for i in range(self.NB):
				plt.plot(time_x, self.x_opt[i, :, s], color="gray", alpha=0.25)
			mean_state = self.x_opt[:, :, s].mean(axis=0)
			plt.plot(time_x, mean_state, "k-", linewidth=2)
			plt.ylabel(f"x[{s}]")
			plt.grid()

		plt.subplot(self.NX + 1, 1, self.NX + 1)
		for i in range(self.NB):
			plt.step(time_u, self.u_opt[i, :, 0], where="post", color="gray", alpha=0.25)
		mean_u = self.u_opt.mean(axis=0)
		plt.step(time_u, mean_u[:, 0], where="post", color="k", linewidth=2)
		plt.xlabel("Time Step")
		plt.ylabel("u")
		plt.grid()

		plt.tight_layout()
		plt.show()

	def save_results(self):
		if self.optimal_params is None:
			raise RuntimeError("Must call solve() before save_results()")

		date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		params_dict = {
			"optimal_params": np.asarray(self.optimal_params).tolist(),
			"n_param": int(self.n_param),
			"hidden_sizes": self.hidden_sizes,
			"model_name": self.model_name,
			"batch_size": int(self.NB),
			"tau": float(self.tau),
			"horizon": int(self.N),
			"date": date_str,
		}

		if self.complementarity_constraints:
			yaml_path = self.model_dir / f"optimal_params_linear_cc_{self.model_name}_{date_str}.yaml"
		else:
			yaml_path = self.model_dir / f"optimal_params_linear_{self.model_name}_{date_str}.yaml"
		with yaml_path.open("w") as f:
			yaml.safe_dump(params_dict, f)
		print(f"Saved parameters to {yaml_path}")

	@staticmethod
	def load_params(params_file, n_param):
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
		if params_init_vec.size < n_param:
			params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
		elif params_init_vec.size > n_param:
			params_init_vec = params_init_vec[:n_param]

		return params_init_vec

	def parameter_initialization_he(self):
		return np.random.randn(self.n_param) * 0.01

	def find_latest_params(self, model_dir, model_name, extension="yaml"):
		if self.complementarity_constraints:
			pattern = f"optimal_params_linear_cc_{model_name}_*.{extension}"
		else:
			pattern = f"optimal_params_linear_{model_name}_*.{extension}"
		files = list(model_dir.glob(pattern))
		if not files:
			return None
		latest_file = max(files, key=lambda f: f.stat().st_mtime)
		return latest_file


def main():
	A = [[1.0, 0.1, 0.0, 0.0],
		 [0.0, 0.9818, 0.2673, 0.0],
		 [0.0, 0.0, 1.0, 0.1],
		 [0.0, -0.0455, 3.1182, 1.0]]
	B = [[0.0], [0.1818], [0.0], [0.4546]]

	tau_init = 1.0 * 1e-0
	tau_k = tau_init
	tau_min = 1e-6
	warm_params = None

	while tau_k >= tau_min:
		print(f"Training with tau = {tau_k:.2e}")
		mpc = LinearMPCSequenceRNN(
			A,
			B,
			dt=0.1,
			hidden_sizes=[6, 6],
			batch_size=100,
			horizon=10,
			q_weights=[1.0, 1.0, 1.0, 1.0],
			r_weight=1.0,
			regularization=1e-4,
			seed=42,
			complementarity_constraints=True,
			tau=tau_k,
			x_bounds=[1.0, 1.5, 0.35, 1.0],
			u_bounds=[1.0],
		)

		mpc.generate_initial_states()

		# if tau_k == tau_init:
		# 	params_file = mpc.find_latest_params(mpc.model_dir, mpc.model_name, extension="yaml")
			# warm_params = mpc.initialize_parameters(params_file)

		mpc.setup_optimization(warm_params, warm_start="mpc")
		mpc.solve()
		if mpc.solver.stats()["success"]:
			warm_params = mpc.optimal_params.copy()
			tau_k *= 0.1
		else:
			print("Optimization failed. Stopping training.")
			break

		gc.collect()

	mpc.plot_results()
	mpc.save_results()


if __name__ == "__main__":
	main()
