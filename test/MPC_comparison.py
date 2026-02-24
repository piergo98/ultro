from datetime import datetime
import json
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

from csnn import set_sym_type, Linear, Sequential, ReLU


# create a 2x2 discrete-time block with a complex-conjugate pole pair and a 2x1 input
def make_discrete_conjugate_pair(magnitude=0.95, angle=0.35, B=None):
    # A = magnitude * rotation(angle)
    a = magnitude * np.cos(angle)
    b = magnitude * np.sin(angle)
    A = ca.DM([[a, -b],
               [b,  a]])
    if B is None:
        # choose a simple input that typically yields controllability
        B = ca.DM([[0.0],
                   [1.0]])
    else:
        B = ca.DM(B)
    return A, B

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
    pattern = f"optimal_params_{model_name}_*.{extension}"
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

print("Setting up the MPC problem for testing...")
start_time = time.time()


r = 1.0                # pole magnitude
phi = 0.3              # pole angle (radians)
A, B = make_discrete_conjugate_pair(r, phi)

# Declare variables
NX = 2      # state dimension
NU = 1      # control dimension
N = 15      # horizon length

# Creathe a neural network approximator for the control policy
set_sym_type("SX")  # can set either MX or SX

# Define a NN with nx inputs, nu outputs, and two hidden layers
# The hidden layers have 16 and 32 neurons, respectively.
# The activation function is ReLU, apart from the output layer.
layer_sizes = [NX, 5, 10, NU]
# Build network from layer_sizes
layers = []
for i in range(len(layer_sizes) - 1):
    layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
    if i < len(layer_sizes) - 2:  # add activation for all but last layer
        layers.append(ReLU())
net = Sequential[ca.SX](tuple(layers))

n_param = net.num_parameters
# print(f"Neural network with {n_param} parameters.")

# Get flattened network parameters
params = []
for _, param in net.parameters():
    params.append(ca.reshape(param, -1, 1))
params_flattened = ca.vertcat(*params)  # column vector of parameters

# Define dynamics and stage cost
x = ca.SX.sym('x', NX)
u = ca.SX.sym('u', NU)
Q = ca.diag([1.0, 1.0])
R = ca.diag([1.0])
x_next = A @ x + B @ u
xr = ca.DM([0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr) + (u - ur).T @ R @ (u - ur)
f_dyn = ca.Function('f_dyn', [x, u], [x_next, l], ['x', 'u'], ['x_next', 'l'])
net_fcn = ca.Function('net_fcn', [x, params_flattened], [net(x.T)])
# vector of all decision variables ordered as [x0, u0, x1, u1, ..., xN]
X = ca.SX.sym('X', NX, N + 1)  # state trajectory
U = ca.SX.sym('U', NU, N)    # control trajectory
# vector of all decision variables ordered as [x0, u0, x1, u1, ..., xN]
decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
# to extract trajectories in nice shape from decvar
extract_traj = ca.Function("extract_traj", [decvar], [X, U])
traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])

# Create an NLP to minimize the cost over a horizon (parameterized by initial state)
w = []      # decision variables
w0 = []     # initial guess
lbw = []    # lower bounds
ubw = []    # upper bounds
J = 0       # objective
g = []      # constraints
lbg = []    # lower bounds on constraints
ubg = []    # upper bounds on constraints

# Initialize state as symbolic parameter
Xk = ca.SX.sym('X_0', NX)
w += [Xk]
lbw += [-50.0]*NX
ubw += [50.0]*NX
w0 += [0.0]*NX
for k in range(N):
    u_k = ca.SX.sym('u_' + str(k), NU)
    w += [u_k]
    lbw += [-1.0]*NU
    ubw += [1.0]*NU
    w0 += [0.0]*NU
    
    # Compute next state
    Xk_next, l_k = f_dyn(Xk, u_k)
    # Add state to decision variables
    Xk = ca.SX.sym('X_' + str(k+1), NX)
    w += [Xk]
    lbw += [-50.0]*NX
    ubw += [50.0]*NX
    w0 += [0.0]*NX
    
    # Add dynamics constraint
    g += [Xk_next - Xk]
    lbg += [0.0]*NX
    ubg += [0.0]*NX
                
    # Accumulate cost
    J += l_k
    
# Add final cost
E = solve_discrete_are(A.full(), B.full(), Q.full(), R.full())
J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)

# Create NLP solver
nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# Create the NLP solver
opts = {"expand": True, "ipopt": {"print_level": 0, "max_iter": 3000, "tol": 1e-8}}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
# solver = ca.nlpsol('solver', 'sqpmethod', nlp)

# Sample batch of initial states uniformly
N_TEST = 200  # number of test initial states
test_bound = 2.0
np.random.seed(42)  # for reproducibility
initial_states = np.random.uniform(-test_bound, test_bound, size=(N_TEST, NX))
print(f"Testing {N_TEST} uniformly sampled initial states...")

# Storage for results
x_opt_batch = []
u_opt_batch = []
x_sim_batch = []
u_sim_batch = []
rmse_states_batch = []
rmse_u_batch = []

# Load optimal parameters from the learned policy NLP
MODEL_DIR = Path(__file__).parent / "models"
model_name = get_model_name(layer_sizes)
params_file = find_latest_params(MODEL_DIR, model_name, "yaml")
# params_file = MODEL_DIR / "optimal_params_2x5x10x1_2026-02-12_16-52-27.yaml"
if params_file is None:
    raise FileNotFoundError(f"No parameter files found for model {model_name} in {MODEL_DIR}")
params_init_vec = load_params(params_file)
print(f"Loaded parameters from {params_file}")

input("Press Enter to start testing...")

# Solve MPC and simulate learned policy for each initial state
for i, x0 in enumerate(initial_states):
    x0_list = x0.tolist()
    
    # Set bounds for initial state
    lbw_i = lbw.copy()
    ubw_i = ubw.copy()
    lbw_i[:NX] = x0_list
    ubw_i[:NX] = x0_list
    
    # Solve the MPC NLP
    solution = solver(x0=w0, lbx=lbw_i, ubx=ubw_i, lbg=lbg, ubg=ubg)
    
    # Extract the optimal solution
    w_opt = solution['x'].full().flatten()
    
    # Extract optimal state and control trajectories
    x_opt, u_opt = extract_traj(w_opt)
    x_opt = np.asarray(x_opt)
    u_opt = np.asarray(u_opt)
    x_opt_batch.append(x_opt)
    u_opt_batch.append(u_opt)

    # Simulate the learned policy for the same initial state
    x_sim = np.zeros((NX, N + 1))
    u_sim = np.zeros((NU, N))
    x_sim[:, 0] = x0
    for k in range(N):
        u_sim[:, k] = net_fcn(x_sim[:, k], params_init_vec).full().flatten()
        x_next, _ = f_dyn(x_sim[:, k], u_sim[:, k])
        x_sim[:, k + 1] = x_next.full().flatten()
    
    x_sim_batch.append(x_sim)
    u_sim_batch.append(u_sim)
    
    # Compute RMSE for states and control
    rmse_states = np.sqrt(np.mean((x_opt - x_sim) ** 2, axis=1))
    rmse_u = np.sqrt(np.mean((u_opt - u_sim) ** 2))
    rmse_states_batch.append(rmse_states)
    rmse_u_batch.append(rmse_u)
    
    if (i + 1) % 5 == 0:
        print(f"Completed {i + 1}/{N_TEST} test cases")
        
# Count how many cases have RMSE above a certain threshold
threshold_rmse = 0.01
num_above_threshold_states = np.sum(np.array(rmse_states_batch) > threshold_rmse, axis=0)
num_above_threshold_u = np.sum(np.array(rmse_u_batch) > threshold_rmse)
print(f"\nNumber of test cases with RMSE above {threshold_rmse}:")
print(f"  State 1: {num_above_threshold_states[0]}")
print(f"  State 2: {num_above_threshold_states[1]}")
print(f"  Control: {num_above_threshold_u}")

# print the x0 for the cases with high RMSE
# print(f"\nInitial states for cases with RMSE above {threshold_rmse}:")
# for i in range(N_TEST):
#     if rmse_states_batch[i][0] > threshold_rmse or rmse_states_batch[i][1] > threshold_rmse or rmse_u_batch[i] > threshold_rmse:
#         print(f"  Case {i}: x0 = {initial_states[i]}, RMSE states = {rmse_states_batch[i]}, RMSE control = {rmse_u_batch[i]}")

# Compute average RMSE over all test cases
avg_rmse_states = np.mean(rmse_states_batch, axis=0)
avg_rmse_u = np.mean(rmse_u_batch)
print(f"\nAverage RMSE over {N_TEST} test cases:")
print(f"  State 1: {avg_rmse_states[0]:.4f}")
print(f"  State 2: {avg_rmse_states[1]:.4f}")
print(f"  Control: {avg_rmse_u:.4f}")

# Plot trajectories for all test cases
t_x = np.arange(N + 1)
t_u = np.arange(N)

plt.figure(figsize=(12, 8))

# State 1 over time
plt.subplot(3, 1, 1)
for i in range(N_TEST):
    plt.plot(t_x, x_opt_batch[i][0, :], '-', alpha=0.3, color='C0')
    plt.plot(t_x, x_sim_batch[i][0, :], '-', alpha=0.3, color='C1')
plt.plot([], [], '-', label=f'Optimal (Avg RMSE={avg_rmse_states[0]:.3f})', color='C0')
plt.plot([], [], '-', label='Learned', color='C1')
plt.ylabel('State 1')
plt.grid(True)
plt.legend()

# State 2 over time
plt.subplot(3, 1, 2)
for i in range(N_TEST):
    plt.plot(t_x, x_opt_batch[i][1, :], '-', alpha=0.3, color='C0')
    plt.plot(t_x, x_sim_batch[i][1, :], '-', alpha=0.3, color='C1')
plt.plot([], [], '-', label=f'Optimal (Avg RMSE={avg_rmse_states[1]:.3f})', color='C0')
plt.plot([], [], '-', label='Learned', color='C1')
plt.ylabel('State 2')
plt.grid(True)
plt.legend()

# Control over time
plt.subplot(3, 1, 3)
for i in range(N_TEST):
    plt.step(t_u, u_opt_batch[i].flatten(), where='post', alpha=0.3, color='C2')
    plt.step(t_u, u_sim_batch[i].flatten(), where='post', alpha=0.3, color='C3')
plt.step([], [], where='post', label=f'Optimal (Avg RMSE={avg_rmse_u:.3f})', color='C2')
plt.step([], [], where='post', label='Learned', color='C3')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.grid(True)
plt.legend()

plt.suptitle(f'Optimal vs Learned Policy ({N_TEST} Test Cases)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Plot trajectory errors
plt.figure(figsize=(12, 8))

# Error in state 1 over time
plt.subplot(3, 1, 1)
for i in range(N_TEST):
    error_x1 = x_opt_batch[i][0, :] - x_sim_batch[i][0, :]
    plt.plot(t_x, error_x1, '-', alpha=0.3, color='C0')
plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[0]:.3f}', color='C0')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.ylabel('State 1 Error')
plt.grid(True)
plt.legend()

# Error in state 2 over time
plt.subplot(3, 1, 2)
for i in range(N_TEST):
    error_x2 = x_opt_batch[i][1, :] - x_sim_batch[i][1, :]
    plt.plot(t_x, error_x2, '-', alpha=0.3, color='C1')
plt.plot([], [], '-', label=f'Avg RMSE={avg_rmse_states[1]:.3f}', color='C1')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.ylabel('State 2 Error')
plt.grid(True)
plt.legend()

# Error in control over time
plt.subplot(3, 1, 3)
for i in range(N_TEST):
    error_u = u_opt_batch[i].flatten() - u_sim_batch[i].flatten()
    plt.step(t_u, error_u, where='post', alpha=0.3, color='C2')
plt.step([], [], where='post', label=f'Avg RMSE={avg_rmse_u:.3f}', color='C2')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Time Step')
plt.ylabel('Control Error')
plt.grid(True)
plt.legend()

plt.suptitle(f'Trajectory Errors (Optimal - Learned) ({N_TEST} Test Cases)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
