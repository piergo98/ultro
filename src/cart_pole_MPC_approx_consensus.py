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

from csnn import set_sym_type, Linear, Sequential, ReLU, Tanh, Sigmoid
from learning_problem.models.cart_pole import CartPole

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

def parameter_initialization_he(layer_sizes, n_param):
    """Kaiming (He) initialization for neural network parameters.

    Parameters
    ----------
    layer_sizes : list of int
        List containing the sizes of each layer in the network.
    n_param : int
        Expected number of parameters.

    Returns
    -------
    params_init : np.ndarray
        Flattened array of initialized parameters.
    """
    params_init = []

    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std = np.sqrt(2.0 / float(fan_in))
        W = np.random.randn(fan_out, fan_in) * std  # weights
        b = np.zeros(fan_out)                        # biases initialized to zero
        params_init.append(W.ravel())
        params_init.append(b.ravel())
        
    params_init_vec = np.concatenate(params_init)
    
    # safety: if sizes mismatch, pad or truncate to n_param
    if params_init_vec.size < n_param:
        params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
    elif params_init_vec.size > n_param:
        params_init_vec = params_init_vec[:n_param]

    return params_init_vec

def custom_parameter_initialization(layer_sizes, n_param):
    """Custom initialization for neural network parameters (e.g., Xavier).

    Parameters
    ----------
    layer_sizes : list of int
        List containing the sizes of each layer in the network.
    n_param : int
        Expected number of parameters.

    Returns
    -------
    params_init : np.ndarray
        Flattened array of initialized parameters.
    """
    params_init = []

    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        W = np.ones((fan_out, fan_in))  # weights
        b = np.ones(fan_out)                                      # biases initialized to zero
        params_init.append(W.ravel())
        params_init.append(b.ravel())
        
    params_init_vec = np.concatenate(params_init)
    
    # safety: if sizes mismatch, pad or truncate to n_param
    if params_init_vec.size < n_param:
        params_init_vec = np.concatenate([params_init_vec, np.zeros(n_param - params_init_vec.size)])
    elif params_init_vec.size > n_param:
        params_init_vec = params_init_vec[:n_param]

    return params_init_vec

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


print("Setting up optimizzation for learning problem...")

cart_pole = CartPole(sym_type='SX')

# Declare variables
NX = 4      # state dimension
NU = 1      # control dimension
NB = 40     # batch size for learning
N = 20      # horizon length

# Creathe a neural network approximator for the control policy
set_sym_type("SX")
# Define a NN with nx inputs, nu outputs, and nh hidden layers
# The activation function is ReLU, apart from the output layer.
layer_sizes = [NX, 20, NU]
# Build network from layer_sizes
layers = []
for i in range(len(layer_sizes) - 1):
    layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
    if i < len(layer_sizes) - 2:  # add activation for all but last layer
        layers.append(ReLU())
net = Sequential[ca.SX](tuple(layers))

n_param = net.num_parameters
print(f"Number of parameters in the NN: {n_param}")

# Get flattened network parameters
params = []
for _, param in net.parameters():
    params.append(ca.reshape(param, -1, 1))
params_flattened = ca.vertcat(*params)  # column vector of parameters

# Define dynamics and stage cost
x = ca.SX.sym('x', NX)
u = ca.SX.sym('u', NU)
Q = ca.diag(ca.DM([100, 1, 30, 1]))  # state cost weights
R = ca.diag(ca.DM([0.01]))  # control cost weight
x_next = cart_pole.step(x, u)
xr = ca.DM([0.0, 0.0, 0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr) + (u - ur).T @ R @ (u - ur)
f_dyn = ca.Function('f_dyn', [x, u], [x_next, l], ['x', 'u'], ['x_next', 'l'])

# Create a neural network function
net_fcn = ca.Function('net_fcn', [x, params_flattened], [net(x.T)])

# Create an NLP to minimize the cost over a horizon
w = []      # decision variables
w0 = []     # initial guess
lbw = []    # lower bounds
ubw = []    # upper bounds
J = 0       # objective
g = []      # constraints
lbg = []    # lower bounds on constraints
ubg = []    # upper bounds on constraints

# Setup model directory and filenames
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_name = get_model_name(layer_sizes)

# Initialize parameters: try loading from file, fall back to He initialization
# PARAMS_FILE = find_latest_params(MODEL_DIR, model_name, "yaml")  # Set to None to force He initialization
PARAMS_FILE = None
if PARAMS_FILE is not None:
    try:
        params_init = load_params(PARAMS_FILE, n_param)
        print(f"Loaded pre-optimized parameters from {PARAMS_FILE}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not load parameters from {PARAMS_FILE}: {e}")
        print("Using Kaiming (He) initialization instead.")
        params_init = parameter_initialization_he(layer_sizes, n_param)
else:
    print("Using Kaiming (He) initialization.")
    params_init = parameter_initialization_he(layer_sizes, n_param)

# Set a seed for reproducibility of initial states
np.random.seed(42)
p_train_bound = 1.0
v_train_bound = 2.0
theta_train_bound = np.pi/6
omega_train_bound = 1.0

p_bound = 2.0
v_bound = 4.0
theta_bound = np.pi/3
omega_bound = 2.0

# Compute the linearized dynamics matrices A and B around the equilibrium point (xr, ur)                   # equilibrium control input (no force)
A, B = cart_pole.lin_dyn(xr, ur)

# Pre-compute the LQR terminal cost matrix outside the loop
E = solve_continuous_are(A.full(), B.full(), Q.full(), R.full())
E_dm = ca.DM(E)

params_vec = []

# z = ca.SX.sym('z', n_param)
# w += [z]
# lbw += [-50.0]*n_param
# ubw += [ 50.0]*n_param
# w0 += params_init.tolist()

# Define one NLP for each i in the batch
for i in range(NB):
    # Initial state for this batch element
    p0 = np.random.uniform(-p_train_bound, p_train_bound)
    v0 = np.random.uniform(-v_train_bound, v_train_bound)
    theta0 = np.random.uniform(-theta_train_bound, theta_train_bound)
    omega0 = np.random.uniform(-omega_train_bound, omega_train_bound)
    x0 = [p0, v0, theta0, omega0]
    
    # Add parameters to decision variables
    NN_params = ca.SX.sym('theta_' + str(i), n_param)  # parameters to optimize
    w += [NN_params]
    lbw += [-50.0]*n_param
    ubw += [ 50.0]*n_param
    w0 += params_init.tolist()
    # Add parameters to the list for later analysis
    params_vec.append(NN_params)

    # Decision variables and constraints for this batch element
    # Initialize state
    Xk = ca.SX.sym('X_' + str(i) + '_0', NX)
    w += [Xk]
    lbw += x0
    ubw += x0
    w0 += [0.0]*NX  # initial guess for states (can be improved by using x0)
    for k in range(N):
        u_k = net_fcn(Xk, NN_params)  # control input from approximator
        
        # Add constraint on control input
        g += [u_k]
        lbg += [-25.0]
        ubg += [25.0]
        
        # Compute next state and cost
        Xk_next, l_k = f_dyn(Xk, u_k)
        
        # Compute next state and cost using dynamics
        # We'll compute control from NN after collecting all states
        Xk = ca.SX.sym('X_' + str(i) + '_' + str(k+1), NX)
        w += [Xk]
        lbw += [-p_bound, -v_bound, -theta_bound, -omega_bound]
        ubw += [p_bound, v_bound, theta_bound, omega_bound]

        w0 += [0.0]*NX  # initial guess for states (can be improved by using x0)
        
        # Add dynamics constraint
        g += [Xk_next - Xk]
        lbg += [0.0]*NX
        ubg += [0.0]*NX
                    
        # Accumulate cost
        J += l_k
    
    # Add the consensus constraint to encourage similar parameters across batches (i.e., theta_i should be close to z)
    # g += [NN_params - z]
    # lbg += [-1e-4]*n_param
    # ubg += [1e-4]*n_param
    
    # Add final cost
    J += (Xk - xr).T @ E_dm @ (Xk - xr)  
    # Add regularization on parameters across batches to encourage similar parameters
    # J += 1e-3 * ca.sumsqr(NN_params)

# Create NLP solver
nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
print("NLP problem created.")
decision_variables_num = sum([var.size1()*var.size2() for var in w])
constraints_num = sum([constr.size1()*constr.size2() for constr in g])
print(f"Number of decision variables: {decision_variables_num}")
print(f"Number of constraints: {constraints_num}")

# Create the NLP solver
print("Creating NLP solver...")
start_time = time.time()
# mode = "jit"
# flags = ["-O2"] # Linux/OSX
# # Pick a compiler
# compiler = "gcc"    # Linux
# # By default, the compiler will be gcc or cl.exe
# jit_options = {"flags": flags, "verbose": True, "compiler": compiler}
opts = {
    # "jit": True,
    # "compiler": "shell",
    # "jit_options": jit_options,
    "expand": True,  # Set to True for faster evaluation at cost of more memory
    "ipopt": {
        "print_level": 5, 
        "max_iter": 3000, 
        "tol": 1e-6, 
        # "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
        # "linear_solver": "ma86",
        "mu_strategy": "adaptive",
    }
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
print(f"Solver creation time: {time.time() - start_time:.2f} seconds")

# Solve the NLP
print("Solving NLP...")
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print("NLP solved.")

# Extract the optimal parameters and trajectories
w_opt = solution['x'].full().flatten()

# Structure: [z, theta_0, X_0_0, X_0_1, ..., X_0_N, theta_1, X_1_0, ..., X_1_N, ...]
# Block size per batch: n_param + (N+1)*NX
block_size_per_batch = n_param + (N + 1) * NX

# Extract consensus parameters (first n_param elements)
# z_opt = w_opt[:n_param]

# Storage
theta_opt = np.zeros((NB, n_param))  # individual parameters for each batch
x_opt = np.zeros((NB, N + 1, NX))     # states for each batch
u_opt = np.zeros((NB, N))             # controls for each batch

offset = 0  # Start after z
for i in range(NB):
    # Extract parameters for batch i
    theta_opt[i, :] = w_opt[offset : offset + n_param]
    offset += n_param
    
    # Extract states for batch i
    for k in range(N + 1):
        x_opt[i, k, :] = w_opt[offset : offset + NX]
        offset += NX
    
    # Compute controls using individual batch parameters
    for k in range(N):
        xk_dm = ca.DM(x_opt[i, k, :])
        uk = net_fcn(xk_dm, theta_opt[i, :])
        u_opt[i, k] = float(uk.full().item())

# Use the consensus parameter z as the "optimal" parameters
# optimal_params = z_opt

# Print diagnostics about parameter consensus
param_deviation = np.zeros(NB)
# for i in range(NB):
#     param_deviation[i] = np.linalg.norm(theta_opt[i, :] - z_opt)

# print(f"\nConsensus parameter statistics for {NB} batches:")
# print(f"  Consensus param (z) L2 norm: {np.linalg.norm(z_opt):.4f}")
print(f"  Mean deviation ||theta_i - z||: {param_deviation.mean():.6f}")
print(f"  Max deviation ||theta_i - z||: {param_deviation.max():.6f}")
print(f"  Min deviation ||theta_i - z||: {param_deviation.min():.6f}")

# Plot trajectories across the batch: individual traces (light) + batch mean (bold)
time_x = np.arange(N + 1)
time_u = np.arange(N)

plt.figure(figsize=(12, 10))

# State 1: Position
plt.subplot(5, 1, 1)
for i in range(NB):
    plt.plot(time_x, x_opt[i, :, 0], color='gray', alpha=0.25)
mean_x1 = x_opt[:, :, 0].mean(axis=0)
plt.plot(time_x, mean_x1, 'b-', linewidth=2, label='mean position')
plt.title('Batch Trajectories (individual & mean)')
plt.ylabel('Position')
plt.grid()
plt.legend()

# State 2: Velocity
plt.subplot(5, 1, 2)
for i in range(NB):
    plt.plot(time_x, x_opt[i, :, 1], color='gray', alpha=0.25)
mean_x2 = x_opt[:, :, 1].mean(axis=0)
plt.plot(time_x, mean_x2, 'r-', linewidth=2, label='mean velocity')
plt.ylabel('Velocity')
plt.grid()
plt.legend()

# State 3: Angle
plt.subplot(5, 1, 3)
for i in range(NB):
    plt.plot(time_x, x_opt[i, :, 2], color='gray', alpha=0.25)
mean_x3 = x_opt[:, :, 2].mean(axis=0)
plt.plot(time_x, mean_x3, 'g-', linewidth=2, label='mean angle')
plt.ylabel('Angle (rad)')
plt.grid()
plt.legend()

# State 4: Angular Velocity
plt.subplot(5, 1, 4)
for i in range(NB):
    plt.plot(time_x, x_opt[i, :, 3], color='gray', alpha=0.25)
mean_x4 = x_opt[:, :, 3].mean(axis=0)
plt.plot(time_x, mean_x4, 'c-', linewidth=2, label='mean angular velocity')
plt.ylabel('Angular Velocity')
plt.grid()
plt.legend()

# Control
plt.subplot(5, 1, 5)
for i in range(NB):
    plt.step(time_u, u_opt[i, :], where='post', color='gray', alpha=0.25)
mean_u = u_opt.mean(axis=0)
plt.step(time_u, mean_u, where='post', color='k', linewidth=2, label='mean control')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Get current date for filename
date_str = datetime.now().strftime("%Y-%m-%d")
hour_str = datetime.now().strftime("%H-%M-%S")
date_str = f"{date_str}_{hour_str}"

# Prepare data for serialization
params_dict = {
    # "optimal_params": np.asarray(optimal_params).tolist(),  # consensus parameter z
    # "consensus_param": np.asarray(z_opt).tolist(),  # same as optimal_params
    "individual_params": np.asarray(theta_opt).tolist(),  # all batch parameters (NB x n_param)
    "param_deviations": np.asarray(param_deviation).tolist(),  # ||theta_i - z|| for each batch
    "mean_deviation": float(param_deviation.mean()),
    "max_deviation": float(param_deviation.max()),
    "n_param": int(n_param),
    "layer_sizes": layer_sizes,
    "model_name": model_name,
    "batch_size": int(NB),
    "horizon": int(N),
    "consensus": True,
    "date": date_str
}

# Save JSON
json_path = MODEL_DIR / f"optimal_params_cp_consensus_{model_name}_{date_str}.json"
with json_path.open("w") as f:
    json.dump(params_dict, f, indent=2)
print(f"Saved parameters to {json_path}")

# Save YAML
yaml_path = MODEL_DIR / f"optimal_params_cp_consensus_{model_name}_{date_str}.yaml"
with yaml_path.open("w") as f:
    yaml.safe_dump(params_dict, f)
print(f"Saved parameters to {yaml_path}")




