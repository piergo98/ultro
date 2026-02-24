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
    matching_files = sorted(model_dir.glob(pattern))
    if matching_files:
        return matching_files[-1]  # Return the last one (most recent alphabetically)
    return None


print("Setting up Explicit MPC problem...")
start_time = time.time()


r = 1.0                # pole magnitude
phi = 0.3              # pole angle (radians)
A, B = make_discrete_conjugate_pair(r, phi)

# Declare variables
NX = 2      # state dimension
NU = 1      # control dimension
NB = 100     # batch size for learning
N = 15      # horizon length

# Creathe a neural network approximator for the control policy
set_sym_type("MX")  # can set either MX or SX

# Define a NN with nx inputs, nu outputs, and two hidden layers
# The hidden layers have 16 and 32 neurons, respectively.
# The activation function is ReLU, apart from the output layer.
layer_sizes = [NX, 20, N]
# Build network from layer_sizes
layers = []
for i in range(len(layer_sizes) - 1):
    layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
    if i < len(layer_sizes) - 2:  # add activation for all but last layer
        layers.append(ReLU())
net = Sequential[ca.MX](tuple(layers))

n_param = int(net.num_parameters)
# print(f"Neural network with {n_param} parameters.")

# Get flattened network parameters
params = []
for _, param in net.parameters():
    params.append(ca.reshape(param, -1, 1))
params_flattened = ca.vertcat(*params)  # column vector of parameters

# Define dynamics and stage cost
x = ca.MX.sym('x', NX)
u = ca.MX.sym('u', NU)
Q = ca.diag([1.0, 1.0])
R = ca.diag([1.0])
x_next = A @ x + B @ u
xr = ca.DM([0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr) + (u - ur).T @ R @ (u - ur)
f_dyn = ca.Function('f_dyn', [x, u], [x_next, l], ['x', 'u'], ['x_next', 'l'])
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

 # Add parameters to decision variables
w += [params_flattened]
lbw += [-20.0]*n_param
ubw += [ 20.0]*n_param

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

w0 += params_init.tolist()

# Define one NLP for each i in the batch
for i in range(NB):
    # Initial state for this batch element
    train_bound = 2.0
    x0 = list(np.random.uniform(-train_bound, train_bound, size=(NX,)).tolist())

    # Decision variables and constraints for this batch element
    # Initialize state
    Xk = ca.MX.sym('X_' + str(i) + '_0', NX)
    w += [Xk]
    lbw += x0
    ubw += x0
    w0 += x0
    for k in range(N):
        u_k_sequence = net_fcn(Xk, params_flattened)  # control input from NN policy
        # Take the first element of the output
        u_k = u_k_sequence[0] 
        # Add constraint on control input
        g += [u_k]
        lbg += [-1.0]
        ubg += [1.0]
        
        # Compute next state
        Xk_next, l_k = f_dyn(Xk, u_k)
        # Add state to decision variables
        Xk = ca.MX.sym('X_' + str(i) + '_' + str(k+1), NX)
        w += [Xk]
        lbw += [-50.0]*NX
        ubw += [50.0]*NX
        w0 += x0
        
        # Add dynamics constraint
        g += [Xk_next - Xk]
        lbg += [0.0]*NX
        ubg += [0.0]*NX
                 
        # Accumulate cost
        J += l_k
        
    # Add final cost
    E = solve_discrete_are(A.full(), B.full(), Q.full(), R.full())
    J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)
# Add a regularization term to the cost to encourage smaller parameters (optional)
reg_weight = 1e-4
# J += reg_weight * ca.sumsqr(params_flattened)

# Create NLP solver
nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
print("NLP problem created.")
decision_variables_num = sum([var.size1()*var.size2() for var in w])
constraints_num = sum([constr.size1()*constr.size2() for constr in g])
print(f"Number of decision variables: {decision_variables_num}")
print(f"Number of constraints: {constraints_num}")

# Create the NLP solver
opts = {
    "expand": True, 
    "ipopt": {
        "print_level": 3, 
        "max_iter": 10000, 
        "tol": 1e-8, 
        "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
        "linear_solver": "ma86",
    }
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the NLP
print("Solving NLP...")
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print("NLP solved.")

# Extract the optimal parameters
w_opt = solution['x'].full().flatten()
optimal_params = w_opt[:n_param]    
# print(f"Regularization term: {reg_weight * np.sum(optimal_params**2):.6f}")

# Get current date for filename
date_str = datetime.now().strftime("%Y-%m-%d")
hour_str = datetime.now().strftime("%H-%M-%S")
date_str = f"{date_str}_{hour_str}"

# Prepare data for serialization
params_dict = {
    "optimal_params": np.asarray(optimal_params).tolist(),
    "n_param": int(n_param),
    "layer_sizes": layer_sizes,
    "model_name": model_name,
    "date": date_str
}

# Save JSON
json_path = MODEL_DIR / f"optimal_params_{model_name}_{date_str}.json"
with json_path.open("w") as f:
    json.dump(params_dict, f, indent=2)
print(f"Saved parameters to {json_path}")

# Save YAML
yaml_path = MODEL_DIR / f"optimal_params_{model_name}_{date_str}.yaml"
with yaml_path.open("w") as f:
    yaml.safe_dump(params_dict, f)
print(f"Saved parameters to {yaml_path}")
# print(f"Optimal parameters: {optimal_params}")

# Extract optimal state and control trajectories for all batches
w_flat = w_opt
state_block_size = (N + 1) * NX
x_opt = np.zeros((NB, N + 1, NX))
u_opt = np.zeros((NB, N))

offset = n_param
for i in range(NB):
    start = offset + i * state_block_size
    # states
    for k in range(N + 1):
        idx0 = start + k * NX
        x_opt[i, k, :] = w_flat[idx0: idx0 + NX]
    # controls (k=0..N-1) using the learned theta_opt and f_approx
    for k in range(N):
        xk_dm = ca.DM(x_opt[i, k, :])
        uk = net_fcn(xk_dm, optimal_params)
        u_opt[i, k] = float(uk[0].full().item())

# Plot trajectories across the batch: individual traces (light) + batch mean (bold)
time_x = np.arange(N + 1)
time_u = np.arange(N)

plt.figure(figsize=(12, 6))

# States subplot
plt.subplot(1, 2, 1)
# individual traces
for i in range(NB):
    plt.plot(time_x, x_opt[i, :, 0], color='gray', alpha=0.25)
    plt.plot(time_x, x_opt[i, :, 1], color='gray', alpha=0.25)
# batch mean
mean_x1 = x_opt[:, :, 0].mean(axis=0)
mean_x2 = x_opt[:, :, 1].mean(axis=0)
plt.plot(time_x, mean_x1, 'b-', linewidth=2, label='mean x1')
plt.plot(time_x, mean_x2, 'r-', linewidth=2, label='mean x2')
plt.title('Batch State Trajectories (individual & mean)')
plt.xlabel('Time Step')
plt.ylabel('States')
plt.grid()
plt.legend()

# Control subplot
plt.subplot(1, 2, 2)
for i in range(NB):
    plt.step(time_u, u_opt[i, :], where='post', color='gray', alpha=0.25)
mean_u = u_opt.mean(axis=0)
plt.step(time_u, mean_u, where='post', color='k', linewidth=2, label='mean u')
plt.title('Batch Control Inputs (individual & mean)')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()




