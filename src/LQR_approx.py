# This code implements a simple Constrained LQR, which also optimizes the parameters of the function
# approximator used for the control policy.

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

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

# example usage
r = 1.0                # pole magnitude
phi = 0.3            # pole angle (radians)
A, B = make_discrete_conjugate_pair(r, phi)

# Declare variables
NX = 2  # state dimension
NU = 1  # control dimension
n_param = 2*NX + 1  # number of parameters in approximator
N = 20  # horizon length
NB = 100  # batch size for learning

x = ca.SX.sym('x', NX)  # state
u = ca.SX.sym('u', NU)  # control input
theta = ca.SX.sym('theta', n_param)  # parameters to optimize

# Define dynamics and cost function
Q = ca.diag([1.0, 1.0])
R = ca.diag([1.0])
x_next = A @ x + B @ u
xr = ca.DM([0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr) + (u - ur).T @ R @ (u - ur)
f_dyn = ca.Function('f_dyn', [x, u], [x_next, l], ['x', 'u'], ['x_next', 'l'])

# Define approximator function
f_approx = ca.Function('f_approx', [x], [ca.vertcat(1, x, x**2)])
# print(f_approx(1.0).shape)  # should be (3, 1)

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
w += [theta]
lbw += [-100]*n_param
ubw += [ 100]*n_param
w0 += [1.0]*n_param


# Define an NLP for each i in the batch
for i in range(NB):
    # Initial state for this batch element
    train_bounds = 1.0
    x0 = list(np.random.uniform(-train_bounds, train_bounds, size=(NX,)).tolist())

    # Decision variables and constraints for this batch element
    # Initialize state
    Xk = ca.SX.sym('X_' + str(i) + '_0', NX)
    w += [Xk]
    lbw += x0
    ubw += x0
    w0 += x0

    for k in range(N):
        u_k = theta.T @ f_approx(Xk)  # control input from approximator
        
        # Add constraint on control input
        g += [u_k]
        lbg += [-0.5]
        ubg += [0.5]
        
        # Compute next state
        Xk_next, l_k = f_dyn(Xk, u_k)
        # Add state to decision variables
        Xk = ca.SX.sym('X_' + str(i) + '_' + str(k+1), NX)
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

# Create NLP solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
# opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', prob)

# Solve NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# Extract optimal parameters as a CasADi row vector
theta_opt = ca.DM(w_opt[0:n_param]).T
print("Optimal parameters:", theta_opt)
# print(theta_opt.shape)

# print(len(w_opt[n_param:]))
# input()

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
        uk = theta_opt @ f_approx(xk_dm)
        u_opt[i, k] = float(uk.full().item())

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

# Starting from a different initial condition, simulate the closed-loop system
# then display the results and the difference with the LQR controller
print("="*40)
print("Simulating closed-loop system with approximator vs LQR")
print("="*40)

# Simulate and compare approximator vs LQR over two initial-state regions (batch tests)
n_tests = 20
sim_steps = 50

# LQR gain (numpy)
A_np = A.full()
B_np = B.full()
P = solve_discrete_are(A_np, B_np, Q.full(), R.full())
K_lqr = np.linalg.inv(B_np.T @ P @ B_np + R.full()) @ (B_np.T @ P @ A_np)  # shape (1,2)
print("LQR gain K:", K_lqr)

def run_batch(x0_array):
    # x0_array: (n_tests, NX)
    traj_app = np.zeros((len(x0_array), sim_steps + 1, NX))
    traj_lqr = np.zeros((len(x0_array), sim_steps + 1, NX))
    u_app = np.zeros((len(x0_array), sim_steps))
    u_lqr = np.zeros((len(x0_array), sim_steps))
    for i, x0 in enumerate(x0_array):
        xa = ca.DM(x0)
        xl = ca.DM(x0)
        traj_app[i, 0, :] = x0
        traj_lqr[i, 0, :] = x0
        for k in range(sim_steps):
            # approximator control
            ua = theta_opt @ f_approx(xa)
            xa_next, _ = f_dyn(xa, ua)
            u_app[i, k] = float(ua.full().item())
            traj_app[i, k+1, :] = xa_next.full().flatten()
            xa = xa_next

            # LQR control
            xl_np = xl.full().flatten()
            ul_np = (-K_lqr @ xl_np).reshape(-1)  # shape (1,)
            ul = ca.DM(ul_np)
            xl_next, _ = f_dyn(xl, ul)
            u_lqr[i, k] = float(np.asarray(ul_np).item())
            traj_lqr[i, k+1, :] = xl_next.full().flatten()
            xl = xl_next
    return traj_app, traj_lqr, u_app, u_lqr

# region 1: same as training (-1..1)
rng = np.random.RandomState(0)
x0_region1 = rng.uniform(-1.0, 1.0, size=(n_tests, NX))

# region 2: wider region (choose -3..3)
wider_bounds = 3.0
x0_region2 = rng.uniform(-wider_bounds, wider_bounds, size=(n_tests, NX))

traj_app_r1, traj_lqr_r1, u_app_r1, u_lqr_r1 = run_batch(x0_region1)
traj_app_r2, traj_lqr_r2, u_app_r2, u_lqr_r2 = run_batch(x0_region2)

# compute mean norm differences over time
traj_normdiff_r1 = np.linalg.norm(traj_app_r1 - traj_lqr_r1, axis=2).mean(axis=0)
traj_normdiff_r2 = np.linalg.norm(traj_app_r2 - traj_lqr_r2, axis=2).mean(axis=0)
u_normdiff_r1 = (u_app_r1 - u_lqr_r1).mean(axis=0)
u_normdiff_r2 = (u_app_r2 - u_lqr_r2).mean(axis=0)

time_sim = np.arange(sim_steps + 1)

# Plotting: state-space plots for both regions + mean norm difference
plt.figure(figsize=(14, 6))

# Region 1 state-space
plt.subplot(1, 4, 1)
for i in range(n_tests):
    plt.plot(traj_app_r1[i, :, 0], traj_app_r1[i, :, 1], color='C0', alpha=0.4)
    plt.plot(traj_lqr_r1[i, :, 0], traj_lqr_r1[i, :, 1], color='C1', alpha=0.25, linestyle='--')
# mark starts
plt.scatter(x0_region1[:, 0], x0_region1[:, 1], color='k', s=10)
# highlight training X0 region rectangle [-1,1] x [-1,1]
plt.gca().add_patch(plt.Rectangle((-1, -1), 2, 2, edgecolor='blue', facecolor='blue', alpha=0.12))
plt.title('State-space trajectories (region: [-1,1])')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()

# Region 2 state-space
plt.subplot(1, 4, 2)
for i in range(n_tests):
    plt.plot(traj_app_r2[i, :, 0], traj_app_r2[i, :, 1], color='C0', alpha=0.4)
    plt.plot(traj_lqr_r2[i, :, 0], traj_lqr_r2[i, :, 1], color='C1', alpha=0.25, linestyle='--')
plt.scatter(x0_region2[:, 0], x0_region2[:, 1], color='k', s=10)
# also show the training region for reference
plt.gca().add_patch(plt.Rectangle((-1, -1), 2, 2, edgecolor='green', facecolor='green', alpha=0.12))
plt.title(f'State-space trajectories (region: [{-wider_bounds},{wider_bounds}])')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()

# Mean norm trajectory difference over time
plt.subplot(1, 4, 3)
plt.plot(time_sim, np.log10(traj_normdiff_r1), label='log mean ||x_app - x_lqr|| (region1)')
plt.plot(time_sim, np.log10(traj_normdiff_r2), label='log mean ||x_app - x_lqr|| (region2)')
plt.xlabel('Time step')
plt.ylabel('Log mean state diff norm')
plt.title('Approximator vs LQR (mean state difference)')
plt.grid()
plt.legend()

# Mean norm control difference over time
plt.subplot(1, 4, 4)
plt.plot(time_sim[:-1], u_normdiff_r1, label='Mean ||u_app - u_lqr|| (region1)')
plt.plot(time_sim[:-1], u_normdiff_r2, label='Mean ||u_app - u_lqr|| (region2)')
plt.xlabel('Time step')
plt.ylabel('Mean control diff norm')
plt.title('Approximator vs LQR (mean control difference)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
    
    