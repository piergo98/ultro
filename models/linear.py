import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are

# Class implementing a linear system of the form x_{k+1} = A x_k + B u_k and a simple MPC controller
class LinearSystem:
    def __init__(self, A, B, dt=0.1, N=10):
        # Chech the instance of A and B
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        if not isinstance(B, np.ndarray):
            B = np.array(B)
        self.A = A
        self.B = B
        self.dt = dt
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
        self.step = self.step_func()
        
        self.good_initial_points = 0
        self.skip = False
        
        self.N = N  # MPC horizon length
        Q = np.diag([1.0, 1.0, 1.0, 1.0])  # state cost weight
        R = np.diag([1.0])  # control cost weight
        self.define_simple_MPC_control(N, Q, R)
    
    def step_func(self):
        """Compute the next state given current state and control input."""
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        x_next = self.A @ x + self.B @ u
        return ca.Function("step", [x, u], [x_next])
    
    def define_simple_MPC_control(self, N, Q, R):
        ''' A simple MPC controller that tries to stabilize the pole at the upright position. '''
        # Define optimization variables and parameters
        x = ca.SX.sym('x', self.nx)  # state variable
        u = ca.SX.sym('u', self.nu)  # control variable
        x0 = ca.SX.sym('x0', self.nx)  # initial state parameter
        
        X = ca.SX.sym('X', self.nx, N+1)  # state trajectory
        U = ca.SX.sym('U', self.nu, N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        # Define cost function (quadratic cost on state deviation and control effort)
        xr = ca.DM([0.0, 0, 0, 0])  # desired state (upright position)
        # Check if Q and R are provided, otherwise use default values
        if Q is None:
            Q = np.diag([10.0, 1.0, 100.0, 1.0])  # state cost weight
        if R is None:
            R = np.diag([1.0])  # control cost weight
            
        E = solve_discrete_are(self.A, self.B, Q, R)
        
        x_next = self.step(x, u)
        l = (x - xr).T @ Q @ (x - xr) + u.T @ R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        x_bounds = [1, 1.5, 0.35, 1.0]
        u_bounds = [1.0]
        
        # State and control bounds
        self.u_min, self.u_max = -np.array(u_bounds), np.array(u_bounds)
        self.x_min, self.x_max = -np.array(x_bounds), np.array(x_bounds)
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.lbw = []    # lower bounds on decision variables
        self.ubw = []    # upper bounds on decision variables
        self.w0 = []     # initial guess for decision variables
        J = 0       # objective function
        g = []      # constraints
        self.lbg = []    # lower bounds on constraints
        self.ubg = []    # upper bounds on constraints
        Xk = ca.SX.sym('X0', self.nx)  # initial state variable
        w += [Xk]
        self.lbw += self.x_min.tolist()
        self.ubw += self.x_max.tolist()
        self.w0 += [0.0 for _ in range(4)]
        g += [Xk - x0]  # initial state constraint
        self.lbg += [0.0 for _ in range(4)]
        self.ubg += [0.0 for _ in range(4)]
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            self.lbw += self.u_min.tolist()  # control limits
            self.ubw += self.u_max.tolist()
            self.w0 += [0.0]     # initial guess
            
            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            self.lbw += self.x_min.tolist()
            self.ubw += self.x_max.tolist()
            self.w0 += [0.0 for _ in range(4)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(4)]
            self.ubg += [0.0 for _ in range(4)]
            
            J += l_k  # accumulate cost
        J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)  # terminal cost
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': x0}
        opts = {"ipopt.print_level": 0, "print_time": False, "verbose": False}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        
    def solve_MPC(self, x0, ret_seq=False):
        ''' Solve the MPC problem for a given initial state x0. '''
        # print(f"self.w0 = {self.w0}")
        # print(f"self.lbw = {self.lbw}")
        # print(f"self.ubw = {self.ubw}")
        # print(f"self.lbg = {self.lbg}")
        # print(f"self.ubg = {self.ubg}")
        # print(f"x0 = {x0}")
        sol = self.solver(
            x0=self.w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=x0
        )
        
        x_opt, u_opt = self.extract_traj(sol['x'])
        # Return the optimal control sequence
        if ret_seq:
            return u_opt.full().flatten()  # return the full control sequence as a 1D array
        
        return u_opt.full().flatten()[0]  # return only the first control input
    
    def close_loop_simulation(self, x0, Nsim = 60, control_policy=None, plot_results=True):
        """Simulate the cart-pole system and plot the results.
        
        Parameters
        ----------
        x0 : array_like
            Initial state [position, velocity, angle, angular_velocity].
        control_policy : callable or array_like
            Either a function that takes state and returns control u = policy(x),
            or an array of control inputs for each time step. If None, the MPC controller will be used.
        Nsim : int, optional
            Number of simulation steps (default: 60).
        plot_results : bool, optional
            Whether to plot the results (default: True).
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (N+1, 4).
        u_traj : np.ndarray
            Control trajectory of shape (N,).
        """
        # Initialize trajectories
        x_traj = np.zeros((Nsim + 1, self.nx))
        u_traj = np.zeros((Nsim, self.nu))
        
        x_traj[0] = np.array(x0).flatten()
        
        # Simulate forward in time
        for k in range(Nsim):
            # Get control input
            if control_policy is None:
                u_k = self.solve_MPC(x_traj[k])  # Solve MPC to get control input for current state
            elif callable(control_policy):
                u_k = control_policy(x_traj[k])  # Get control input from the provided policy function
            else:
                u_k = control_policy[k]  # Get control input from the provided array
            
            # Ensure u_k is scalar
            if hasattr(u_k, '__len__'):
                u_k = float(u_k[0])
            else:
                u_k = float(u_k)
            
            u_traj[k, :] = u_k
            
            # Step forward
            x_next = self.step(x_traj[k], [u_k])
            x_traj[k + 1] = np.array(x_next.full()).flatten()
        
        if control_policy is None:
            if self.solver.stats()['success']:
                self.good_initial_points += 1
            else:
                # print(self.solver.stats()['return_status'])
                self.skip = True
        
        if not plot_results:
            return x_traj, u_traj
        
        # Plot results
        time = np.arange(Nsim + 1) * self.dt
        time_u = np.arange(Nsim) * self.dt
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 10))
        title = "Cart-Pole Simulation"
        
        # Position
        axes[0].plot(time, x_traj[:, 1], 'b-', linewidth=2)
        axes[0].set_ylabel('Position (m)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        
        # Velocity
        axes[1].plot(time, x_traj[:, 3], 'r-', linewidth=2)
        axes[1].set_ylabel('Velocity (m/s)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Angle
        axes[2].plot(time, x_traj[:, 0], 'g-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (rad)', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Angular velocity
        axes[3].plot(time, x_traj[:, 2], 'c-', linewidth=2)
        axes[3].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Control
        axes[4].step(time_u, u_traj[:, 0], 'k-', where='post', linewidth=2)
        axes[4].set_ylabel('Control Force (N)', fontsize=10)
        axes[4].set_xlabel('Time (s)', fontsize=10)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj
    
if __name__ == "__main__":
    # Define A and B matrices
    A = [[1.0, 0.1, 0.0, 0.0],
         [0.0, 0.9818, 0.2673, 0.0],
         [0.0, 0.0, 1.0, 0.1],
         [0.0, -0.0455, 3.1182, 1.0]]
    
    B = [[0.0], [0.1818], [0.0], [0.4546]]
    
    # Create linear system instance
    lin_sys = LinearSystem(A, B, dt=0.1)
    Ncsim = 200
    
    # Example usage: simulate with zero control input
    alpha = 0.3
    p_bound = 1.5 * alpha
    v_bound = 1.0 * alpha 
    theta_bound = 1.0 * alpha
    omega_bound = 0.35 * alpha
    np.random.seed(36)  # for reproducibility
    for _ in range(Ncsim):
        p0 = np.random.uniform(-p_bound, p_bound)
        v0 = np.random.uniform(-v_bound, v_bound)
        theta0 = np.random.uniform(-theta_bound, theta_bound)
        omega0 = np.random.uniform(-omega_bound, omega_bound)
        x0 = [theta0, p0, omega0, v0]
        print (f"Initial state: {x0}")
        control_policy = lambda x: np.array([0.0])  # zero control input
        
        # Close-loop simulation with MPC control
        lin_sys.close_loop_simulation(x0, plot_results=False)
        
    print(f"Number of good initial points: {lin_sys.good_initial_points} out of {Ncsim}")