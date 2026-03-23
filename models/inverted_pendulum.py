import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are

# Class implementing the cart-pole environment
class InvertedPendulum:
    def __init__(self, dt=0.1, sym_type='SX'):
        self.dt = dt  # time step
        self.g = 9.81  # gravity
        self.l = 0.5   # length of the pole
        self.m = 0.05   # mass of the pole
        self.sym_type = sym_type  # symbolic type for CasADi (SX or MX)
        
        self.dynamics = self.dynamics_func()  # CasADi function for dynamics
        self.step = self.step_func()  # CasADi function for RK4 integration
        self.lin_dyn = self.linearized_dynamics()  # CasADi function for linearized dynamics
        
        N = 10  # MPC horizon
        self.define_simple_MPC_control(N)  # Define the MPC controller

    def dynamics_func(self):
        ''' Computes the time derivative of the state given the current state and control input. '''
        # x = [position, velocity, angle, angular_velocity]
        # u = [force]
        
        x = getattr(ca, self.sym_type).sym('x', 2)  # state vector
        u = getattr(ca, self.sym_type).sym('u', 1)  # control input
        
        s = ca.sin(x[0])
        
        theta_dot = x[1]
        
        common_denominator = self.m * self.l**2
        
        theta_ddot = (u + self.m * self.l + self.g * s) / common_denominator
        
        return ca.Function('dynamics', [x, u], [ca.vertcat(theta_dot, theta_ddot)])
    
    def step_func(self):
        ''' Computes the next state given the current state and control input using RK4 integration. '''
        x0 = getattr(ca, self.sym_type).sym('x0', 2)
        u0 = getattr(ca, self.sym_type).sym('u0', 1)
        M = 4  # number of stages for RK4
        DT = self.dt / M
        X = x0
        for i in range(M):
            k1 = self.dynamics(X, u0)
            k2 = self.dynamics(X + DT/2 * k1, u0)
            k3 = self.dynamics(X + DT/2 * k2, u0)
            k4 = self.dynamics(X + DT * k3, u0)
            X = X + DT/6 * (k1 + 2*k2 + 2*k3 + k4)
        x_next = X
        
        return ca.Function('step', [x0, u0], [x_next])
    
    def linearized_dynamics(self):
        ''' Computes the linearized dynamics matrices A and B around an equilibrium point (x_eq, u_eq). '''
        x = getattr(ca, self.sym_type).sym('x', 2)  # state vector
        u = getattr(ca, self.sym_type).sym('u', 1)  # control input
        
        s = ca.sin(x[0])
        
        theta_dot = x[1]
        
        common_denominator = self.m * self.l**2
        
        theta_ddot = (u + self.m * self.l + self.g * s) / common_denominator
        
        A_sym = ca.jacobian(ca.vertcat(theta_dot, theta_ddot), x)
        B_sym = ca.jacobian(ca.vertcat(theta_dot, theta_ddot), u)
        
        return ca.Function('linearized_dynamics', [x, u], [A_sym, B_sym])
    
    def define_simple_MPC_control(self, N):
        ''' A simple MPC controller that tries to stabilize the pole at the upright position. '''
        # Define optimization variables and parameters
        x = ca.SX.sym('x', 2)  # state variable
        u = ca.SX.sym('u', 1)  # control variable
        x0 = ca.SX.sym('x0', 2)  # initial state parameter
        
        X = ca.SX.sym('X', 2, N+1)  # state trajectory
        U = ca.SX.sym('U', 1, N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        # Define cost function (quadratic cost on state deviation and control effort)
        xr = ca.DM([0.0, 0])  # desired state (upright position)
        Q = ca.diag(ca.DM([30, 1]))  # state cost weights
        R = ca.diag(ca.DM([1.0]))  # control cost weight
        A, B = self.lin_dyn(xr, ca.DM([0]))  # linearized dynamics around the upright position
        E = solve_continuous_are(A.full(), B.full(), Q.full(), R.full())
        
        x_next = self.step(x, u)
        l = (x - xr).T @ Q @ (x - xr) + u.T @ R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        p_bound = 2.0
        v_bound = 4.0
        theta_bound = np.pi/3
        omega_bound = 2.0
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.lbw = []    # lower bounds on decision variables
        self.ubw = []    # upper bounds on decision variables
        self.w0 = []     # initial guess for decision variables
        J = 0       # objective function
        g = []      # constraints
        self.lbg = []    # lower bounds on constraints
        self.ubg = []    # upper bounds on constraints
        Xk = ca.SX.sym('X0', 2)  # initial state variable
        w += [Xk]
        self.lbw += [-theta_bound, -omega_bound]
        self.ubw += [theta_bound, omega_bound]
        self.w0 += [0.0 for _ in range(2)]
        g += [Xk - x0]  # initial state constraint
        self.lbg += [0.0 for _ in range(2)]
        self.ubg += [0.0 for _ in range(2)]
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            self.lbw += [0.0]  # control limits
            self.ubw += [10.0]
            self.w0 += [0.0]     # initial guess
            
            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), 2)
            w += [Xk]
            self.lbw += [-theta_bound, -omega_bound]
            self.ubw += [theta_bound, omega_bound]
            self.w0 += [0.0 for _ in range(2)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(2)]
            self.ubg += [0.0 for _ in range(2)]
            
            J += l_k  # accumulate cost
        J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)  # terminal cost
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': x0}
        opts = {"ipopt.print_level": 0, "print_time": False}
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
            Initial state [angle, angular_velocity].
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
            State trajectory of shape (Nsim+1, 2).
        u_traj : np.ndarray
            Control trajectory of shape (Nsim,).
        """
        # Initialize trajectories
        x_traj = np.zeros((Nsim + 1, 2))
        u_traj = np.zeros(Nsim)
        
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
            
            u_traj[k] = u_k
            
            # Step forward
            x_next = self.step(x_traj[k], [u_k])
            x_traj[k + 1] = np.array(x_next.full()).flatten()
        
        if not plot_results:
            return x_traj, u_traj
        
        # Plot results
        time = np.arange(Nsim + 1) * self.dt
        time_u = np.arange(Nsim) * self.dt
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        title = "Inverted-Pendulum Simulation"
        
        # Angle
        axes[0].plot(time, x_traj[:, 0], 'g-', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('Angle (rad)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Angular velocity
        axes[1].plot(time, x_traj[:, 1], 'c-', linewidth=2)
        axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Control
        axes[2].step(time_u, u_traj, 'k-', where='post', linewidth=2)
        axes[2].set_ylabel('Control Force (N)', fontsize=10)
        axes[2].set_xlabel('Time (s)', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj
    
    def animate(self, x0, control_policy, N, interval=50, save_path=None, title="Inverted-Pendulum Animation"):
        """Animate the cart-pole system.
        
        Parameters
        ----------
        x0 : array_like
            Initial state [position, velocity, angle, angular_velocity].
        control_policy : callable or array_like
            Either a function that takes state and returns control u = policy(x),
            or an array of control inputs for each time step.
        N : int
            Number of simulation steps.
        interval : int, optional
            Time between frames in milliseconds (default: 50ms).
        save_path : str, optional
            If provided, save animation to this file path (e.g., 'animation.gif' or 'animation.mp4').
        title : str, optional
            Title for the animation.
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (N+1, 2).
        u_traj : np.ndarray
            Control trajectory of shape (N,).
        anim : matplotlib.animation.FuncAnimation
            The animation object.
        """
        # First, simulate to get the trajectory
        x_traj = np.zeros((N + 1, 2))
        u_traj = np.zeros(N)
        
        x_traj[0] = np.array(x0).flatten()
        
        for k in range(N):
            if control_policy is None:
                u_k = self.solve_MPC(x_traj[k])  # Solve MPC to get control input for current state
            elif callable(control_policy):
                u_k = control_policy(x_traj[k])
            else:
                u_k = control_policy[k]
            
            if hasattr(u_k, '__len__'):
                u_k = float(u_k[0])
            else:
                u_k = float(u_k)
            
            u_traj[k] = u_k
            x_next = self.step(x_traj[k], [u_k])
            x_traj[k + 1] = np.array(x_next.full()).flatten()
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate axis limits based on trajectory
        x_min = - 1.5
        x_max = + 1.5
        y_min = -self.l - 0.5
        y_max = self.l + 0.5
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title(title)
        
        # Draw ground
        ax.plot([x_min, x_max], [0, 0], 'k-', linewidth=2)
        
        # Initialize objects for animation
        pole, = ax.plot([], [], 'r-', linewidth=4, marker='o', markersize=8, markerfacecolor='red')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def init():
            """Initialize animation."""
            pole.set_data([], [])
            time_text.set_text('')
            return pole, time_text
        
        def update(frame):
            """Update animation frame."""
            # Get state at this frame
            theta = x_traj[frame, 0]
            
            # Update pole position (pole rotates around cart center)
            pole_x = [0, self.l * np.sin(theta)]
            pole_y = [0, self.l * np.cos(theta)]
            pole.set_data(pole_x, pole_y)
            
            # Update time text
            time = frame * self.dt
            time_text.set_text(f'Time: {time:.2f} s\nAngle: {theta:.3f} rad')
            
            return pole, time_text
        
        anim = FuncAnimation(fig, update, frames=N+1, init_func=init,
                           blit=True, interval=interval, repeat=True)
        
        plt.show()
        
        # Save animation if path is provided
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
            print(f"Animation saved!")
        
        return x_traj, u_traj, anim

if __name__ == "__main__":
    # Create cart-pole environment
    inv_pend = InvertedPendulum(dt=0.1, sym_type='SX')
    
    # Example usage: simulate with zero control input
    theta_bound = np.pi/6
    omega_bound = 1.0
    theta0 = np.random.uniform(-theta_bound, theta_bound)
    omega0 = np.random.uniform(-omega_bound, omega_bound)
    x0 = [theta0, omega0]
    print (f"Initial state: {x0}")
    N = 10  # number of simulation steps
    control_policy = lambda x: np.array([0.0])  # zero control input
    
    inv_pend.close_loop_simulation(x0)
    # Animate
    # x0 = [0.0, 0.0, np.pi/6, 0.0]  # Small initial angle
    x_traj, u_traj, anim = inv_pend.animate(
        x0, control_policy=None, N=10, 
        interval=50,  # 50ms between frames
        # save_path='inv_pend.gif'  # Optional: save to file
)