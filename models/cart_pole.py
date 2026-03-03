import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are

# Class implementing the cart-pole environment
class CartPole:
    def __init__(self, dt=0.05, sym_type='SX'):
        self.dt = dt  # time step
        self.g = 9.81  # gravity
        self.l = 0.5   # length of the pole
        self.m = 0.08   # mass of the pole
        self.M = 0.8   # mass of the cart
        self.sym_type = sym_type  # symbolic type for CasADi (SX or MX)
        
        self.dynamics = self.dynamics_func()  # CasADi function for dynamics
        self.step = self.step_func()  # CasADi function for RK4 integration
        self.lin_dyn = self.linearized_dynamics()  # CasADi function for linearized dynamics
        
        self.N = 20  # MPC horizon
        self.define_simple_MPC_control(self.N)  # Define the MPC controller

    def dynamics_func(self):
        ''' Computes the time derivative of the state given the current state and control input. '''
        # x = [position, velocity, angle, angular_velocity]
        # u = [force]
        
        x = getattr(ca, self.sym_type).sym('x', 4)  # state vector
        u = getattr(ca, self.sym_type).sym('u', 1)  # control input
        
        s = ca.sin(x[2])
        c = ca.cos(x[2])
        
        x_dot = x[1]
        theta_dot = x[3]
        
        common_denominator = self.M + self.m - self.m * c**2
        
        x_ddot = (u[0] + self.m * s * (-theta_dot**2 * self.l + self.g * c)) / common_denominator
        theta_ddot = (u[0] * c - self.m * s * c * theta_dot**2 * self.l + (self.M + self.m) * self.g * s) / (self.l * common_denominator)
        
        return ca.Function('dynamics', [x, u], [ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)])
    
    def step_func(self):
        ''' Computes the next state given the current state and control input using RK4 integration. '''
        x0 = getattr(ca, self.sym_type).sym('x0', 4)
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
        x = getattr(ca, self.sym_type).sym('x', 4)  # state vector
        u = getattr(ca, self.sym_type).sym('u', 1)  # control input
        
        s = ca.sin(x[2])
        c = ca.cos(x[2])
        
        x_dot = x[1]
        theta_dot = x[3]
        
        common_denominator = self.M + self.m - self.m * c**2
        
        x_ddot = (u[0] + self.m * s * (-theta_dot**2 * self.l + self.g * c)) / common_denominator
        theta_ddot = (u[0] * c - self.m * s * c * theta_dot**2 * self.l + (self.M + self.m) * self.g * s) / (self.l * common_denominator)
        A_sym = ca.jacobian(ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot), x)
        B_sym = ca.jacobian(ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot), u)
        
        return ca.Function('linearized_dynamics', [x, u], [A_sym, B_sym])
    
    def define_simple_MPC_control(self, N):
        ''' A simple MPC controller that tries to stabilize the pole at the upright position. '''
        # Define optimization variables and parameters
        x = ca.SX.sym('x', 4)  # state variable
        u = ca.SX.sym('u', 1)  # control variable
        x0 = ca.SX.sym('x0', 4)  # initial state parameter
        
        X = ca.SX.sym('X', 4, N+1)  # state trajectory
        U = ca.SX.sym('U', 1, N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        # Define cost function (quadratic cost on state deviation and control effort)
        xr = ca.DM([0.0, 0, 0, 0])  # desired state (upright position)
        Q = ca.diag(ca.DM([100, 1, 30, 1]))  # state cost weights
        R = ca.diag(ca.DM([0.01]))  # control cost weight
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
        Xk = ca.SX.sym('X0', 4)  # initial state variable
        w += [Xk]
        self.lbw += [-p_bound, -v_bound, -theta_bound, -omega_bound]
        self.ubw += [p_bound, v_bound, theta_bound, omega_bound]
        self.w0 += [0.0 for _ in range(4)]
        g += [Xk - x0]  # initial state constraint
        self.lbg += [0.0 for _ in range(4)]
        self.ubg += [0.0 for _ in range(4)]
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            self.lbw += [-25.0]  # control limits
            self.ubw += [25.0]
            self.w0 += [0.0]     # initial guess
            
            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            self.lbw += [-p_bound, -v_bound, -theta_bound, -omega_bound]
            self.ubw += [p_bound, v_bound, theta_bound, omega_bound]
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
    
    def solve_MPC(self, x0):
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
        return u_opt.full().flatten()[0]  # return only the first control input
        
    def close_loop_simulation(self, x0, control_policy=None, plot_results=True):
        """Simulate the cart-pole system and plot the results.
        
        Parameters
        ----------
        x0 : array_like
            Initial state [position, velocity, angle, angular_velocity].
        control_policy : callable or array_like
            Either a function that takes state and returns control u = policy(x),
            or an array of control inputs for each time step. If None, the MPC controller will be used.
        N : int
            Number of simulation steps.
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
        x_traj = np.zeros((self.N + 1, 4))
        u_traj = np.zeros(self.N)
        
        x_traj[0] = np.array(x0).flatten()
        
        # Simulate forward in time
        for k in range(self.N):
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
        time = np.arange(self.N + 1) * self.dt
        time_u = np.arange(self.N) * self.dt
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 10))
        title = "Cart-Pole Simulation"
        
        # Position
        axes[0].plot(time, x_traj[:, 0], 'b-', linewidth=2)
        axes[0].set_ylabel('Position (m)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        
        # Velocity
        axes[1].plot(time, x_traj[:, 1], 'r-', linewidth=2)
        axes[1].set_ylabel('Velocity (m/s)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Angle
        axes[2].plot(time, x_traj[:, 2], 'g-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (rad)', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Angular velocity
        axes[3].plot(time, x_traj[:, 3], 'c-', linewidth=2)
        axes[3].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Control
        axes[4].step(time_u, u_traj, 'k-', where='post', linewidth=2)
        axes[4].set_ylabel('Control Force (N)', fontsize=10)
        axes[4].set_xlabel('Time (s)', fontsize=10)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj
    
    def animate(self, x0, control_policy, interval=50, save_path=None, title="Cart-Pole Animation"):
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
            State trajectory of shape (N+1, 4).
        u_traj : np.ndarray
            Control trajectory of shape (N,).
        anim : matplotlib.animation.FuncAnimation
            The animation object.
        """
        # First, simulate to get the trajectory
        x_traj = np.zeros((self.N + 1, 4))
        u_traj = np.zeros(self.N)
        
        x_traj[0] = np.array(x0).flatten()
        
        for k in range(self.N):
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
        x_min = np.min(x_traj[:, 0]) - 1.5
        x_max = np.max(x_traj[:, 0]) + 1.5
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
        
        # Cart dimensions
        cart_width = 0.3
        cart_height = 0.2
        
        # Initialize objects
        cart = Rectangle((0, 0), cart_width, cart_height, 
                        fill=True, color='blue', ec='black', linewidth=2)
        ax.add_patch(cart)
        
        pole, = ax.plot([], [], 'r-', linewidth=4, marker='o', markersize=8, markerfacecolor='red')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def init():
            """Initialize animation."""
            cart.set_xy((0, 0))
            pole.set_data([], [])
            time_text.set_text('')
            return cart, pole, time_text
        
        def update(frame):
            """Update animation frame."""
            # Get state at this frame
            pos = x_traj[frame, 0]
            theta = x_traj[frame, 2]
            
            # Update cart position
            cart.set_xy((pos - cart_width/2, 0))
            
            # Update pole position (pole rotates around cart center)
            pole_x = [pos, pos + self.l * np.sin(theta)]
            pole_y = [cart_height/2, cart_height/2 + self.l * np.cos(theta)]
            pole.set_data(pole_x, pole_y)
            
            # Update time text
            time = frame * self.dt
            time_text.set_text(f'Time: {time:.2f} s\nAngle: {theta:.3f} rad\nPosition: {pos:.3f} m')
            
            return cart, pole, time_text
        
        anim = FuncAnimation(fig, update, frames=self.N+1, init_func=init,
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
    cart_pole = CartPole(dt=0.1, sym_type='SX')
    
    # Example usage: simulate with zero control input
    p_bound = 1.0
    v_bound = 2.0
    theta_bound = np.pi/6
    omega_bound = 1.0
    np.random.seed(42)  # for reproducibility
    p0 = np.random.uniform(-p_bound, p_bound)
    v0 = np.random.uniform(-v_bound, v_bound)
    theta0 = np.random.uniform(-theta_bound, theta_bound)
    omega0 = np.random.uniform(-omega_bound, omega_bound)
    x0 = [p0, v0, theta0, omega0]
    print (f"Initial state: {x0}")
    control_policy = lambda x: np.array([0.0])  # zero control input
    
    # Close-loop simulation with MPC control
    cart_pole.close_loop_simulation(x0)
    # Animate
    # x0 = [0.0, 0.0, np.pi/6, 0.0]  # Small initial angle
    x_traj, u_traj, anim = cart_pole.animate(
        x0, control_policy=None, 
        interval=50,  # 50ms between frames
        # save_path='cart_pole.gif'  # Optional: save to file
)