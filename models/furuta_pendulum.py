import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are


class FurutaPendulum:
    def __init__(self, dt=0.01, sym_type='SX'):
        self.dt = dt
        self.sym_type = sym_type

        # Physical parameters from Parameters_FurutaPendulum_Coulomb.m
        self.grav = 9.81
        self.k_factor = 2.0

        self.kSteller = 1.0983
        self.kM = 0.0236
        self.RM = 0.5200

        self.b1vis = 6.0700e-04
        self.b1coul = 0.0104
        self.hJ0 = 0.0021
        self.L1 = 0.1215

        self.m2 = 0.0461
        self.l2 = 0.0790
        self.b2vis = 2.6830e-05
        self.b2coul = 7.5418e-04
        self.hJ2 = 3.7054e-04

        self.dynamics = self.dynamics_func()
        self.step = self.step_func()
        self.lin_dyn = self.linearized_dynamics()
        
        N = 10  # MPC horizon
        self.define_simple_MPC_control(N, seek_x0=False)  # Define the MPC controller

    def _mass_matrix(self, theta2):
        s2 = ca.sin(theta2)
        c2 = ca.cos(theta2)
        m2L1l2 = self.m2 * self.L1 * self.l2

        return ca.vertcat(
            ca.horzcat(self.hJ0 + self.hJ2 * s2**2, m2L1l2 * c2),
            ca.horzcat(m2L1l2 * c2, self.hJ2),
        )

    def dynamics_func(self):
        x = getattr(ca, self.sym_type).sym('x', 4)
        u = getattr(ca, self.sym_type).sym('u', 1)

        theta1 = x[0]
        theta1p = x[1]
        theta2 = x[2]
        theta2p = x[3]

        s2 = ca.sin(theta2)
        s2_2 = ca.sin(2 * theta2)
        J2_s2_2 = self.hJ2 * s2_2
        m2L1l2 = self.m2 * self.L1 * self.l2

        rhs1 = (
            (-self.b1vis - (self.kM**2) / self.RM) * theta1p
            - J2_s2_2 * theta1p * theta2p
            + m2L1l2 * s2 * theta2p**2
            + (self.kM / self.RM) * self.kSteller * u[0]
            - self.b1coul * ca.tanh(self.k_factor * theta1p)
        )

        rhs2 = (
            -self.b2vis * theta2p
            + 0.5 * J2_s2_2 * theta1p**2
            - self.b2coul * ca.tanh(self.k_factor * theta2p)
            - self.l2 * self.m2 * s2 * self.grav
        )

        rhs = ca.vertcat(rhs1, rhs2)
        ddq = ca.solve(self._mass_matrix(theta2), rhs)

        xdot = ca.vertcat(theta1p, ddq[0], theta2p, ddq[1])
        return ca.Function('dynamics', [x, u], [xdot])

    def step_func(self):
        x0 = getattr(ca, self.sym_type).sym('x0', 4)
        u0 = getattr(ca, self.sym_type).sym('u0', 1)
        M = 4
        DT = self.dt / M
        X = x0
        for _ in range(M):
            k1 = self.dynamics(X, u0)
            k2 = self.dynamics(X + DT / 2 * k1, u0)
            k3 = self.dynamics(X + DT / 2 * k2, u0)
            k4 = self.dynamics(X + DT * k3, u0)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = X

        return ca.Function('step', [x0, u0], [x_next])

    def linearized_dynamics(self):
        x = getattr(ca, self.sym_type).sym('x', 4)
        u = getattr(ca, self.sym_type).sym('u', 1)
        xdot = self.dynamics(x, u)
        A_sym = ca.jacobian(xdot, x)
        B_sym = ca.jacobian(xdot, u)
        return ca.Function('linearized_dynamics', [x, u], [A_sym, B_sym])

    def define_simple_MPC_control(self, N, seek_x0=False):
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
        xr = ca.DM([0.0, 0.0, 0.0, 0.0])  # desired state (upright position)
        Q = ca.diag(ca.DM([10, 5, 100, 1]))  # state cost weights
        R = ca.diag(ca.DM([0.01]))  # control cost weight
        A, B = self.lin_dyn(xr, ca.DM([0]))  # linearized dynamics around the upright position
        E = solve_continuous_are(A.full(), B.full(), Q.full(), R.full())
        
        self.theta_1_bound = ca.inf
        self.omega_1_bound = 20.0
        self.theta_2_bound = ca.inf
        self.omega_2_bound = 20.0
        
        self.u_bound = 10.0
        
        x_next = self.step(x, u)
        l = (x - xr).T @ Q @ (x - xr) + u.T @ R @ u 
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
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
        self.lbw += [-self.theta_1_bound, -self.omega_1_bound, -self.theta_2_bound, -self.omega_2_bound]
        self.ubw += [self.theta_1_bound, self.omega_1_bound, self.theta_2_bound, self.omega_2_bound]
        if not seek_x0:
            g += [Xk - x0]  # initial state constraint
            self.lbg += [0.0 for _ in range(4)]
            self.ubg += [0.0 for _ in range(4)]
            self.w0 += [0.0 for _ in range(4)]
        else:
            # Randowmize initial state in the bounds
            self.w0 += [0.0 for _ in range(4)]

        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            if seek_x0 and k == 0:
                self.lbw += [-self.u_bound]  # control limits
                self.ubw += [self.u_bound]
                self.w0 += [0.0]     # initial guess
            else:
                self.lbw += [-self.u_bound]  # control limits
                self.ubw += [self.u_bound]
                self.w0 += [0.0]     # initial guess

            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            self.lbw += [-self.theta_1_bound, -self.omega_1_bound, -self.theta_2_bound, -self.omega_2_bound]
            self.ubw += [self.theta_1_bound, self.omega_1_bound, self.theta_2_bound, self.omega_2_bound]
            self.w0 += [0.0 for _ in range(4)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(4)]
            self.ubg += [0.0 for _ in range(4)]
            
            J += l_k  # accumulate cost
        J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)  # terminal cost
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': x0}
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 5000,
                "tol": 1e-8,
                "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
                "linear_solver": "ma27",
            }
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        
    def solve_MPC(self, x0, ret_seq=False, plot_results=False):
        ''' Solve the MPC problem for a given initial state x0. '''
        sol = self.solver(
            x0=self.w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=x0
        )
        
        x_opt, u_opt = self.extract_traj(sol['x'])
        if not self.solver.stats()['success']:
            print("stocazzo")
            return None
        # Return the optimal control sequence
        
        if plot_results:
            x_pred = np.array(x_opt.full())          # shape: (2, N+1)
            u_pred = np.array(u_opt.full()).flatten()  # shape: (N,)

            tx = np.arange(x_pred.shape[1]) * self.dt
            tu = np.arange(u_pred.shape[0]) * self.dt

            if not hasattr(self, "_opt_plot") or self._opt_plot is None:
                fig, axes = plt.subplots(5, 1, figsize=(9, 8), sharex=False)
                self._opt_plot = (fig, axes)
            else:
                fig, axes = self._opt_plot
                for ax in axes:
                    ax.clear()

            axes[0].plot(tx, x_pred[0], "g-o", linewidth=1.5, markersize=3)
            axes[0].axhline(0.0, color="k", linestyle="--", alpha=0.4)
            axes[0].set_ylabel(r"$\\theta_1$ (rad)")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(tx, x_pred[1], "c-o", linewidth=1.5, markersize=3)
            axes[1].set_ylabel(r"$\\omega_1$ (rad/s)")
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(tx, x_pred[2], "m-o", linewidth=1.5, markersize=3)
            axes[2].axhline(0.0, color="k", linestyle="--", alpha=0.4)
            axes[2].set_ylabel(r"$\\theta_2$ (rad)")
            axes[2].grid(True, alpha=0.3)   
            
            axes[3].plot(tx, x_pred[3], "b-o", linewidth=1.5, markersize=3)
            axes[3].set_ylabel(r"$\\omega_2$ (rad/s)")
            axes[3].grid(True, alpha=0.3)

            axes[4].step(tu, u_pred, "k-", where="post", linewidth=1.8)
            axes[4].set_ylabel(r"$u$ (N m)")
            axes[4].set_xlabel("Time (s)")
            axes[4].grid(True, alpha=0.3)

            fig.suptitle("MPC Optimization (Predicted Trajectory)")
            fig.tight_layout()
            plt.show()
            
        if ret_seq:
            return u_opt.full().flatten()  # return the full control sequence as a 1D array
        
        return u_opt.full().flatten()[0]  # return only the first control input
    
    def solve_MPC_for_initial_states(self, x0, plot_results=False):
        ''' Solve the MPC problem for a given initial state x0. '''
        sol = self.solver(
            x0=self.w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=x0
        )
        
        x_opt, u_opt = self.extract_traj(sol['x'])
        if not self.solver.stats()['success']:
            raise RuntimeError("MPC solver failed to find a solution.")
        # Return the optimal control sequence
        
        if plot_results:
            x_pred = np.array(x_opt.full())          # shape: (2, N+1)
            u_pred = np.array(u_opt.full()).flatten()  # shape: (N,)

            tx = np.arange(x_pred.shape[1]) * self.dt
            tu = np.arange(u_pred.shape[0]) * self.dt

            if not hasattr(self, "_opt_plot") or self._opt_plot is None:
                fig, axes = plt.subplots(5, 1, figsize=(9, 8), sharex=False)
                self._opt_plot = (fig, axes)
            else:
                fig, axes = self._opt_plot
                for ax in axes:
                    ax.clear()

            axes[0].plot(tx, x_pred[0], "g-o", linewidth=1.5, markersize=3)
            axes[0].axhline(0.0, color="k", linestyle="--", alpha=0.4)
            axes[0].set_ylim(-self.theta_bound*1.2, self.theta_bound*1.2)
            axes[0].set_ylabel(r"$\theta_1$ (rad)")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(tx, x_pred[1], "c-o", linewidth=1.5, markersize=3)
            axes[1].set_ylim(-self.omega_bound*1.2, self.omega_bound*1.2)
            axes[1].set_ylabel(r"$\omega_1$ (rad/s)")
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(tx, x_pred[2], "m-o", linewidth=1.5, markersize=3)
            axes[2].axhline(0.0, color="k", linestyle="--", alpha=0.4)
            axes[2].set_ylim(-self.theta_bound*1.2, self.theta_bound*1.2)
            axes[2].set_ylabel(r"$\theta_2$ (rad)")
            axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(tx, x_pred[3], "b-o", linewidth=1.5, markersize=3)
            axes[3].set_ylim(-self.omega_bound*1.2, self.omega_bound*1.2)
            axes[3].set_ylabel(r"$\omega_2$ (rad/s)")
            axes[3].grid(True, alpha=0.3)   

            axes[4].plot(tu, u_pred, "k-", linewidth=1.8)
            axes[4].set_ylim(-self.u_bound*1.2, self.u_bound*1.2)
            axes[4].set_ylabel(r"$u$ (N m)")
            axes[4].set_xlabel("Time (s)")
            axes[4].grid(True, alpha=0.3)

            fig.suptitle("MPC Optimization (Predicted Trajectory)")
            fig.tight_layout()
            plt.show()
            
        # Return the first state
        return x_opt[:, 0].full().flatten()  # return the first state in the optimal trajectory

    def animate(self, x0, Nsim=120, control_policy=None, interval=40, repeat=False, save_path=None, fps=25, show=True):
        """Run a closed-loop simulation and animate the Furuta pendulum motion.

        Parameters
        ----------
        x0 : array_like
            Initial state [theta1, omega1, theta2, omega2].
        Nsim : int, optional
            Number of simulation steps.
        control_policy : callable or array_like, optional
            Control policy passed to ``close_loop_simulation``.
        interval : int, optional
            Delay between frames in milliseconds.
        repeat : bool, optional
            Whether the animation should repeat.
        save_path : str or None, optional
            If provided, save animation to this path (e.g. ``.gif`` or ``.mp4``).
        fps : int, optional
            Output frame rate when saving animation.
        show : bool, optional
            Whether to display the animation window.

        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (Nsim+1, 4).
        u_traj : np.ndarray
            Control trajectory of shape (Nsim,).
        anim : matplotlib.animation.FuncAnimation
            Animation object.
        """
        x_traj, u_traj = self.close_loop_simulation(
            x0=x0,
            Nsim=Nsim,
            control_policy=control_policy,
            plot_results=False,
        )

        theta1 = x_traj[:, 0]
        theta2 = x_traj[:, 2]

        # Planar coordinates for top view (arm in x-y plane).
        arm_x = self.L1 * np.cos(theta1)
        arm_y = self.L1 * np.sin(theta1)

        # Side-view projection along radial arm direction.
        pend_r = self.L1 + self.l2 * np.sin(theta2)
        pend_z = -self.l2 * np.cos(theta2)

        r_lim = 1.25 * (self.L1 + self.l2)
        z_lim = 1.25 * self.l2

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Top view setup.
        ax_top = axes[0]
        ax_top.set_title("Top View (x-y)")
        ax_top.set_aspect("equal")
        ax_top.set_xlim(-r_lim, r_lim)
        ax_top.set_ylim(-r_lim, r_lim)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xlabel("x (m)")
        ax_top.set_ylabel("y (m)")
        base_patch = Rectangle((-0.01, -0.01), 0.02, 0.02, color="k", alpha=0.8)
        ax_top.add_patch(base_patch)
        top_arm_line, = ax_top.plot([], [], "tab:blue", lw=3)
        top_tip, = ax_top.plot([], [], "o", color="tab:blue", ms=6)

        # Side view setup.
        ax_side = axes[1]
        ax_side.set_title("Side View (r-z)")
        ax_side.set_aspect("equal")
        ax_side.set_xlim(-0.2 * self.L1, r_lim)
        ax_side.set_ylim(-z_lim, 0.25 * z_lim)
        ax_side.grid(True, alpha=0.3)
        ax_side.set_xlabel("radial distance r (m)")
        ax_side.set_ylabel("z (m)")
        side_arm_line, = ax_side.plot([], [], "tab:orange", lw=3)
        side_pend_line, = ax_side.plot([], [], "tab:red", lw=3)
        side_bob, = ax_side.plot([], [], "o", color="tab:red", ms=7)

        time_text = ax_top.text(
            0.03,
            0.95,
            "",
            transform=ax_top.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        def _init():
            top_arm_line.set_data([], [])
            top_tip.set_data([], [])
            side_arm_line.set_data([], [])
            side_pend_line.set_data([], [])
            side_bob.set_data([], [])
            time_text.set_text("")
            return top_arm_line, top_tip, side_arm_line, side_pend_line, side_bob, time_text

        def _update(i):
            # Top view arm segment.
            top_arm_line.set_data([0.0, arm_x[i]], [0.0, arm_y[i]])
            top_tip.set_data([arm_x[i]], [arm_y[i]])

            # Side view: horizontal arm and pendulum projection.
            side_arm_line.set_data([0.0, self.L1], [0.0, 0.0])
            side_pend_line.set_data([self.L1, pend_r[i]], [0.0, pend_z[i]])
            side_bob.set_data([pend_r[i]], [pend_z[i]])

            time_text.set_text(f"t = {i * self.dt:.2f} s")
            return top_arm_line, top_tip, side_arm_line, side_pend_line, side_bob, time_text

        anim = FuncAnimation(
            fig,
            _update,
            frames=x_traj.shape[0],
            init_func=_init,
            interval=interval,
            blit=True,
            repeat=repeat,
        )

        fig.tight_layout()

        if save_path is not None:
            anim.save(save_path, fps=fps)

        if show:
            plt.show()

        return x_traj, u_traj, anim
        
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
        x_traj = np.zeros((Nsim + 1, 4))
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
            
            # Check if is a scalar
            if u_k is None:
                print("Control is None")
                continue
            
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
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 10))
        title = "Inverted-Pendulum Simulation"
        
        # Angle
        axes[0].plot(time, x_traj[:, 0], 'g-', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel(r"$\theta_1$ (rad)", fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Angular velocity
        axes[1].plot(time, x_traj[:, 1], 'c-', linewidth=2)
        axes[1].set_ylabel(r"$\omega_1$ (rad/s)", fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Angle 2
        axes[2].plot(time, x_traj[:, 2], 'm-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel(r"$\theta_2$ (rad)", fontsize=10)
        axes[2].grid(True, alpha=0.3)

        # Angular velocity 2
        axes[3].plot(time, x_traj[:, 3], 'b-', linewidth=2)
        axes[3].set_ylabel(r"$\omega_2$ (rad/s)", fontsize=10)
        axes[3].grid(True, alpha=0.3) 
        
        # Control
        axes[4].step(time_u, u_traj, 'k-', where='post', linewidth=2)
        axes[4].set_ylabel(r"$u$ (N m)", fontsize=10)
        axes[4].set_xlabel("Time (s)", fontsize=10)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj
    
    
if __name__ == "__main__":
    # Create cart-pole environment
    fur_pend = FurutaPendulum(dt=0.1, sym_type='SX')
    
    # Example usage: simulate with zero control input
    theta_1_bound = np.pi
    omega_1_bound = 1.0
    theta_2_bound = np.pi
    omega_2_bound = 1.0
    theta1_0 = np.random.uniform(-theta_1_bound, theta_1_bound)
    omega1_0 = np.random.uniform(-omega_1_bound, omega_1_bound)
    theta2_0 = np.random.uniform(-theta_2_bound, theta_2_bound)
    omega2_0 = np.random.uniform(-omega_2_bound, omega_2_bound)
    x0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    # print (f"Initial state: {x0_dump}")
    N = 10  # number of simulation steps
    control_policy = lambda x: np.array([0.0])  # zero control input
    
    # X0 = inv_pend.solve_extreme_x0(N, plot_results=False)
    print(x0)
    fur_pend.close_loop_simulation(x0, Nsim=20)
    # Animate
    # x0 = [0.0, 0.0, np.pi/6, 0.0]  # Small initial angle
    # x_traj, u_traj, anim = fur_pend.animate(
    #     x0, control_policy=None)  # 50ms between frames
        # save_path='inv_pend.gif'  # Optional: save to file