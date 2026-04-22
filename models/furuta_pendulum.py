import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import solve_discrete_are, solve_continuous_are


class FurutaPendulum:
    def __init__(self, dt=0.02, sym_type='SX'):
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
        y_ref = ca.DM([np.pi, 0.0, np.pi, 0.0, 0.0])  # desired state and control
        y_ref_e = ca.DM([np.pi, 0.0, np.pi, 0.0])  # desired terminal state
        W_0 = 1e-2*ca.DM.eye(4)  # initial state cost weight
        W_x = ca.diag(ca.DM([20, 5, 100, 1]))  # state cost weights
        W_u = ca.diag(ca.DM([0.01]))  # control cost weight
        W = ca.blockcat([[W_x, np.zeros((4, 1))], [np.zeros((1, 4)), W_u]])  # combined state-control cost weight
        W_e = W_x  # terminal state cost weight
        # A, B = self.lin_dyn(xr, ca.DM([0]))  # linearized dynamics around the upright position
        # E = solve_continuous_are(A.full(), B.full(), Q.full(), R.full())
        
        self.theta_1_bound = ca.inf
        self.omega_1_bound = ca.inf
        self.theta_2_bound = ca.inf
        self.omega_2_bound = ca.inf
        
        self.u_bound = 10.0
        
        x_next = self.step(x, u)
        F = ca.Function('F', [x, u], [x_next])
        
        # Cost function
        # Nonlinear output maps from your acados model
        y_expr = ca.vertcat(
            ca.pi * (1 + ca.cos(x[0] / 2)),
            x[1],
            ca.pi * (1 + ca.cos(x[2] / 2)),
            x[3],
            u
        )

        y_expr_e = ca.vertcat(
            ca.pi * (1 + ca.cos(x[0] / 2)),
            x[1],
            ca.pi * (1 + ca.cos(x[2] / 2)),
            x[3]
        )

        # Initial cost: LINEAR_LS with Vx_0 = I, Vu_0 = 0, yref_0 = x0
        e0 = x - x0
        cost_0 = 0.5 * ca.mtimes([e0.T, W_0, e0])

        # Path cost: NONLINEAR_LS
        ep = y_expr - y_ref
        cost_path = 0.5 * ca.mtimes([ep.T, W, ep])

        # Terminal cost: NONLINEAR_LS
        ee = y_expr_e - y_ref_e
        cost_terminal = 0.5 * ca.mtimes([ee.T, W_e, ee])
        
        # Create a CasADi function for the stage cost
        init_cost = ca.Function('init_cost', [x, x0], [cost_0])
        stage_cost = ca.Function('stage_cost', [x, u], [cost_path])
        terminal_cost = ca.Function('terminal_cost', [x], [cost_terminal])
        
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

        # Initial cost
        J += init_cost(Xk, x0)
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            self.lbw += [-self.u_bound]  # control limits
            self.ubw += [self.u_bound]
            self.w0 += [0.0]     # initial guess

            Xk_next = F(Xk, uk)
            
            J += stage_cost(Xk, uk)  # stage cost
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            self.lbw += [-self.theta_1_bound, -self.omega_1_bound, -self.theta_2_bound, -self.omega_2_bound]
            self.ubw += [self.theta_1_bound, self.omega_1_bound, self.theta_2_bound, self.omega_2_bound]
            self.w0 += [0.0 for _ in range(4)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(4)]
            self.ubg += [0.0 for _ in range(4)]
            
        J += terminal_cost(Xk)  # terminal cost
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

    def animate(self, x0, Nsim=120, control_policy=None, interval=40, repeat=False,
        save_path=None, fps=25, show=True, show_desired=True, desired_state=None,
        show_motion_plane=True):
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
        show_desired : bool, optional
            Whether to overlay the desired configuration as a translucent target pose.
        desired_state : array_like or None, optional
            Desired state [theta1, omega1, theta2, omega2]. If None, defaults to
            [pi, 0, pi, 0].
        show_motion_plane : bool, optional
            Whether to show the rotating vertical plane in which the pendulum moves.

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
        # 3D coordinates.
        arm_x = self.L1 * np.cos(theta1)
        arm_y = self.L1 * np.sin(theta1)
        arm_z = np.zeros_like(arm_x)

        bob_x = (self.L1 + self.l2 * np.sin(theta2)) * np.cos(theta1)
        bob_y = (self.L1 + self.l2 * np.sin(theta2)) * np.sin(theta1)
        bob_z = -self.l2 * np.cos(theta2)

        lim = 1.25 * (self.L1 + self.l2)

        if desired_state is None:
            desired_state = np.array([np.pi, 0.0, np.pi, 0.0], dtype=float)
        desired_state = np.array(desired_state, dtype=float).flatten()
        if desired_state.shape[0] != 4:
            raise ValueError("desired_state must be an array-like with 4 elements.")

        desired_theta1 = desired_state[0]
        desired_theta2 = desired_state[2]
        desired_arm_x = self.L1 * np.cos(desired_theta1)
        desired_arm_y = self.L1 * np.sin(desired_theta1)
        desired_arm_z = 0.0
        desired_bob_x = (self.L1 + self.l2 * np.sin(desired_theta2)) * np.cos(desired_theta1)
        desired_bob_y = (self.L1 + self.l2 * np.sin(desired_theta2)) * np.sin(desired_theta1)
        desired_bob_z = -self.l2 * np.cos(desired_theta2)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Furuta Pendulum 3D Animation")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, 0.5 * lim)
        ax.set_box_aspect((1.0, 1.0, 0.8))
        ax.view_init(elev=24, azim=38)
        ax.grid(True, alpha=0.3)

        def _plane_vertices(theta):
            e_r = np.array([np.cos(theta), np.sin(theta), 0.0])
            e_z = np.array([0.0, 0.0, 1.0])
            p_arm = self.L1 * e_r
            r_half = 1.05 * self.l2
            z_half = 1.05 * self.l2
            v1 = p_arm - r_half * e_r - z_half * e_z
            v2 = p_arm + r_half * e_r - z_half * e_z
            v3 = p_arm + r_half * e_r + z_half * e_z
            v4 = p_arm - r_half * e_r + z_half * e_z
            return [v1, v2, v3, v4]

        # Base marker at the rotary joint.
        base_marker, = ax.plot([0.0], [0.0], [0.0], "o", color="k", ms=6)
        desired_arm_line = None
        desired_pend_line = None
        desired_bob_marker = None
        if show_desired:
            desired_arm_line, = ax.plot(
                [0.0, desired_arm_x],
                [0.0, desired_arm_y],
                [0.0, desired_arm_z],
                "--",
                color="tab:green",
                lw=2.2,
                alpha=0.35,
            )
            desired_pend_line, = ax.plot(
                [desired_arm_x, desired_bob_x],
                [desired_arm_y, desired_bob_y],
                [desired_arm_z, desired_bob_z],
                "--",
                color="tab:green",
                lw=2.2,
                alpha=0.35,
            )
            desired_bob_marker, = ax.plot(
                [desired_bob_x],
                [desired_bob_y],
                [desired_bob_z],
                "o",
                color="tab:green",
                ms=6,
                alpha=0.35,
            )
        arm_line, = ax.plot([], [], [], color="tab:blue", lw=3)
        pend_line, = ax.plot([], [], [], color="tab:red", lw=3)
        bob_marker, = ax.plot([], [], [], "o", color="tab:red", ms=7)
        trace_line, = ax.plot([], [], [], color="tab:gray", lw=1.2, alpha=0.6)
        motion_plane = None
        if show_motion_plane:
            motion_plane = Poly3DCollection(
                [_plane_vertices(theta1[0])],
                facecolors="tab:blue",
                edgecolors="tab:blue",
                linewidths=0.7,
                linestyles=":",
                alpha=0.10,
            )
            ax.add_collection3d(motion_plane)

        desired_label = " (with desired pose)" if show_desired else ""
        time_text = ax.text2D(
            0.03,
            0.95,
            f"Target{desired_label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        def _init():
            arm_line.set_data([], [])
            arm_line.set_3d_properties([])
            pend_line.set_data([], [])
            pend_line.set_3d_properties([])
            bob_marker.set_data([], [])
            bob_marker.set_3d_properties([])
            trace_line.set_data([], [])
            trace_line.set_3d_properties([])
            if motion_plane is not None:
                motion_plane.set_verts([_plane_vertices(theta1[0])])
            time_text.set_text(f"t = 0.00 s{desired_label}")
            return base_marker, arm_line, pend_line, bob_marker, trace_line, time_text

        def _update(i):
            arm_line.set_data([0.0, arm_x[i]], [0.0, arm_y[i]])
            arm_line.set_3d_properties([0.0, arm_z[i]])

            pend_line.set_data([arm_x[i], bob_x[i]], [arm_y[i], bob_y[i]])
            pend_line.set_3d_properties([arm_z[i], bob_z[i]])

            bob_marker.set_data([bob_x[i]], [bob_y[i]])
            bob_marker.set_3d_properties([bob_z[i]])

            trace_line.set_data(bob_x[: i + 1], bob_y[: i + 1])
            trace_line.set_3d_properties(bob_z[: i + 1])

            if motion_plane is not None:
                motion_plane.set_verts([_plane_vertices(theta1[i])])

            time_text.set_text(f"t = {i * self.dt:.2f} s{desired_label}")
            return base_marker, arm_line, pend_line, bob_marker, trace_line, time_text

        anim = FuncAnimation(
            fig,
            _update,
            frames=x_traj.shape[0],
            init_func=_init,
            interval=interval,
            blit=False,
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
    fur_pend = FurutaPendulum(dt=0.05, sym_type='SX')
    
    # Example usage: simulate with zero control input
    theta_1_bound = np.pi
    omega_1_bound = 1.0
    theta_2_bound = np.pi
    omega_2_bound = 1.0
    theta1_0 = np.random.uniform(-theta_1_bound, theta_1_bound)
    # x0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    x0 = [theta1_0, 0, 0.0, 0]
    # print (f"Initial state: {x0_dump}")
    # N = 10  # number of simulation steps
    # control_policy = lambda x: np.array([0.0])  # zero control input
    
    # X0 = inv_pend.solve_extreme_x0(N, plot_results=False)
    print(x0)
    fur_pend.close_loop_simulation(x0, Nsim=30)
    # Animate
    # x0 = [0.0, 0.0, np.pi/6, 0.0]  # Small initial angle
    # x_traj, u_traj, anim = fur_pend.animate(
    #     x0, control_policy=None)  # 50ms between frames
        # save_path='inv_pend.gif'  # Optional: save to file