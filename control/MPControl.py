import numpy as np
import cvxpy as cp
from control.BaseControl import BaseControl  # Assuming you have a BaseControl class
from control.DSLPIDControl import DSLPIDControl  # Assuming you have a BaseControl class

from scipy.spatial.transform import Rotation


class MPControl(BaseControl):
    def __init__(self, drone_model, g=9.81, control_timestep=0.01):
        super().__init__(drone_model=drone_model, g=g)
        self.g = g
        self.mass = 0.027
        self.control_timestep = control_timestep
        self.N = 10  # Prediction horizon
        self.nx = 6  # Number of states
        self.nu = 3  # Number of inputs
        self.Q = np.diag([10, 10, 10, 1, 1, 1])  # State weighting matrix
        self.R = np.diag([0.1, 0.1, 0.1])        # Input weighting matrix
        self.u_min = np.array([-2.0, -2.0, -2.0])  # Min accelerations (m/s^2)
        self.u_max = np.array([2.0, 2.0, 2.0])     # Max accelerations (m/s^2)
        self.inner_pid = DSLPIDControl(drone_model=drone_model, g=g)
        self._build_mpc_problem()

    def reset(self):
        super().reset()
        self.prev_solution = None

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                       cur_ang_vel, target_pos, target_rpy=np.zeros(3),
                       target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        # Prepare current state and reference
        x0 = np.hstack((cur_pos, cur_vel))
        x_ref = np.hstack((target_pos, target_vel))

        # Check if x0 and x_ref are different
        print("x0:", x0)
        print("x_ref:", x_ref)

        # Solve MPC
        u_opt = self._solve_mpc(x0, x_ref)

        # Generate RPMs from control input
        rpm = self._mpc_to_rpm(u_opt, cur_quat)

        return rpm, x_ref - x0, 0.0  # Replace 0.0 with actual yaw error if needed

    def _mpc_to_rpm(self, u_opt, cur_quat):
        # Include gravity compensation
        desired_acceleration = u_opt + np.array([0, 0, self.g])  # Add gravity
        # Compute desired thrust vector
        desired_thrust_vector = self.mass * desired_acceleration
        # Compute total thrust magnitude
        total_thrust = np.linalg.norm(desired_thrust_vector)
        # Compute desired orientation
        desired_rpy = self._acceleration_to_rpy(desired_thrust_vector, cur_quat)
        # Use inner PID controller
        rpm, _, _ = self.inner_pid.computeControl(
            control_timestep=self.control_timestep,
            cur_pos=np.zeros(3),  # Not used in the inner loop
            cur_quat=cur_quat,
            cur_vel=np.zeros(3),
            cur_ang_vel=np.zeros(3),
            target_pos=np.zeros(3),
            target_rpy=desired_rpy,
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3)
        )
        return rpm

    # def _mpc_to_rpm(self, u_opt, cur_quat):
        
    #     # Convert desired accelerations to desired thrust and orientation
    #     thrust = self.mass * (u_opt + np.array([0, 0, self.g]))
    #     # Compute desired roll, pitch, yaw
    #     desired_rpy = self._acceleration_to_rpy(thrust, cur_quat)
    #     # Use an inner-loop controller (e.g., PID) to compute motor RPMs
    #     rpm, _, _ = self.inner_pid.computeControl(
    #         self.control_timestep,
    #         cur_pos=np.zeros(3),  # Not used in inner loop
    #         cur_quat=cur_quat,
    #         cur_vel=np.zeros(3),
    #         cur_ang_vel=np.zeros(3),
    #         target_pos=np.zeros(3),
    #         target_rpy=desired_rpy,
    #         target_vel=np.zeros(3),
    #         target_rpy_rates=np.zeros(3)
    #     )
    #     return rpm

    def _acceleration_to_rpy(self, desired_thrust_vector, cur_quat):
        # Desired body z-axis (assuming up direction)
        z_b_des = desired_thrust_vector / np.linalg.norm(desired_thrust_vector)
        # Define desired yaw angle (use current yaw for simplicity)
        cur_rpy = Rotation.from_quat(cur_quat).as_euler('xyz')
        desired_yaw = cur_rpy[2]  # Keep current yaw angle
        # Compute x-axis in inertial frame
        x_c_des = np.array([np.cos(desired_yaw), np.sin(desired_yaw), 0])
        y_b_des = np.cross(z_b_des, x_c_des)
        y_b_des /= np.linalg.norm(y_b_des)
        x_b_des = np.cross(y_b_des, z_b_des)
        # Construct desired rotation matrix
        R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        desired_rpy = Rotation.from_matrix(R_des).as_euler('xyz')
        return desired_rpy

    # def _acceleration_to_rpy(self, thrust, cur_quat):
    #     # Compute desired roll and pitch angles to achieve desired accelerations
    #     total_thrust = np.linalg.norm(thrust)
    #     if total_thrust == 0:
    #         return np.zeros(3)
    #     z_b = thrust / total_thrust  # Desired body z-axis
    #     x_c = np.array([np.cos(0), np.sin(0), 0])  # Arbitrary yaw angle, can be improved
    #     y_b = np.cross(z_b, x_c)
    #     y_b /= np.linalg.norm(y_b)
    #     x_b = np.cross(y_b, z_b)
    #     R = np.vstack((x_b, y_b, z_b)).T
    #     desired_rpy = Rotation.from_matrix(R).as_euler('xyz')
    #     return desired_rpy

    def _solve_mpc(self, x0, x_ref):
        # Update parameter values
        self.x0_param.value = x0
        self.x_ref_param.value = x_ref

        # Solve the optimization problem
        try:
            result = self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.u_var.value is not None:
                u_opt = self.u_var[:, 0].value
                # Diagnostic print
                print("u_opt:", u_opt)
                self.prev_solution = u_opt
                return u_opt
            else:
                print("Solver failed, using previous solution or zeros.")
                return self.prev_solution if self.prev_solution is not None else np.zeros(self.nu)
        except cp.error.SolverError as e:
            print("SolverError:", e)
            return self.prev_solution if self.prev_solution is not None else np.zeros(self.nu)

    def _build_mpc_problem(self):
        # Define variables
        N = self.N
        nx = self.nx
        nu = self.nu

        x = cp.Variable((nx, N+1))
        u = cp.Variable((nu, N))

        # Parameters to update at each timestep
        self.x0_param = cp.Parameter(nx)
        self.x_ref_param = cp.Parameter(nx)

        # Discrete-time system matrices
        Ad, Bd = self._get_discrete_system_matrices()

        # Cost function
        cost = 0
        constraints = [x[:, 0] == self.x0_param]
        for k in range(N):
            cost += cp.quad_form(x[:, k+1] - self.x_ref_param, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            # System dynamics constraints
            constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]
            # Input constraints
            constraints += [self.u_min <= u[:, k], u[:, k] <= self.u_max]
        # Terminal state constraint (optional)
        # constraints += [x[:, N] == self.x_ref_param]

        # Define the problem
        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        self.u_var = u

    def _get_discrete_system_matrices(self):
        T = self.control_timestep
        nx = self.nx
        nu = self.nu
        # Continuous-time matrices
        Ac = np.zeros((nx, nx))
        Ac[0:3, 3:6] = np.eye(3)
        Bc = np.zeros((nx, nu))
        Bc[3:6, 0:3] = np.eye(3)
        # Discretize using Zero-Order Hold
        Ad = np.eye(nx) + Ac * T
        Bd = Bc * T
        return Ad, Bd