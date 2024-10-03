import numpy as np
from scipy.optimize import minimize
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel
import pybullet as p
from scipy.spatial.transform import Rotation

class MPCControl(BaseControl):
    """MPC control class for Crazyflies."""

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLMPCControl.__init__(), DSLMPCControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        
        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.dt = 0.1  # Time step for prediction

        # Weights for the cost function
        self.Q = np.diag([10, 10, 10, 1, 1, 1])  # State cost
        self.R = np.diag([0.1, 0.1, 0.1, 0.1])  # Input cost

        # Constraints
        self.max_thrust = 0.6 * self.GRAVITY  # Maximum thrust per motor
        self.max_angle = np.pi/6  # Maximum roll and pitch angle
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3

        # Mixer matrix
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                                    [1, 1, 1, 1],
                                    [0, 1, 0, -1],
                                    [-1, 0, 1, 0],
                                    [-1, 1, -1, 1]
                                ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [1, 1, 1, 1],
                                    [0, 1, 0, -1],
                                    [-1, 0, 1, 0],
                                    [-1, 1, -1, 1]
                                ])

        self.reset()

    def reset(self):
        """Resets the control classes."""
        super().reset()
        self.last_rpy = np.zeros(3)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the MPC control action (as RPMs) for a single drone."""
        self.control_counter += 1

        # Current state
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        x0 = np.concatenate([cur_pos, cur_rpy])

        # Desired state
        xd = np.concatenate([target_pos, target_rpy])

        # Solve MPC problem
        u_optimal = self._solve_mpc(x0, xd)

        # Convert optimal control to RPM
        rpm = self._control_to_rpm(u_optimal[0])

        pos_e = target_pos - cur_pos
        return rpm, pos_e, target_rpy[2] - cur_rpy[2]
    
    def _solve_mpc(self, x0, xd):
        """Solves the MPC optimization problem."""
        def objective(u_flat):
            u = u_flat.reshape(self.horizon, 4)
            x = x0
            cost = 0
            for k in range(self.horizon):
                cost += np.dot((x - xd).T, np.dot(self.Q, (x - xd)))
                cost += np.dot(u[k].T, np.dot(self.R, u[k]))
                x = self._dynamics(x, u[k])
            return cost

        def constraint(u_flat):
            u = u_flat.reshape(self.horizon, 4)
            return np.concatenate([self.max_thrust - u.flatten(), u.flatten()])

        u0 = np.zeros((self.horizon, 4))
        bounds = [(-self.max_thrust, self.max_thrust) for _ in range(4*self.horizon)]
        cons = {'type': 'ineq', 'fun': constraint}

        result = minimize(objective, u0.flatten(), method='SLSQP', bounds=bounds, constraints=cons)
        return result.x.reshape(self.horizon, 4)

    def _dynamics(self, x, u):
        """Simple dynamics model for prediction."""
        pos = x[:3]
        rpy = x[3:]
        
        # Simple dynamics (constant acceleration model)
        pos_next = pos + self.dt * np.array([0, 0, u[0] - self.GRAVITY])
        rpy_next = rpy + self.dt * u[1:]

        return np.concatenate([pos_next, rpy_next])
    
    def _control_to_rpm(self, u):
        """Converts MPC control output to motor RPMs."""
        thrust = u[0]
        torques = u[1:]

        # print ("self.MIXER_MATRIX: ", self.MIXER_MATRIX.shape)
        # print ("np.concatenate([[thrust], torques]): ", np.concatenate([[thrust], torques]).shape)
    
        # print("thrust:", thrust)
        # print("torques:", torques)
                
        motor_forces = np.dot(np.linalg.inv(self.MIXER_MATRIX), np.concatenate([[thrust], torques]))
        pwm = np.clip(np.sqrt(motor_forces / self.KF), self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return rpm

    # def _solve_mpc(self, x0, xd):
    #     """Solves the MPC optimization problem."""
    #     def objective(u):
    #         x = x0
    #         cost = 0
    #         for k in range(self.horizon):
    #             cost += np.dot((x - xd).T, np.dot(self.Q, (x - xd)))
    #             cost += np.dot(u[k].T, np.dot(self.R, u[k]))
    #             x = self._dynamics(x, u[k])
    #         return cost

    #     def constraint(u):
    #         return np.concatenate([self.max_thrust - u, u])

    #     u0 = np.zeros((self.horizon, 4))
    #     bounds = [(-self.max_thrust, self.max_thrust) for _ in range(4*self.horizon)]
    #     cons = {'type': 'ineq', 'fun': constraint}

    #     result = minimize(objective, u0.flatten(), method='SLSQP', bounds=bounds, constraints=cons)
    #     return result.x.reshape(self.horizon, 4)

    # def _dynamics(self, x, u):
    #     """Simple dynamics model for prediction."""
    #     pos = x[:3]
    #     rpy = x[3:]
        
    #     print ("u: ", u)
    #     # Simple dynamics (constant acceleration model)
    #     pos_next = pos + self.dt * np.array([0, 0, u[0] - self.GRAVITY])
    #     rpy_next = rpy + self.dt * np.array([u[1], u[2], u[3]])

    #     return np.concatenate([pos_next, rpy_next])

    