import numpy as np
import math
from scipy.spatial.transform import Rotation
from control.BaseControl import BaseControl  # Assuming you have a BaseControl class

class GeometricControl(BaseControl):
    def __init__(self, drone_model, g=9.81, m=0.027):
        super().__init__(drone_model=drone_model, g=g)
        self.mass = m
        self.g = g
        # Control gains
        self.kx = np.array([4.0, 4.0, 4.0])       # Position gain
        self.kv = np.array([2.0, 2.0, 2.0])       # Velocity gain
        self.kR = np.array([2.0, 2.0, 2.0])       # Attitude gain
        self.kOmega = np.array([0.1, 0.1, 0.1])   # Angular velocity gain
        # Inertia matrix
        self.J = np.diag([1.43e-5, 1.43e-5, 2.89e-5])  # Adjust for your drone

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                       cur_ang_vel, target_pos, target_rpy=np.zeros(3),
                       target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        # Position and velocity errors
        e_pos = cur_pos - target_pos
        e_vel = cur_vel - target_vel

        # Desired force in inertial frame
        F_des = -self.kx * e_pos - self.kv * e_vel + self.mass * self.g * np.array([0, 0, 1]) - self.mass * np.array([0, 0, 0])

        # Desired acceleration
        a_des = F_des / self.mass

        # Compute desired orientation
        z_b_des = a_des / np.linalg.norm(a_des)
        x_c = np.array([np.cos(target_rpy[2]), np.sin(target_rpy[2]), 0])
        y_b_des = np.cross(z_b_des, x_c)
        y_b_des /= np.linalg.norm(y_b_des)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.vstack((x_b_des, y_b_des, z_b_des)).T

        # Current orientation
        R = Rotation.from_quat(cur_quat).as_matrix()

        # Attitude error
        e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])

        # Angular velocity error
        Omega = cur_ang_vel
        Omega_des = np.zeros(3)
        e_Omega = Omega - R.T @ R_des @ Omega_des

        # Control torques
        M = -self.kR * e_R - self.kOmega * e_Omega + np.cross(Omega, self.J @ Omega)

        # Compute total thrust
        f_total = np.dot(F_des, R[:, 2])

        # Motor mixing to compute RPMs
        forces = np.array([f_total])
        torques = M
        pwm = self._motor_mixing(forces, torques)
        rpm = self._pwm_to_rpm(pwm)

        return rpm, e_pos, e_R

    # def _motor_mixing(self, forces, torques):
    #     # Assuming a quadcopter with plus configuration
    #     l = 0.046  # Arm length in meters (adjust to your drone)
    #     kf = 3.16e-10  # Thrust coefficient (adjust to your drone)
    #     km = 7.94e-12  # Moment coefficient (adjust to your drone)
    #     mix_matrix = np.array([
    #         [1 / (4 * kf), 1 / (2 * kf * l), 1 / (4 * km)],
    #         [1 / (4 * kf), -1 / (2 * kf * l), 1 / (4 * km)],
    #         [1 / (4 * kf), -1 / (2 * kf * l), -1 / (4 * km)],
    #         [1 / (4 * kf), 1 / (2 * kf * l), -1 / (4 * km)]
    #     ])
    #     u = np.hstack((forces, torques[0:2], torques[2]))
    #     pwm = mix_matrix @ u
    #     return pwm

    def _motor_mixing(self, forces, torques):
        # Assuming 'forces' is a scalar total thrust
        f_total = forces  # Should be a scalar
        tau_phi, tau_theta, tau_psi = torques  # Torques around x, y, z axes
        print ("f_total: ", f_total, " tau_phi: ", tau_phi, " tau_theta: ", tau_theta, " tau_psi: ", tau_psi)
        # Build u vector
        u = np.array([f_total[0], tau_phi, tau_theta, tau_psi])
        # Mix matrix (rows: motors, columns: [thrust, tau_phi, tau_theta, tau_psi])
        mix_matrix = np.array([
            [1, 1, -1, -1],
            [1, -1, -1, 1],
            [1, -1, 1, -1],
            [1, 1, 1, 1]
        ])
        # Motor commands
        motor_commands = mix_matrix @ u
        # Scale motor commands
        pwm = motor_commands / 4  # Adjust based on your scaling
        return pwm

    def _pwm_to_rpm(self, pwm):
        # Convert PWM signals to RPMs
        # You need to adjust the conversion based on your drone's motor characteristics
        # Assuming linear mapping for simplicity
        PWM_MIN = 0
        PWM_MAX = 65535
        RPM_MIN = 0
        RPM_MAX = 25000  # Adjust based on maximum motor speed
        pwm = np.clip(pwm, PWM_MIN, PWM_MAX)
        rpm = RPM_MIN + (pwm - PWM_MIN) * (RPM_MAX - RPM_MIN) / (PWM_MAX - PWM_MIN)
        rpm = np.sqrt(rpm)  # Assuming rpm^2 proportional to thrust
        return rpm