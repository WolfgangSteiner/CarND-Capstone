from pid import PID
from math import fabs


class Controller(object):
    def __init__(self, **kwargs):
        self.sample_rate_in_hertz = kwargs["sample_rate_in_hertz"]
        self.brake_deadband = kwargs["brake_deadband"]
        self.wheel_radius = kwargs["wheel_radius"]
        self.vehicle_mass = kwargs["vehicle_mass"]
        self.fuel_capacity = kwargs["fuel_capacity"]
        self.decel_limit = kwargs["decel_limit"]
        self.accel_limit = kwargs["accel_limit"]
        self.max_steer_angle = kwargs["max_steer_angle"]
        self.linear_velocity_pid = PID(1.0, 0.1, 0.5, mn=-fabs(self.decel_limit), mx=self.accel_limit)
        self.speed_pid = PID(1.0, 0.0, 0.04,  mn=-fabs(self.decel_limit), mx=self.accel_limit)
        self.steering_pid_parameters = kwargs.get("steering_pid_parameters", [0.1, 0.01, 32.0])
        steer_p, steer_i, steer_d = self.steering_pid_parameters
        max_steer = 1.0
        self.steering_pid = PID(
            steer_p, steer_i, steer_d,
            mn=-max_steer, mx=max_steer)


    def control(self,
        target_linear_velocity, target_angular_velocity,
        current_linear_velocity, current_angular_velocity,
        current_accel, current_cte):

        sample_time = 1.0 / self.sample_rate_in_hertz

        vel_error = target_linear_velocity - current_linear_velocity
        speed_cmd = self.speed_pid.step(vel_error, sample_time)

        steer = self.steering_pid.step(current_cte, sample_time)
        brake_cmd = 0.0
        throttle_cmd = 0.0


        if speed_cmd < -self.brake_deadband:
            brake_cmd = (abs(speed_cmd) - self.brake_deadband) / (1.0 - self.brake_deadband)
    
        elif speed_cmd < 0.0:
            pass      

        else:
            throttle_cmd = speed_cmd

        return throttle_cmd, brake_cmd, steer


    def reset(self):
        self.linear_velocity_pid.reset()
        self.steering_pid.reset()
