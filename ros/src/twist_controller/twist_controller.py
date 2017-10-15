from pid import PID
from math import fabs

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# Minimal driving speed in m/s.
MIN_SPEED_MPS = 2 # or 4.47 mph = 7.2 km/h

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
        self.accel_pid = PID(0.4, 0.1, 0.0, mn=0.0, mx=1.0)
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

        # we don't know fuel level, so we use 100% of fuel_capacity.
        vehicle_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY

        vel_error = target_linear_velocity - current_linear_velocity
        accel_cmd = self.linear_velocity_pid.step(vel_error, sample_time)

        steer = self.steering_pid.step(current_cte, sample_time)

        if (target_linear_velocity <= 1e-2):
            accel_cmd = min(accel_cmd, -530.0 / vehicle_mass / self.wheel_radius)
        elif (target_linear_velocity < MIN_SPEED_MPS):
            # possible to increase linear velocity here for low speed!
            target_linear_velocity = MIN_SPEED_MPS # comment this line to remove MIN_SPEED_MPS

        if (accel_cmd >= 0.0):
            throttle = self.accel_pid.step(accel_cmd - current_accel, sample_time)
        else:
            self.accel_pid.reset()
            throttle = 0.0

        if (accel_cmd < -fabs(self.brake_deadband)) or (target_linear_velocity < MIN_SPEED_MPS):
            brake_cmd = -accel_cmd * vehicle_mass * self.wheel_radius
        else:
            brake_cmd = 0.0

        return throttle, brake_cmd, steer


    def reset(self):
        self.linear_velocity_pid.reset()
        self.steering_pid.reset()
