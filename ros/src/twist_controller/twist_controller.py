from pid import PID
from math import fabs

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# Minimal driving speed in m/s.
MIN_SPEED_MPS = 2 # or 4.47 mph = 7.2 km/h

class Controller(object):
    def __init__(self,
        sample_rate_in_hertz,
        vehicle_mass,
        fuel_capacity,
        brake_deadband,
        wheel_radius,
        decel_limit,
        accel_limit, **kwargs):

        self.sample_rate_in_hertz = sample_rate_in_hertz
        self.cfg_brake_deadband = brake_deadband
        self.cfg_wheel_radius = wheel_radius
        self.cfg_vehicle_mass = vehicle_mass
        self.cfg_fuel_capacity = fuel_capacity

#        self.linear_velocity_pid = PID(2.0, 0.0, 0.0, mn=-fabs(decel_limit), mx=accel_limit)
        self.linear_velocity_pid = PID(1.0, 0.1, 0.5, mn=-fabs(decel_limit), mx=accel_limit)
        self.accel_pid = PID(0.4, 0.1, 0.0, mn=0.0, mx=1.0)
        self.angluar_velocity_pid = PID(10.0, 0.1, 0.5, mn=-0.43, mx=0.43)


    def control(self,
        target_linear_velocity, target_angular_velocity,
        current_linear_velocity, current_angular_velocity,
        current_accel,
        dbw_enabled, **kwargs):
        
        sample_time = 1.0 / self.sample_rate_in_hertz

        # we don't know fuel level, so we use 100% of fuel_capacity.
        vehicle_mass = self.cfg_vehicle_mass + self.cfg_fuel_capacity * GAS_DENSITY
        
        vel_error = target_linear_velocity - current_linear_velocity
        accel_cmd = self.linear_velocity_pid.step(vel_error, sample_time)

        steer = self.angluar_velocity_pid.step(target_angular_velocity, sample_time)

        throttle = None
        brake_cmd = None

        MIN_SPEED = MIN_SPEED_MPS
        if (target_linear_velocity <= 1e-2):
            accel_cmd = min(accel_cmd, -530.0 / vehicle_mass / self.cfg_wheel_radius)
        elif (target_linear_velocity < MIN_SPEED):
            # possible to increase angular velocity here for low speed!
            target_linear_velocity = MIN_SPEED # comment this line to remove MIN_SPEED

        if (accel_cmd >= 0.0):
            throttle = self.accel_pid.step(accel_cmd - current_accel, sample_time)
        else:
            self.accel_pid.reset()
            throttle = 0.0

        if (accel_cmd < -fabs(self.cfg_brake_deadband)) or (target_linear_velocity < MIN_SPEED):
            brake_cmd = -accel_cmd * vehicle_mass * self.cfg_wheel_radius
        else:
            brake_cmd = 0.0

        return throttle, brake_cmd, steer

