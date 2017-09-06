import rospy
from pid import PID
from math import fabs

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# Minimal driving speed in mph.
MIN_SPEED_MPH = 5

def mphToMps(mph):
    return mph * ONE_MPH

# Original Controller
class Controller1(object):
    def __init__(self, *args, **kwargs):
        self.linear_velocity_pid = PID(1.0, 0.1, 0.5, mn=0.0, mx=1.0)
        self.angluar_velocity_pid = PID(10.0, 0.1, 0.5, mn=-0.43, mx=0.43)

    def control(self,
        target_linear_velocity, target_angular_velocity,
        current_linear_velocity, current_angular_velocity,
        lpf_accel,
        dbw_enabled, **kwargs):
        sample_time = 1 / 50.0

        vel_error = target_linear_velocity - current_linear_velocity

        throttle = self.linear_velocity_pid.step(vel_error, sample_time)
        steer = self.angluar_velocity_pid.step(target_angular_velocity, sample_time)

        return throttle, 0.0, steer

# New version with brake
class Controller(object):
    def __init__(self, *args, **kwargs):

        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        
        self.cfg_brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.cfg_wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.cfg_vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.cfg_fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)

#        self.linear_velocity_pid = PID(2.0, 0.0, 0.0, mn=-fabs(decel_limit), mx=accel_limit)
        self.linear_velocity_pid = PID(1.0, 0.1, 0.5, mn=-fabs(decel_limit), mx=accel_limit)
        self.accel_pid = PID(0.4, 0.1, 0.0, mn=0.0, mx=1.0)
        self.angluar_velocity_pid = PID(10.0, 0.1, 0.5, mn=-0.43, mx=0.43)


    def control(self,
        target_linear_velocity, target_angular_velocity,
        current_linear_velocity, current_angular_velocity,
        lpf_accel,
        dbw_enabled, **kwargs):
        sample_time = 1 / 50.0

        # we don't know fuel level, so we use 100% of fuel_capacity.
        vehicle_mass = self.cfg_vehicle_mass + self.cfg_fuel_capacity * GAS_DENSITY
        
        vel_error = target_linear_velocity - current_linear_velocity
        accel_cmd = self.linear_velocity_pid.step(vel_error, sample_time)

        steer = self.angluar_velocity_pid.step(target_angular_velocity, sample_time)

        throttle = None
        brake_cmd = None

        MIN_SPEED = mphToMps(MIN_SPEED_MPH)
        if (target_linear_velocity <= 1e-2):
            accel_cmd = min(accel_cmd, -530.0 / vehicle_mass / self.cfg_wheel_radius)
        elif (target_linear_velocity < MIN_SPEED):
            # possible to increase angular velocity here for low speed!
            target_linear_velocity = MIN_SPEED # comment this line to remove MIN_SPEED

        if (accel_cmd >= 0.0):
            throttle = self.accel_pid.step(accel_cmd - lpf_accel.get(), sample_time)
        else:
            self.accel_pid.reset()
            throttle = 0.0

        if (accel_cmd < -fabs(self.cfg_brake_deadband)) or (target_linear_velocity < MIN_SPEED):
            brake_cmd = -accel_cmd * vehicle_mass * self.cfg_wheel_radius
        else:
            brake_cmd = 0.0

        return throttle, brake_cmd, steer

