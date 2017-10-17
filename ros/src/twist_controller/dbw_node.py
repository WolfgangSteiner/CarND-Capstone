#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float64
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller
from lowpass import LowPassFilter
from yaw_controller import YawController

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node', log_level=rospy.INFO)

        self.target_linear_velocity = None
        self.target_angular_velocity = None
        self.current_linear_velocity = None
        self.current_angular_velocity = None
        self.current_acceleration = None
        self.current_cte = None
        self.dbw_enabled = None
        self.accel_tau = 0.5
        self.sample_rate_in_hertz = 50.0 # 50Hz
        self.min_speed = 1.0

        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.controller = Controller(
            sample_rate_in_hertz = self.sample_rate_in_hertz,
            vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35),
            fuel_capacity = rospy.get_param('~fuel_capacity', 13.5),
            brake_deadband = rospy.get_param('~brake_deadband', .1),
            wheel_radius = rospy.get_param('~wheel_radius', 0.2413),
            decel_limit = rospy.get_param('~decel_limit', -5),
            accel_limit = rospy.get_param('~accel_limit', 1.),
            max_steer_angle = self.max_steer_angle)

        self.lpf_accel = LowPassFilter(self.accel_tau, 1.0 / self.sample_rate_in_hertz)
        self.lpf_steer = LowPassFilter(1.0, 1.0)
        self.lpf_steer.set_filter_constant(0.25)
        self.yaw_controller = YawController(
            self.wheel_base,
            self.steer_ratio,
            self.min_speed,
            self.max_lat_accel,
            self.max_steer_angle)

        # Publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # Subscribe to required topics:
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/current_cte', Float64, self.cte_cb)

        rospy.Subscriber('/steering_controller_p', Float64, self.steering_controller_p_cb)
        rospy.Subscriber('/steering_controller_i', Float64, self.steering_controller_i_cb)
        rospy.Subscriber('/steering_controller_d', Float64, self.steering_controller_d_cb)

        rospy.Subscriber('/speed_controller_p', Float64, self.speed_controller_p_cb)
        rospy.Subscriber('/speed_controller_i', Float64, self.speed_controller_i_cb)
        rospy.Subscriber('/speed_controller_d', Float64, self.speed_controller_d_cb)

        self.loop()


    def loop(self):
        rate = rospy.Rate(self.sample_rate_in_hertz) # 50Hz
        while not rospy.is_shutdown():
            if self.current_linear_velocity is None \
            or self.target_linear_velocity is None \
            or self.current_cte is None \
            or not self.dbw_enabled:
                continue

            throttle, brake, steer_a = self.controller.control(
                self.target_linear_velocity,
                self.target_angular_velocity,
                self.current_linear_velocity,
                self.current_angular_velocity,
                self.current_acceleration,
                self.current_cte)

            steer_b = self.yaw_controller.get_steering(
                self.target_linear_velocity,
                self.target_angular_velocity,
                self.current_linear_velocity)

            steer = self.lpf_steer.filter(steer_a + steer_b)
            self.publish(throttle, brake, steer)

            rate.sleep()


    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_PERCENT
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


    def twist_cb(self, msg):
        self.target_linear_velocity = msg.twist.linear.x
        self.target_angular_velocity = msg.twist.angular.z


    def velocity_cb(self, msg):
        if self.current_linear_velocity is not None:
            raw_accel = self.sample_rate_in_hertz * (self.current_linear_velocity - msg.twist.linear.x)
            self.current_acceleration = self.lpf_accel.filter(raw_accel)

        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z


    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data

        if (self.dbw_enabled):
            rospy.loginfo('DBW enabled.')
        else:
            rospy.loginfo('DBW disabled. Resetting Twist Controller.')
            self.controller.reset()


    def cte_cb(self, msg):
        self.current_cte = msg.data


    def steering_controller_p_cb(self, msg):
        rospy.logdebug("Setting P to %.2f", msg.data)
        self.controller.steering_pid.kp = msg.data


    def steering_controller_i_cb(self, msg):
        rospy.logdebug("Setting I to %.2f", msg.data)
        self.controller.steering_pid.ki = msg.data


    def steering_controller_d_cb(self, msg):
        rospy.logdebug("Setting D to %.2f", msg.data)
        self.controller.steering_pid.kd = msg.data


    def speed_controller_p_cb(self, msg):
        rospy.logdebug("Setting P to %.2f", msg.data)
        self.controller.speed_pid.kp = msg.data


    def speed_controller_i_cb(self, msg):
        rospy.logdebug("Setting I to %.2f", msg.data)
        self.controller.speed_pid.ki = msg.data


    def speed_controller_d_cb(self, msg):
        rospy.logdebug("Setting D to %.2f", msg.data)
        self.controller.speed_pid.kd = msg.data



if __name__ == '__main__':
    DBWNode()
