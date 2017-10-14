#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import tfrunner
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        # print("Running current pose")
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        # print("Running image cb")
        light_wp, state = self.process_traffic_lights()
        #print(light_wp, state)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0


    def get_light_state(self):
        """Determines the current color of the closeset traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        loc = tfrunner.run(cv_image)
        threshold = 0.99
        r = np.sum(loc[0,0,:,:,1]>threshold)
        g = np.sum(loc[0,0,:,:,2]>threshold)
        y = np.sum(loc[0,0,:,:,3]>threshold)

        if(r+g+y)>5:
            if(r>=g and r>=y):
                print("red detect")
                return TrafficLight.RED
            elif(g>y):
                print("green detect")
                return TrafficLight.GREEN
            else:
                print("yellow detect")
                return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if self.waypoints is None:
            return -1, TrafficLight.UNKNOWN


        stop_line_position = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        orientation = self.pose.pose.orientation
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _,_,phi = tf.transformations.euler_from_quaternion(q)

        x0 = self.pose.pose.position.x
        y0 = self.pose.pose.position.y


        s = np.sin(phi)
        c = np.cos(phi)
        rot = np.array([[c,s],[-s,c]])
        local = [rot.dot(np.array([x-x0, y-y0])) for x,y in stop_line_positions]

        ## search closest avoinding backside lights
        m = 9e99
        mi = None
        for i,c in enumerate(local):
            if(c[0]>-15): # 0 is the stopping line but you can see it a bit later
                d = c[0]**2 + c[1]**2
                if(d<m):
                    m = d
                    mi = i
        #print("Looking at tl %d"% mi)

        #TODO find the closest visible traffic light (if one exists)
        if mi is not None:
            stop_line_position = stop_line_positions[mi]
            min_dist = 1e9
            min_idx = None

            for idx,wp in enumerate(self.waypoints.waypoints):
                wx, wy = wp.pose.pose.position.x, wp.pose.pose.position.y
                dx = wx - stop_line_position[0]
                dy = wy - stop_line_position[1]
                dist =  np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx

            return min_idx, self.get_light_state()

        else:
            return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
