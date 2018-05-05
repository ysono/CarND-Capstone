#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100         # Number of waypoints we will publish. You can change this number
MAX_DECEL = 6.0             # Maximum deceleration allowed during braking
COMF_DECEL = 2              # A comfortable rate of deceleration
STOP_LINE_OFFSET = 1.5      # Distance before the stop line to actually stop the car
STOP_INDEX_OFFSET = 2       # Number of Waypoints indices within which to stop before stop line
MIN_VELOCITY = 3.0          # Minimum velocity over which a late red light will be ignored

DRIVE_STATE_INIT = 0
DRIVE_STATE_DRIVING = 1     # Driving states
DRIVE_STATE_STOPPING = 2
REF_VELOCITY = 10.0          # Reference velocity at which to drive the car normally

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.traffic_wp_idx = None
        self.driving_state = DRIVE_STATE_INIT
        self.current_velocity = 0.0
        self.closest_wp_idx = -1
        self.stop_idx = -1

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.current_velocity = rospy.Subscriber('/current_velocity', TwistStamped,
                                                 self.current_velocity_cb)

        # TODO: Add a subscriber for /obstacle_waypoint

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints and self.waypoints_2d and self.waypoint_tree:
                # Get closest waypoint
                self.closest_wp_idx = self.get_closest_waypoint_idx()
                self.drive_state_machine()
                self.publish_waypoints()

            rate.sleep()

    def drive_state_machine(self):
        farthest_wp_idx = self.closest_wp_idx + LOOKAHEAD_WPS
        if self.closest_wp_idx > 0:
            self.stop_idx = self.get_stop_idx(self.closest_wp_idx)
            if self.stop_idx > 0:
                # red light ahead, near or far
                distance_to_stop_line = self.distances_to_end(self.waypoints[self.closest_wp_idx : self.stop_idx])
                comfort_stopping_distance = (self.current_velocity * self.current_velocity)
                comfort_stopping_distance = comfort_stopping_distance / COMF_DECEL

                minimum_stop_distance = self.current_velocity * self.current_velocity
                minimum_stop_distance = minimum_stop_distance / MAX_DECEL
                print(distance_to_stop_line[0])
                if (distance_to_stop_line[0] - STOP_LINE_OFFSET) < comfort_stopping_distance:
                    if self.driving_state == DRIVE_STATE_DRIVING and \
                            (distance_to_stop_line[0] - STOP_LINE_OFFSET) < minimum_stop_distance:
                        # keep going, stopping distance not enough for comfortable stop
                        if self.current_velocity < MIN_VELOCITY and \
                                distance_to_stop_line[0] > STOP_LINE_OFFSET:
                            # Possible to stop
                            rospy.loginfo(" Emergency stop ")
                            self.driving_state = DRIVE_STATE_STOPPING
                        else:
                            rospy.loginfo("Ignoring late red light")
                            self.driving_state = DRIVE_STATE_DRIVING
                    else:
                        # should slow down and stop now
                        if self.driving_state != DRIVE_STATE_STOPPING:
                            rospy.loginfo("Changing to STOPPING state")
                        self.driving_state = DRIVE_STATE_STOPPING
                else:
                    if self.driving_state == DRIVE_STATE_STOPPING:
                        # we are already stopping, stay in this state if the light is still red
                        rospy.loginfo("Holding STOPPING state")
                    else:
                        if self.driving_state != DRIVE_STATE_DRIVING:
                            rospy.loginfo("Changing to DRIVING state")
                        self.driving_state = DRIVE_STATE_DRIVING
            else:
                # no red light
                if self.driving_state != DRIVE_STATE_DRIVING:
                   rospy.loginfo("Changing to DRIVING state")
                self.driving_state = DRIVE_STATE_DRIVING

    def publish_waypoints(self):
        closest_wp_idx = self.closest_wp_idx
        start_point_velocity = self.get_waypoint_velocity(self.waypoints[closest_wp_idx])
        # get the distance from each waypoint till the stop line
        distance_to_stop_line = self.distances_to_end(self.waypoints[closest_wp_idx : self.stop_idx])

        if self.driving_state == DRIVE_STATE_STOPPING and closest_wp_idx < self.stop_idx:
            # Loop over the waypoints up to next_red_light waypoint
            # setting desired velocity at each waypoint as a ratio of remaining distance to stop
            # from the respective waypoint to the total distance from current position
            # to stop line
            # print(closest_wp_idx, self.stop_idx)
            for i in range(0, (abs(closest_wp_idx - self.stop_idx))):
                # get the distance to the i-th way point
                # i_point_distance = self.distances_to_end(self.waypoints, self.closest_waypoint, i)
                if (distance_to_stop_line[0]) > STOP_LINE_OFFSET:
                    if self.current_velocity < 0.8:
                        i_point_target_velocity = -10.0
                    else:
                        i_point_target_velocity = distance_to_stop_line[i]/distance_to_stop_line[0]
                        i_point_target_velocity = (start_point_velocity * i_point_target_velocity)

                else:
                    i_point_target_velocity = -10.0     # negative stops car 'creep' when stopped
                self.set_waypoint_velocity(self.waypoints, closest_wp_idx + i, i_point_target_velocity)
        else:
            # Set the velocity to reference velocity if driving state is not stopping
            for i in range(closest_wp_idx, closest_wp_idx + LOOKAHEAD_WPS):
                if i < len(self.waypoints):
                    self.set_waypoint_velocity(self.waypoints, i, REF_VELOCITY)

        # now publish the waypoints
        # get LOOKAHEAD_WPS number of waypoints ahead of the car
        waypoints_ahead = []
        n_waypoints = len(self.waypoints)  # can only get this many waypoints
        if n_waypoints > LOOKAHEAD_WPS:
            n_waypoints = LOOKAHEAD_WPS  # max waypoints to pass over
        for i in range(n_waypoints):
            # check that the waypoints we want are in the range of the waypoint array
            if closest_wp_idx + i < len(self.waypoints):
                waypoints_ahead.append(self.waypoints[closest_wp_idx + i])

        # structure the data to match the expected styx_msgs/Lane form
        lane = Lane()
        lane.waypoints = waypoints_ahead  # list of waypoints ahead of the car
        lane.header.stamp = rospy.Time(0)  # timestamp

        self.final_waypoints_pub.publish(lane)

    def get_stop_idx(self, closest_wp_idx):
        farthest_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
        if self.traffic_wp_idx is None or \
        (self.traffic_wp_idx < 0) or \
        (self.traffic_wp_idx >= farthest_wp_idx):

            stop_idx = -1
        else:
            # Stop 2 waypoints before closest waypoint to the traffic light
            stop_idx = max(self.traffic_wp_idx - STOP_INDEX_OFFSET, 0)

        return stop_idx

    # This function is not used anymore
    def decelerate_waypoints(self, waypoints, stop_idx):
        waypoints_up_to_stop = waypoints[:stop_idx]
        dists_to_stop = self.distances_to_end(waypoints_up_to_stop)

        result = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            if i >= stop_idx:
                vel = 0
            else:
                dist_to_stop = dists_to_stop[i]
                
                # Linear decrease in velocity wrt time. We could smoothen this.
                vel = math.sqrt(2 * MAX_DECEL * dist_to_stop)

                # obey speed limit
                vel = min(vel, wp.twist.twist.linear.x)
            p.twist.twist.linear.x = vel
            result.append(p)

        return result

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            # closest_coord is behind us
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [
                [waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                for waypoint in waypoints.waypoints
            ]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        if self.traffic_wp_idx != msg.data:
            rospy.loginfo('waypoint_updater received new traffic wp {}'.format(msg.data))
        self.traffic_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def distances_to_end(self, waypoints):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

        len_wps = len(waypoints)

        dists_reversed = [0]

        for i in range(len_wps - 1):
            wp_idx = len_wps - 2 - i

            incremental_dist = dl(
                waypoints[wp_idx].pose.pose.position,
                waypoints[wp_idx + 1].pose.pose.position)
            # Accumulate previous distance upto this waypoint and add the current distance
            total_dist = dists_reversed[-1] + incremental_dist

            dists_reversed.append(total_dist)
        # reverse the list so that it contains waypoints distance starting from closest waypoint
        return dists_reversed[::-1]


    def get_waypoint_velocity(self, waypoint):

        # Get velocity for a given waypoint object
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):

        # Set velocity for a given waypoint in the list of waypoints
        waypoints[waypoint].twist.twist.linear.x = velocity

    def current_velocity_cb(self, msg):
        # store the current velocity TwistStamped message
        self.current_velocity = msg.twist.linear.x
if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
