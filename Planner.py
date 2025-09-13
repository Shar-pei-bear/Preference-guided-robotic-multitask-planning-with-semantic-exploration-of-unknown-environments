#!/usr/bin/env python2
import numpy
import numpy as np
from numpy.linalg import inv, norm, eig, det
from scipy.interpolate import interp1d
import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Bool
from darknet_ros_msgs.msg import Object2D, Object2DArray, RobotCommand, GripState
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_ros import TransformException, ConnectivityException, \
    ExtrapolationException
from cv_bridge import CvBridge
import cv2
import tf
from scipy.special import softmax
import math
from time import time
from threading import Thread, Lock
from datetime import datetime
import actionlib
from ipa_building_msgs.msg import MapSegmentationAction, MapSegmentationGoal
from darknet_ros_msgs.srv import *
from DFA import DFA
from copy import deepcopy

from upper_bound import upper_bound_efficient
import pickle

def color_converter(value):
    if value == -1:
        return 128
    elif value == 0:
        return 255
    else:
        return 0


class MapObject:
    def __init__(self, object_id, pos, pos_var, class_probs):
        self.id = object_id
        self.pos = pos
        self.pos_var = pos_var
        self.class_probs = class_probs
        self.visualized = False
        # self.grip_pub = rospy.Publisher("/grip_image_" + str(object_id), Image, queue_size=10)

    def update(self, pos, pos_var, class_probs):
        self.pos = pos
        self.pos_var = pos_var
        self.class_probs = class_probs

class Planner:
    def __init__(self):
        self.target_frame = "camera_link"
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.ang_var = None
        self.pos_var = None

        self.old_ang = None
        self.ang = None
        self.pos = None
        self.vel = None

        self.sigma_p = None
        self.sigma_p_inv = None

        self.seg_map = None
        self.old_seg_map = None

        self.grid_map_grey = None
        self.old_grid_map = None
        self.old_obstacle_map = None

        self.old_sighting_map = None
        self.old_sighting_phase = None
        self.old_sighting_prob = None

        self.old_frontier_map = None
        self.old_frontier_prob = None
        self.old_frontier_phase = None

        self.grid_map_color = None
        self.obstacle_map = None
        self.obstacle_prob = None

        self.goal_map = None
        self.goal_phase = None
        self.goal_prob = None

        self.grip_probs = {}
        self.grip_maps = {}

        self.old_grid_maps = None
        self.old_grip_probs = None

        self.old_goal_map = None
        self.old_goal_prob = None
        self.old_goal_phase = None

        self.frontier_phase = None
        self.frontier_map = None
        self.frontier_prob = None
        self.frontier_centroid = None
        self.frontier_num_labels = None
        self.frontier_stat = None

        self.width = None
        self.height = None

        self.old_height = None
        self.old_width = None

        self.resolution = None
        self.old_resolution = None

        self.offset_x = None
        self.offset_y = None

        self.old_offset_x = None
        self.old_offset_y = None

        self.objects = {}
        self.sightings_map = {}
        self.sightings_phase = {}
        self.sightings_prob = {}
        self.room_ids_2_objects = {}
        self.room_ids_2_prob = {}

        self.bridge = CvBridge()

        self.seq = 1

        self.goal = None
        self.goal_status = False
        self.mission_status = False
        self.planner_type = None
        self.ang_goal1_status = False
        self.ang_goal2_status = False
        self.ang_calculation = True

        self.ang_goal1 = None
        self.ang_goal2 = None

        self.initial_obs = True
        self.replan = True
        self.observed_ang = 0

        self.blind_radius = 1.0

        self.goal_radius = 1.1
        self.grip_radius = 0.6 # 0.76

        self.action_deltas = [[1, -1],
                              [1, 0],
                              [1, 1],
                              [0, 1],
                              [-1, 1],
                              [-1, 0],
                              [-1, -1],
                              [0, -1]]
        self.nA = 8

        self.action_probabilities = (0.8, 0.1, 0.1)

        self.V = {}
        self.gamma = 0.99
        self.P = {}
        self.states_index = None
        self.policy = None

        self.classes = ['towel', 'objects', 'lighting', 'stool', 'counter',
                        'door', 'clothes', 'appliances', 'furniture',
                        'shelving', 'bed', 'blinds', 'table', 'cabinet',
                        'shower', 'chair', 'chest_of_drawers', 'tv_monitor',
                        'toilet', 'mirror', 'sofa', 'cushion', 'sink',
                        'car', 'stegosaurus', 'train', 'parasaurolophus','tank']
        self.target_classes = ['car', 'train', 'parasaurolophus', 'stegosaurus', 'tank']

        self.objs_of_interest = {}
        self.old_objs_of_interest = {}

        self.max_class_probs = {}

        self.table_of_interest = None
        self.max_table_prob = 0

        self.agent_radius = 0.16 #0.15
        self.angular_error_threshold = 0.5
        self.max_linear_speed = 0.2
        self.max_turn_speed = 0.4
        self.planning_time = 0
        self.start_time = 0
        self.end_time = 0

        self.dfa = []
        self.construct_dfa()
        self.dfa_state = self.get_dfa_state()
        self.grip_state = 0

        self.task_1_reward = 1.0
        self.task_2_reward = 1.0
        self.task_3_reward = 0.8

        self.gains = np.asarray(
            [0.0, 0.0, 0.8, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0,
             0.0, 0.8, 0.0, 0.8, 1.8, 1.8, 1.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8,
             1.8,
             1.8, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8,
             0.0, 0.0, 0.8, 0.0, 0.8, 1.8, 1.8, 1.8, 0.0, 0.8, 0.0, 0.8, 0.0,
             0.8,
             1.8, 1.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 1.8, 1.8, 0.0, 0.0, 0.8,
             0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0,
             0.8,
             1.8, 1.8, 1.8, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 1.8, 1.8, 1.8, 1.8,
             1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8,
             1.8,
             1.8, 2.8, 2.8, 2.8])

        self.dfa_states = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 33, 35, 45, 47, 51, 53, 54, 55, 56, 57, 59, 60,
                          61, 62, 63, 65, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 87, 89, 99, 101, 105, 107, 135, 137, 141, 143, 153, 155, 159,
                          161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 173, 177, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 191, 195, 197, 207,
                          209, 213, 215, 216, 217, 218, 219, 221, 222, 223, 224, 225, 227, 231, 233, 234, 235, 236, 237, 239, 240, 241, 242]

        self.dfa_states_indices = {}
        for i in range(len(self.dfa_states)):
            self.dfa_states_indices[self.dfa_states[i]] = i


        self.explore_reward = 0.02
        self.reset = False

        self.old_task_prob = 0
        self.prev_task_prob = 0
        self.log_1 = []
        self.log_2 = []
        self.log_3 = []
        self.log_4 = []
        self.log_5 = []
        self.log_6 = []
        self.log_7 = []

        self.mutex = Lock()
        self.value_mutex = Lock()

        self.map_client = actionlib.SimpleActionClient(
            '/room_segmentation/room_segmentation_server',
            MapSegmentationAction)
        rospy.loginfo("Waiting for action server to start.")
        # wait for the action server to start
        self.map_client.wait_for_server()  # will wait for infinite time
        rospy.loginfo("Action server started, sending goal.")

        self.odom_sub = rospy.Subscriber("/rtabmap/odom", Odometry,
                                         self.odom_callback)
        self.grip_sub = rospy.Subscriber("/gripper", GripState, self.grip_callback)
        self.semantic_map_sub = rospy.Subscriber("/semantic_map",
                                                 Object2DArray,
                                                 self.semantic_map_callback)
        self.map_sub = rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid,
                                        self.map_callback)

        self.map_click_sub = rospy.Subscriber("/map_image_mouse_left", Point,
                                              self.map_click_callback)
        self.image_pub = rospy.Publisher("/map_image", Image, queue_size=10)
        self.debug_pub = rospy.Publisher("/debug_image", Image, queue_size=10)
        self.debug_pub2 = rospy.Publisher("/debug_image2", Image, queue_size=10)
        self.debug_pub3 = rospy.Publisher("/debug_image3", Image, queue_size=10)
        self.debug_pub4 = rospy.Publisher("/debug_image4", Image, queue_size=10)
        self.debug_pub5 = rospy.Publisher("/debug_image5", Image, queue_size=10)
        # self.debug_pub6 = rospy.Publisher("/debug_image6", Image, queue_size=10)
        self.object_info_pub = rospy.Publisher("/obj_info", Image,
                                               queue_size=10)
        self.command_pub = rospy.Publisher("/robot_command", RobotCommand,
                                           queue_size=10)
        self.obstacle_map_pub = rospy.Publisher("/obstacle_map", Image,
                                                queue_size=10)
        self.frontier_map_pub = rospy.Publisher("/frontier_map", Image,
                                                queue_size=10)
        self.seg_map_pub = rospy.Publisher("/seg_map", Image, queue_size=10)
        self.sightings_map_pub = rospy.Publisher("/sighting_map",
                                                 Image, queue_size=10)
        self.goal_map_pub = rospy.Publisher("/goal_map", Image, queue_size=10)
        self.mission_status_pub = rospy.Publisher("/mission_status",
                                                  Image, queue_size=10)
        self.planning_time_pub = rospy.Publisher("/planning_time",
                                                  Float64, queue_size=10)
        self.total_time_pub = rospy.Publisher("/total_time",
                                                Float64, queue_size=10)
        self.c = rospy.ServiceProxy('BBN_infer', BBNInfer)
        rospy.wait_for_service('BBN_infer')
        self.rate = rospy.Rate(20)  # 3hz
        self.dt = 1 / 30.0
        self.iteration = 0

        rospy.on_shutdown(self.save_data)


    def get_dfa_state(self):
        dfa_state = []
        for dfa in self.dfa:
            dfa_state.append(dfa.state)

        return tuple(dfa_state)
    def construct_dfa(self):
        # '1' E
        # '2' !E
        explore_dfa = DFA(0, ['1', '2'])
        explore_dfa.add_transition('1', 0, 1)
        explore_dfa.add_transition('2', 1, 0)

        self.dfa.append(deepcopy(explore_dfa))
        # '3' A picks up car
        # '8' !A & G
        # '9' !A & !G

        car_dfa = DFA(0, ['3', '8', '9'], final_states={2})
        car_dfa.add_transition('3', 0, 1)
        car_dfa.add_transition('8', 1, 2)
        car_dfa.add_transition('9', 1, 0)

        self.dfa.append(deepcopy(car_dfa))

        # '4' B picks up train
        # '10' !B & G
        # '11' !B & !G

        train_dfa = DFA(0, ['4', '10', '11'], final_states={2})
        train_dfa.add_transition('4', 0, 1)
        train_dfa.add_transition('10', 1, 2)
        train_dfa.add_transition('11', 1, 0)

        self.dfa.append(deepcopy(train_dfa))

        # '5' C picks up parasaurolophus
        # '12' !C & G
        # '13' !C & !G

        parasaurolophus_dfa = DFA(0, ['5', '12', '13'], final_states={2})
        parasaurolophus_dfa.add_transition('5', 0, 1)
        parasaurolophus_dfa.add_transition('12', 1, 2)
        parasaurolophus_dfa.add_transition('13', 1, 0)

        self.dfa.append(deepcopy(parasaurolophus_dfa))

        # '6' D picks up stegosaurus
        # '14' !D & G
        # '15' !D & !G

        stegosaurus_dfa = DFA(0, ['6', '14', '15'], final_states={2})
        stegosaurus_dfa.add_transition('6', 0, 1)
        stegosaurus_dfa.add_transition('14', 1, 2)
        stegosaurus_dfa.add_transition('15', 1, 0)

        self.dfa.append(deepcopy(stegosaurus_dfa))

        # '7' F picks up tank
        # '16' !F & G
        # '17' !F & !G

        tank_dfa = DFA(0, ['7', '16', '17'], final_states={2})
        tank_dfa.add_transition('7', 0, 1)
        tank_dfa.add_transition('16', 1, 2)
        tank_dfa.add_transition('17', 1, 0)

        self.dfa.append(deepcopy(tank_dfa))

    def label_function(self, state):
        # '1' E
        # '2' !E

        # '3' A picks up car
        # '4' B picks up train
        # '5' C picks up parasaurolophus
        # '6' D picks up stegosaurus
        # '7' F picks up tank

        # '8' !A & G
        # '9' !A & !G

        # '10' !B & G
        # '11' !B & !G

        # '12' !C & G
        # '13' !C & !G

        # '14' !D & G
        # '15' !D & !G

        # '16' !F & G
        # '17' !F & !G

        symbols = []

        E = self.old_frontier_map[state[0], state[1]]

        if E:
            symbols.append('1')
            return symbols
        else:
            symbols.append('2')

        A = (state[2] == 1)
        B = (state[2] == 2)
        C = (state[2] == 3)
        D = (state[2] == 4)
        F = (state[2] == 5)

        G = self.old_goal_map[state[0], state[1]]

        if A:
            symbols.append('3')
        else:
            if G:
                symbols.append('8')
            else:
                symbols.append('9')

        if B:
            symbols.append('4')
        else:
            if G:
                symbols.append('10')
            else:
                symbols.append('11')

        if C:
            symbols.append('5')
        else:
            if G:
                symbols.append('12')
            else:
                symbols.append('13')

        if D:
            symbols.append('6')
        else:
            if G:
                symbols.append('14')
            else:
                symbols.append('15')

        if F:
            symbols.append('7')
        else:
            if G:
                symbols.append('16')
            else:
                symbols.append('17')


        return symbols

    def get_next_dfa_state(self, symbols, dfa_state):
        next_dfa_states = []
        for i in range(len(dfa_state)):
            next_dfa_state = None
            for symbol in symbols:
                next_dfa_state = self.dfa[i].get_transition(symbol, dfa_state[i])
                if next_dfa_state is not None:
                    break

            if next_dfa_state is None:
                next_dfa_state = dfa_state[i]

            next_dfa_states.append(next_dfa_state)

        return tuple(next_dfa_states)

    @staticmethod
    def is_terminal(dfa_state):
        return (dfa_state[1] == 2 and dfa_state[2] == 2 and dfa_state[3] == 2 and dfa_state[4] == 2) or dfa_state[0]

    def get_reward(self, dfa_state, next_dfa_state):
        reward = 0.0
        if not (dfa_state[1] == 2 and dfa_state[2] == 2) and (next_dfa_state[1] == 2 and next_dfa_state[2] == 2):
            reward += self.task_1_reward

        if not (dfa_state[3] == 2 and dfa_state[4] == 2) and (next_dfa_state[3] == 2 and next_dfa_state[4] == 2):
            reward += self.task_2_reward

        if reward and dfa_state[5] == 2:
            reward -= self.task_3_reward

        if dfa_state[5] != 2 and next_dfa_state[5] == 2:
            reward += self.task_3_reward

        if not dfa_state[0] and next_dfa_state[0]:
            reward += self.explore_reward

        return reward

    def update_dfa_state(self, symbols):
        for dfa in self.dfa:
            dfa.update_state(symbols)

        self.dfa_state = self.get_dfa_state()

    def save_data(self):
        with open("/media/bear/T7/Toys/Replanning_Contracted/Metric.pkl", "wb") as fp:  # Pickling
            pickle.dump([self.log_1, self.log_2, self.log_3,
                             self.log_4, self.log_5, self.log_6, self.log_7], fp)

        print('total_planning time is ', self.planning_time)

    def acquire_transition(self, state):
        state_actions = self.P.get(state, False)
        dfa_state = state[2:]
        pos_grid = np.unravel_index(state[0],(self.old_height, self.old_width))
        if not state_actions:
            self.P[state] = {}
            for action in range(self.nA):
                action_deltas = np.take(self.action_deltas, [action, action + 1, action - 1], axis=0, mode='wrap')
                next_states = np.unravel_index(state[0], (self.old_height, self.old_width)) + action_deltas

                next_states[next_states[:, 0] < 0, 0] = 0
                next_states[next_states[:, 1] < 0, 1] = 0
                next_states[next_states[:, 0] > self.old_height - 1, 0] = self.old_height - 1
                next_states[next_states[:, 1] > self.old_width - 1, 1] = self.old_width - 1

                next_states_indices = np.ravel_multi_index(next_states.T, (self.old_height, self.old_width))

                out_of_free_grids = np.in1d(next_states_indices, self.states_index, invert=True)
                pnsrt = []

                for i in range(3):
                    next_state = next_states[i]
                    labels = self.label_function([next_state[0], next_state[1], state[1]])
                    next_dfa_state = self.get_next_dfa_state(labels, dfa_state)
                    reward = self.get_reward(dfa_state, next_dfa_state)
                    terminal = out_of_free_grids[i] or self.is_terminal(next_dfa_state)
                    pnsrt.append((self.action_probabilities[i], (next_states_indices[i], state[1]) + next_dfa_state, reward, terminal))

                self.P[state][action] = pnsrt

            if not state[1]:
                for index in range(len(self.target_classes)):
                    target_class = self.target_classes[index]
                    if target_class not in self.old_grip_probs:
                        continue
                    pnsrt = []
                    action = self.nA + index
                    p = self.old_grip_probs[target_class][pos_grid[0], pos_grid[1]]
                    if not p or state[index + 3] == 2:
                        continue
                    labels = self.label_function([pos_grid[0], pos_grid[1], index + 1])

                    next_dfa_state = self.get_next_dfa_state(labels, dfa_state)
                    reward = self.get_reward(dfa_state, next_dfa_state)
                    terminal = self.is_terminal(next_dfa_state)
                    pnsrt.append((p, (state[0], index + 1) + next_dfa_state, reward, terminal))

                    labels = self.label_function([pos_grid[0], pos_grid[1], 0])
                    next_dfa_state = self.get_next_dfa_state(labels, dfa_state)
                    reward = self.get_reward(dfa_state, next_dfa_state)
                    terminal = self.is_terminal(next_dfa_state)
                    pnsrt.append((1 - p, (state[0], 0) + next_dfa_state, reward, terminal))
                    self.P[state][action] = pnsrt

            elif self.old_goal_map[pos_grid[0], pos_grid[1]]:
                action = self.nA + len(self.target_classes)
                pnsrt = []
                labels = self.label_function([pos_grid[0], pos_grid[1], 0])
                next_dfa_state = self.get_next_dfa_state(labels, dfa_state)
                reward = self.get_reward(dfa_state, next_dfa_state)
                terminal = self.is_terminal(next_dfa_state)
                pnsrt.append((1, (state[0], 0) + next_dfa_state, reward, terminal))

                self.P[state][action] = pnsrt

            return self.P.get(state, False)
        else:
            return state_actions

    def select_action(self, state, train=False):
        StateActions = self.acquire_transition(state)
        if not StateActions:
            return -1

        Q_s = {}
        for a in StateActions:
            transitions = StateActions[a]
            for transition in transitions:
                probability = transition[0]
                nextstate = transition[1]
                reward = transition[2]
                terminal = transition[3]

                if terminal:
                    Q_s[a] = Q_s.get(a, 0) + probability * reward
                else:
                    if (nextstate[0],) + nextstate[2:] not in self.V:
                        print('state', state)
                        print('probability', probability)
                        print('nextstate', nextstate)
                        print('reward', reward)
                        print('terminal', terminal)
                    Q_s[a] = Q_s.get(a, 0) + probability * (reward + self.gamma * self.V[(nextstate[0],) + nextstate[2:]])

        # select an action
        actions = Q_s.keys()
        Qs_values = Q_s.values()

        actions = np.asarray(actions)
        Qs_values = np.asarray(Qs_values)

        if train:
            action_probs = softmax(Qs_values)
            action = np.random.choice(actions, p=action_probs)
            self.value_mutex.acquire()
            self.V[(state[0],) + state[2:]] = np.max(Qs_values)
            self.value_mutex.release()
        else:
            action_indexes = np.flatnonzero(Qs_values == np.max(Qs_values))
            actions = actions[action_indexes]
            action = np.random.choice(actions)
        return action

    def trial(self, s0):
        done = False
        state = s0
        states = []
        while not done:
            action = self.select_action(state, train=True)
            # return a reward and new state
            transitions = self.P[state][action]
            probabilities, nextstates, rewards, terminals = zip(*transitions)

            if np.min(probabilities) < 0:
                probabilities = np.asarray(probabilities) - np.min(
                    probabilities)
            probabilities = probabilities / np.sum(probabilities)

            index = np.random.choice(len(nextstates), p=probabilities)

            new_state = nextstates[index]
            done = terminals[index]

            # append state, action, reward to episode
            states.append(state)
            # update state to new state

            # print(state, action, new_state, rewards[index], done)
            state = new_state

            if len(states) > 10 * (self.old_width + self.old_height):
                break

    def wild_fire(self, Vh, non_states, pos_grid):
        if np.amax(Vh) > 0:
            count = 0
            kernel = np.ones((3, 3), np.uint8)

            is_states = np.logical_not(non_states)
            old_non_zero = np.count_nonzero(Vh)
            new_non_zero = -1
            while not np.all(Vh[is_states]) and old_non_zero != new_non_zero:
                old_non_zero = new_non_zero
                Vh_new = cv2.dilate(Vh, kernel, iterations=1,
                                    borderType=cv2.BORDER_ISOLATED) * self.gamma
                Vh = np.maximum(Vh, Vh_new)
                Vh[non_states] = 0
                new_non_zero = np.count_nonzero(Vh)
                count += 1

        return Vh

    def run(self):
        while not rospy.is_shutdown():
            if self.obstacle_map is not None and not self.initial_obs and not self.mission_status:
                pos_grid = np.asarray(self.to_grid(self.pos))
                start_time = time()

                probs = [self.max_class_probs.get('train', 0),
                         self.max_class_probs.get('car', 0),
                         self.max_class_probs.get('stegosaurus', 0),
                         self.max_class_probs.get('parasaurolophus', 0),
                         self.max_class_probs.get('tank', 0)]

                self.mutex.acquire()
                self.old_grid_map = self.grid_map_grey.copy()
                self.old_goal_map = self.goal_map.copy()
                self.old_grip_maps = self.grip_maps.copy()
                self.mutex.release()

                if self.old_goal_map is None:
                    continue

                if self.old_grip_maps and self.old_goal_map is not None and self.old_grid_map is not None and self.grid_map_grey[pos_grid[1], pos_grid[0]] == 255:
                    if self.old_grid_map.shape != self.old_goal_map.shape:
                        continue

                    for old_grip_map in self.old_grip_maps.values():
                        if old_grip_map.shape != self.old_grid_map.shape:
                            continue

                    bound_start_time = time()
                    task_prob = upper_bound_efficient(probs, self.gains, self.dfa_state[1:6], np.flipud(pos_grid), self.dfa_states,
                                    self.dfa_states_indices, self.old_grid_map, self.old_grip_maps,
                                    self.old_goal_map)
                    
                    bound_end_time = time()

                    print("bound computation time is", bound_end_time - bound_start_time)

                    # print('upper bound is ', task_prob)
                    if  (task_prob - self.old_task_prob) > 0.4 and (task_prob - self.prev_task_prob) < 0.01: # 0.5
                        self.old_task_prob = task_prob
                        self.replan = True
                        print('I am here metric is', task_prob)

                    self.prev_task_prob = task_prob

                    self.log_1.append(probs[4])
                    self.log_2.append(probs[3])
                    self.log_3.append(probs[2])
                    self.log_4.append(probs[1])
                    self.log_5.append(probs[0])
                    self.log_6.append(self.dfa_state)
                    self.log_7.append(task_prob)


                if self.replan:
                    print('start replanning')
                    self.iteration = 0
                    self.mutex.acquire()
                    print('acquire mutex')
                    self.old_frontier_map = self.frontier_map.copy()
                    self.old_frontier_prob = self.frontier_prob.copy()
                    self.old_frontier_phase = self.frontier_phase.copy()

                    self.old_goal_prob = self.goal_prob.copy()
                    self.old_grip_probs = self.grip_probs.copy()

                    self.old_height = self.height
                    self.old_width = self.width
                    self.old_resolution = self.resolution
                    self.old_offset_x = self.offset_x
                    self.old_offset_y = self.offset_y

                    self.old_objs_of_interest = self.objs_of_interest.copy()

                    print('copy data')
                    self.mutex.release()
                    print('mutex release')
                    self.planner_type = 'Preference'

                    state_space_grid = (self.old_grid_map == 255).astype(np.uint8)
                    states = np.nonzero(state_space_grid)
                    self.states_index = np.ravel_multi_index(states, (self.old_height, self.old_width))
                    states = np.asarray(states).T

                    self.V = {}
                    self.P = {}

                    non_states = (self.old_grid_map != 255)

                    if self.dfa_state[1] == 2 and self.dfa_state[2] == 2:
                        self.task_1_reward = 0.0
                        self.task_3_reward = 0.0

                    if self.dfa_state[3] == 2 and self.dfa_state[4] == 2:
                        self.task_2_reward = 0.0
                        self.task_3_reward = 0.0

                    if self.dfa_state[5] == 2:
                        self.task_3_reward = 0.0

                    Vh_frontier = self.explore_reward * self.old_frontier_map
                    Vh_frontier = self.wild_fire(Vh_frontier, non_states, pos_grid)

                    Task1_V = {}
                    Task1_V[(2, 2)] = 0

                    Vh_task1 = self.task_1_reward * self.old_goal_map
                    Vh_task1 = self.wild_fire(Vh_task1, non_states, pos_grid)
                    Task1_V[(2, 1)] = Vh_task1
                    Task1_V[(1, 2)] = Vh_task1

                    Vh_grip_car = Vh_task1.copy()
                    Vh_grip_car[np.logical_not(self.old_grip_maps['car'])] = 0
                    Vh_grip_car *= self.max_class_probs['car']
                    Vh_grip_car = self.wild_fire(Vh_grip_car, non_states, pos_grid)

                    Task1_V[(0, 2)] = Vh_grip_car

                    Vh_grip_train = Vh_task1.copy()
                    Vh_grip_train[np.logical_not(self.old_grip_maps['train'])] = 0
                    Vh_grip_train *= self.max_class_probs['train']
                    Vh_grip_train = self.wild_fire(Vh_grip_train, non_states, pos_grid)

                    Task1_V[(2, 0)] = Vh_grip_train

                    Vh_release_train = Vh_grip_car.copy()
                    Vh_release_train[np.logical_not(self.old_goal_map)] = 0
                    Vh_release_train = self.wild_fire(Vh_release_train, non_states, pos_grid)

                    Task1_V[(0, 1)] = Vh_release_train

                    Vh_release_car = Vh_grip_train.copy()
                    Vh_release_car[np.logical_not(self.old_goal_map)] = 0
                    Vh_release_car = self.wild_fire(Vh_release_car, non_states, pos_grid)

                    Task1_V[(1, 0)] = Vh_release_car

                    temp1 = Vh_release_car.copy()
                    temp2 = Vh_release_train.copy()

                    temp1[np.logical_not(self.old_grip_maps['car'])] = 0
                    temp1 *= self.max_class_probs['car']

                    temp2[np.logical_not(self.old_grip_maps['train'])] = 0
                    temp2 *= self.max_class_probs['train']

                    temp1 = self.wild_fire(temp1, non_states, pos_grid)
                    temp2 = self.wild_fire(temp2, non_states, pos_grid)

                    Task1_V[(0, 0)] = np.maximum(temp1, temp2)

                    Task2_V = {}
                    Task2_V[(2, 2)] = 0

                    Vh_task2 = self.task_2_reward * self.old_goal_map
                    Vh_task2 = self.wild_fire(Vh_task2, non_states, pos_grid)

                    Task2_V[(2, 1)] = Vh_task2
                    Task2_V[(1, 2)] = Vh_task2

                    Vh_grip_parasaurolophus = Vh_task2.copy()
                    Vh_grip_parasaurolophus[np.logical_not(self.old_grip_maps['parasaurolophus'])] = 0
                    Vh_grip_parasaurolophus *= self.max_class_probs['parasaurolophus']
                    Vh_grip_parasaurolophus = self.wild_fire(Vh_grip_parasaurolophus, non_states, pos_grid)

                    Task2_V[(0, 2)] = Vh_grip_parasaurolophus

                    Vh_grip_stegosaurus = Vh_task2.copy()
                    Vh_grip_stegosaurus[np.logical_not(self.old_grip_maps['stegosaurus'])] = 0
                    Vh_grip_stegosaurus *= self.max_class_probs['stegosaurus']
                    Vh_grip_stegosaurus = self.wild_fire(Vh_grip_stegosaurus, non_states, pos_grid)

                    Task2_V[(2, 0)] = Vh_grip_stegosaurus

                    Vh_release_stegosaurus = Vh_grip_parasaurolophus.copy()
                    Vh_release_stegosaurus[np.logical_not(self.old_goal_map)] = 0
                    Vh_release_stegosaurus = self.wild_fire(Vh_release_stegosaurus, non_states, pos_grid)

                    Task2_V[(0, 1)] = Vh_release_stegosaurus

                    Vh_release_parasaurolophus = Vh_grip_stegosaurus.copy()
                    Vh_release_parasaurolophus[np.logical_not(self.old_goal_map)] = 0
                    Vh_release_parasaurolophus = self.wild_fire(Vh_release_parasaurolophus, non_states, pos_grid)

                    Task2_V[(1, 0)] = Vh_release_parasaurolophus

                    temp1 = Vh_release_stegosaurus.copy()
                    temp2 = Vh_release_parasaurolophus.copy()

                    temp1[np.logical_not(self.old_grip_maps['stegosaurus'])] = 0
                    temp1 *= self.max_class_probs['stegosaurus']

                    temp2[np.logical_not(self.old_grip_maps['parasaurolophus'])] = 0
                    temp2 *= self.max_class_probs['parasaurolophus']

                    temp1 = self.wild_fire(temp1, non_states, pos_grid)
                    temp2 = self.wild_fire(temp2, non_states, pos_grid)

                    Task2_V[(0, 0)] = np.maximum(temp1, temp2)

                    Vh_task3 = self.task_3_reward * self.old_goal_map
                    Vh_task3 = self.wild_fire(Vh_task3, non_states, pos_grid)

                    Vh_grip_tank = Vh_task3.copy()
                    Vh_grip_tank[np.logical_not(self.old_grip_maps['tank'])] = 0
                    Vh_grip_tank *= self.max_class_probs['tank']
                    Vh_grip_tank = self.wild_fire(Vh_grip_tank, non_states, pos_grid)

                    Task3_V = {}
                    Task3_V[2] = 0
                    Task3_V[1] = Vh_task3
                    Task3_V[0] = Vh_grip_tank

                    Task1_states = [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2)]

                    for i in Task1_states:
                        for j in Task1_states:
                            for k in range(3):
                                dfa_state = i + j + (k,)
                                temp = np.maximum(np.maximum(Vh_frontier, Task1_V[i] + Task2_V[j]), Task3_V[k])
                                for l in range(states.shape[0]):
                                    state_index = self.states_index[l]
                                    state = states[l]
                                    # exploration state is always zero.
                                    product_state = (state_index, 0) + dfa_state
                                    self.V[product_state] = temp[state[0], state[1]]

                    print('construct heuristic')
                    self.replan = False

                start_time = time()
                pos_grid = np.asarray(self.to_grid_old(self.pos))
                pos_index = np.ravel_multi_index(np.flipud(pos_grid), (self.old_height, self.old_width))

                if pos_index not in self.states_index:
                    print('replan because enter unknown states')
                    self.replan = True
                    continue

                symbols = self.label_function([pos_grid[1], pos_grid[0], self.grip_state])
                old_dfa_state = deepcopy(self.dfa_state)
                self.update_dfa_state(symbols)
                if not self.goal_status:
                    # self.goal_status = (old_dfa_state != self.dfa_state) or (self.get_reward(old_dfa_state, self.dfa_state) > 0)
                    self.goal_status =  self.dfa_state[0]
                if old_dfa_state != self.dfa_state:
                    print(old_dfa_state, self.dfa_state, symbols)
 

                self.trial((pos_index, self.grip_state)  + self.dfa_state)
                self.iteration += 1

                end_time = time()

                #print('time spent in iteration is ', end_time - start_time)
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2  # 0.6
                font_thickness = 1
                mission_info_image = np.zeros((250, 600), dtype=np.uint8)
                text_coord = [0, 0]
                text1 = "DFA state: " + str(self.dfa_state)
                text2 = "Intermediate goal reached: " + str(self.goal_status)
                text3 = "Most probable object: "
                text4 = "Current planner: " + self.planner_type
                text5 = "Value Iteration num: " + str(self.iteration)
                texts = [text1, text2, text3, text4, text5]
                for text in texts:
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    _, text_h = text_size
                    text_coord[1] = text_coord[1] + text_h + 25
                    cv2.putText(mission_info_image, text, tuple(text_coord), font, font_scale, 255, font_thickness)

                image_message = self.bridge.cv2_to_imgmsg(mission_info_image, encoding="mono8")
                self.mission_status_pub.publish(image_message)
                end_time = time()
                if self.iteration < 300:
                    self.planning_time += (end_time - start_time)
                if self.iteration == 300:
                    self.planning_time_pub.publish(self.planning_time)
                self.total_time_pub.publish(end_time - self.start_time)
            # self.rate.sleep()

    def angle_tracking(self, angle_error):
        rot_dir = 1.0
        if angle_error < 0:
            rot_dir = -1.0

        angular_correction = 0.0
        if np.abs(angle_error) > (
            self.max_turn_speed * 10.0 * self.dt):
            angular_correction = self.max_turn_speed
        else:
            angular_correction = np.abs(angle_error) / 2.0

        angular_correction = np.clip(rot_dir * angular_correction,
                                     -self.max_turn_speed,
                                     self.max_turn_speed)

        return angular_correction

    def grip_callback(self, msg):
        # '3' A picks up car
        # '4' B picks up train
        # '5' C picks up parasaurolophus
        # '6' D picks up stegosaurus
        # '7' F picks up tank

 

        if msg.grip:
            if msg.grip_id == 187:
                self.grip_state = 1
            elif msg.grip_id == 188:
                self.grip_state = 4
            elif msg.grip_id == 189:
                self.grip_state = 2
            elif msg.grip_id == 190:
                self.grip_state = 3
            elif msg.grip_id == 191:
                self.grip_state = 5
            else:
                print('unknown grip id')
        else:
            self.grip_state = 0

    def odom_callback(self, msg):
        # only takes the covariance, the pose is taken from tf transformation
        self.pos_var = msg.pose.covariance[0]
        self.ang_var = msg.pose.covariance[-1]

        self.sigma_p = np.diag([self.pos_var, self.pos_var, self.ang_var])
        self.sigma_p_inv = inv(self.sigma_p)

        self.vel = msg.twist.twist.linear.x

        from_frame_rel = self.target_frame
        to_frame_rel = 'map'

        try:
            trans = self.tfBuffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rospy.Time(0))

            self.pos = np.asarray([trans.transform.translation.x,
                                   trans.transform.translation.y])

            self.old_ang = self.ang
            self.ang = tf.transformations.euler_from_quaternion(
                [trans.transform.rotation.x,
                 trans.transform.rotation.y,
                 trans.transform.rotation.z,
                 trans.transform.rotation.w])[-1]

        except (TransformException, ConnectivityException,
                ExtrapolationException) as ex:
            rospy.loginfo('Could not transform %s to %s: ', to_frame_rel,
                          from_frame_rel)
            return

        if self.initial_obs and self.old_ang is not None:
            self.observed_ang += np.arctan2(
                np.sin(self.ang - self.old_ang),
                np.cos(self.ang - self.old_ang))

            if self.observed_ang > 2 * np.pi:
                self.initial_obs = False
                self.start_time = time()

        command_msg = RobotCommand()
        if self.initial_obs:
            command_msg.linear_vel = 0
            command_msg.angular_vel = self.max_turn_speed
        elif self.mission_status:
            command_msg.linear_vel = 0
            command_msg.angular_vel = 0
        else:
            pos_grid = self.to_grid(self.pos)
            if self.grid_map_grey[pos_grid[1], pos_grid[0]] != 255:
                grid_map_convervative = cv2.erode(self.grid_map_grey, np.ones((3,3), dtype=np.uint8), iterations=1)

                free_pnts = np.asarray(np.where(grid_map_convervative == 255)).T
                distances = norm(free_pnts - np.flipud(pos_grid), axis=1)
                goal_grid = free_pnts[np.argmin(distances), :]

                self.goal = self.to_map(np.flipud(goal_grid))
                to_waypoint = self.goal - self.pos
                angle_error = np.arctan2(to_waypoint[1],
                                         to_waypoint[0]) - self.ang
                angle_error = np.arctan2(np.sin(angle_error),
                                         np.cos(angle_error))

                if np.abs(angle_error) < self.angular_error_threshold:
                    # speed up to max
                    # the positive direction of car bearing is opposite in
                    # simulation
                    new_velocity = (-self.vel - self.max_linear_speed) / 2.0
                else:
                    # slow down to 0
                    new_velocity = -self.vel / 2.0

                command_msg.linear_vel = new_velocity
                command_msg.angular_vel = self.angle_tracking(angle_error)

            elif self.iteration > 300 and not self.goal_status and not self.replan:
                pos_grid = self.to_grid_old(self.pos)
                pos_index = np.ravel_multi_index(np.flipud(pos_grid),(self.old_height, self.old_width))
                # symbols = self.label_function([pos_grid[1], pos_grid[0], self.grip_state])
                # self.update_dfa_state(symbols)
                # action_index = self.policy[pos_index]

                if self.old_grid_map[pos_grid[1], pos_grid[0]] == 255:
                    self.value_mutex.acquire()
                    if self.old_goal_map[pos_grid[1], pos_grid[0]] and self.dfa_state == (0, 2, 1, 2, 2, 0):
                        print('inside goal map')
                    action_index = self.select_action((pos_index, self.grip_state) + self.dfa_state)
                    self.value_mutex.release()

                    if action_index >= 0 and action_index < self.nA:
                        goal_grid = np.flipud(
                            self.action_deltas[action_index]) + \
                                    np.asarray(pos_grid)
                        # if self.goal is None:
                        self.goal = self.to_map_old(goal_grid)
                        to_waypoint = self.goal - self.pos
                        angle_error = np.arctan2(to_waypoint[1],
                                                 to_waypoint[0]) - self.ang
                        angle_error = np.arctan2(np.sin(angle_error),
                                                 np.cos(angle_error))


                        if np.abs(angle_error) < self.angular_error_threshold:
                            # speed up to max
                            # the positive direction of car bearing is opposite in
                            # simulation
                            new_velocity = (-self.vel - self.max_linear_speed) / 2.0
                        else:
                            # slow down to 0
                            new_velocity = -self.vel / 2.0
                        command_msg.linear_vel = new_velocity
                        command_msg.angular_vel = self.angle_tracking(angle_error)

                    elif action_index == self.nA + len(self.target_classes):
                        print('release object')
                        command_msg.release = True
                    else:
                        index = action_index - self.nA
                        target_class = self.target_classes[index]
                        command_msg.grip = True
                        recovered_state = (pos_index, self.grip_state) + self.dfa_state
                        print(action_index, recovered_state, recovered_state[3 + index])
                        print('grip object')
                        command_msg.grip_id = self.objs_of_interest[target_class]
                else:
                    grid_map_convervative = cv2.erode(self.old_grid_map, np.ones((3, 3),dtype=np.uint8), iterations=1)

                    free_pnts = np.asarray(np.where(grid_map_convervative == 255)).T
                    distances = norm(free_pnts - np.flipud(pos_grid), axis=1)
                    goal_grid = free_pnts[np.argmin(distances), :]

                    self.goal = self.to_map(np.flipud(goal_grid))
                    to_waypoint = self.goal - self.pos
                    angle_error = np.arctan2(to_waypoint[1],
                                             to_waypoint[0]) - self.ang
                    angle_error = np.arctan2(np.sin(angle_error),
                                             np.cos(angle_error))

                    if np.abs(angle_error) < self.angular_error_threshold:
                        # speed up to max
                        # the positive direction of car bearing is opposite in
                        # simulation
                        new_velocity = (
                                               -self.vel - self.max_linear_speed) / 2.0
                    else:
                        # slow down to 0
                        new_velocity = -self.vel / 2.0

                    command_msg.linear_vel = new_velocity
                    command_msg.angular_vel = self.angle_tracking(angle_error)

            else:
                if self.goal_status:
                    goal_phase = None
                    if self.dfa[0].state:
                        goal_phase = self.old_frontier_phase
                    if goal_phase is not None:
                        if self.ang_calculation:
                            pos_grid = self.to_grid_old(self.pos)
                            angle_error1 = goal_phase[pos_grid[1],
                                                           pos_grid[0]] \
                                           - self.ang + 0.3 * np.pi

                            angle_error2 = goal_phase[pos_grid[1],
                                                           pos_grid[0]] \
                                           - self.ang - 0.3 * np.pi

                            angle_error1 = np.arctan2(np.sin(angle_error1),
                                                      np.cos(angle_error1))

                            angle_error2 = np.arctan2(np.sin(angle_error2),
                                                      np.cos(angle_error2))

                            if np.abs(angle_error1) < np.abs(angle_error2):
                                self.ang_goal1 = goal_phase[pos_grid[1],
                                                                 pos_grid[0]] \
                                                 + 0.3 * np.pi

                                self.ang_goal2 = goal_phase[pos_grid[1],
                                                                 pos_grid[0]] \
                                                 - 0.3 * np.pi
                            else:
                                self.ang_goal1 = goal_phase[pos_grid[1],
                                                                 pos_grid[0]] \
                                                 - 0.3 * np.pi

                                self.ang_goal2 = goal_phase[pos_grid[1],
                                                                 pos_grid[0]] \
                                                 + 0.3 * np.pi

                        self.ang_calculation = False

                        if not self.ang_goal1_status:
                            angle_error = self.ang_goal1 - self.ang
                            angle_error = np.arctan2(np.sin(angle_error),
                                                     np.cos(angle_error))
                            command_msg.linear_vel = 0
                            command_msg.angular_vel = self.angle_tracking(
                                angle_error)

                            if np.abs(angle_error) < 0.1:
                                self.ang_goal1_status = True

                            #print('tracking angular goal 1')
                        elif not self.ang_goal2_status:
                            angle_error = self.ang_goal2 - self.ang
                            angle_error = np.arctan2(np.sin(angle_error),
                                                     np.cos(angle_error))

                            command_msg.linear_vel = 0
                            command_msg.angular_vel = self.angle_tracking(
                                angle_error)

                            if np.abs(angle_error) < 0.1:
                                self.ang_goal2_status = True
                        else:
                            self.replan = True
                            self.goal_status = False

                            self.ang_goal1_status = False
                            self.ang_goal2_status = False
                            self.ang_calculation = True

                            command_msg.linear_vel = 0
                            command_msg.angular_vel = 0

                    else:
                        self.replan = True
                        self.goal_status = False

                        self.ang_goal1_status = False
                        self.ang_goal2_status = False
                        self.ang_calculation = True

                        command_msg.linear_vel = 0
                        command_msg.angular_vel = 0

        self.command_pub.publish(command_msg)

    def to_grid(self, pose):
        return (int(round((pose[0] - self.offset_x) / self.resolution)),
                self.height - int(
                    round((pose[1] - self.offset_y) / self.resolution)))

    def to_map(self, coord):
        return (self.offset_x + self.resolution * coord[0],
                self.offset_y + (self.height - coord[1]) * self.resolution)

    def to_grid_old(self, pose):
        return (
        int(round((pose[0] - self.old_offset_x) / self.old_resolution)),
        self.old_height - int(
            round((pose[1] - self.old_offset_y) / self.old_resolution)))

    def to_map_old(self, coord):
        return (self.old_offset_x + self.old_resolution * coord[0],
                self.old_offset_y + (
                        self.old_height - coord[1]) * self.old_resolution)

    def map_click_callback(self, msg):
        if not self.objects:
            return
        clicked_pt = np.asarray([msg.x, msg.y])
        dists = []
        obj_ids = []
        for obj_id, obj in self.objects.items():
            object_grid = np.asarray(self.to_grid(obj.pos))
            dist = norm(clicked_pt - object_grid)
            obj_ids.append(obj_id)
            dists.append(dist)

        clicked_object = self.objects[obj_ids[np.argmin(dists)]]
        if clicked_object.visualized:
            clicked_object.visualized = False
            print('now is false')
        else:
            clicked_object.visualized = True
            print('now is true')

    def map_callback(self, msg):
        """Callback from map
        :type msg: OccupancyGrid
        """
        np_data = np.array([color_converter(e) for e in msg.data])
        room_data = np_data.copy().astype(np.uint8)
        room_data[room_data == 128] = 0
        room_data = np.flipud(np.reshape(room_data, (msg.info.height,
                                                     msg.info.width)))
        goal = MapSegmentationGoal()
        goal.map_origin.position.x = msg.info.origin.position.x
        goal.map_origin.position.y = msg.info.origin.position.y
        goal.map_resolution = msg.info.resolution
        goal.return_format_in_meter = False
        goal.return_format_in_pixel = True
        goal.robot_radius = self.agent_radius

        if self.mutex.acquire(False):
            self.resolution = msg.info.resolution

            self.height = msg.info.height
            self.width = msg.info.width

            self.offset_x = msg.info.origin.position.x
            self.offset_y = msg.info.origin.position.y

            self.grid_map_grey = np.flipud(np.reshape(np_data,
                                                      (self.height, self.width)
                                                      )).astype('uint8')

            if self.pos is not None:
                roi = np.zeros_like(self.grid_map_grey)
                pos_grid = self.to_grid(self.pos)
                cv2.circle(roi, pos_grid,
                           int(self.blind_radius / self.resolution), 1, -1)

                mask = np.logical_and(self.grid_map_grey == 128,
                                      roi.astype(bool))

                self.grid_map_grey[mask] = 255

                room_data[mask] = 255
                goal.input_map = self.bridge.cv2_to_imgmsg(room_data,
                                                           encoding="mono8")
                # self.debug_pub.publish(goal.input_map)
                self.map_client.send_goal(goal)

                numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.grid_map_grey, 8)
                pos_ID = labels[pos_grid[1], pos_grid[0]]
                mask2 = np.logical_and(labels != pos_ID,  self.grid_map_grey == 255)
                self.grid_map_grey[mask2] = 128

                self.grid_map_color = cv2.cvtColor(self.grid_map_grey,
                                                   cv2.COLOR_GRAY2BGR)
                self.findFrontier()

                if self.objects:
                    unfree_region = self.grid_map_grey != 255
                    self.find_sightings(unfree_region)
                    #print('completed find sighting')

                    self.find_goal_region(unfree_region)

                sigma_x = np.sqrt(self.pos_var) / self.resolution
                kx = int(math.ceil(
                    max(3 * np.sqrt(self.pos_var) / self.resolution, 1)))
                kx = 2 * kx + 1

                self.obstacle_map = (np.not_equal(self.grid_map_grey, 255)
                                     ).astype(float)

                self.obstacle_prob = cv2.GaussianBlur(self.obstacle_map,
                                                     (kx, kx), sigma_x,
                                                     borderType=cv2.BORDER_ISOLATED)

                self.frontier_prob = cv2.GaussianBlur(
                    (self.frontier_map > 0).astype(float),
                    (kx, kx), sigma_x,
                    borderType=cv2.BORDER_ISOLATED)

                self.grip_probs = {}
                for target_class, grip_map in self.grip_maps.items():
                    self.grip_probs[target_class] = cv2.GaussianBlur(grip_map,(kx, kx),sigma_x, borderType=cv2.BORDER_ISOLATED) * \
                                                   self.max_class_probs[target_class]

                if self.goal_map is not None:
                    self.goal_prob = cv2.GaussianBlur(self.goal_map, (kx, kx), sigma_x,borderType=cv2.BORDER_ISOLATED)*self.max_table_prob

                image_message = self.bridge.cv2_to_imgmsg(
                    (self.obstacle_map * 255).astype(np.uint8),
                    encoding="mono8")
                self.obstacle_map_pub.publish(image_message)

                image_message2 = self.bridge.cv2_to_imgmsg(
                    (self.frontier_map * 255 / np.amax(self.frontier_map)
                     ).astype(np.uint8), encoding="mono8")
                self.frontier_map_pub.publish(image_message2)

                #print('time spent waiting is', end_time - start_time)

            self.mutex.release()

    def find_sightings(self, unfree):
        self.sightings_map = {}
        self.sightings_phase = {}
        self.sightings_prob = {}

        self.objs_of_interest = {}
        max_class_probs = {}

        for obj_id, obj in self.objects.items():
            object_grid = self.to_grid(obj.pos)

            if object_grid[1] >= self.height:
                continue

            if object_grid[1] < 0:
                continue

            if object_grid[0] < 0:
                continue

            if object_grid[0] >= self.width:
                continue

            # if self.grid_map_grey[object_grid[1], object_grid[0]] == 128:
            #     continue

            for target_class in self.target_classes:
                target_class_index = self.classes.index(target_class)
                if obj.class_probs[target_class_index] > max_class_probs.get(target_class, 0):
                    max_class_probs[target_class] = obj.class_probs[target_class_index]
                    self.objs_of_interest[target_class] = obj_id

        self.max_class_probs = max_class_probs.copy()

        self.grip_maps = {}

        for target_class, obj_id in self.objs_of_interest.items():
            obj = self.objects[obj_id]
            object_grid = self.to_grid(obj.pos)
            object_goal = np.zeros_like(self.grid_map_grey)
            cv2.circle(object_goal, object_grid, int(self.grip_radius/ self.resolution), 1, -1)
            object_goal[unfree] = 0

            self.grip_maps[target_class] = object_goal
            if not np.any(object_goal):
                self.max_class_probs[target_class] = 0

 

    def find_goal_region(self, unfree):
        self.goal_map = None
        self.goal_phase = None
        self.goal_prob = None

        self.max_table_prob = -1.0
        target_class_index = self.classes.index('table')

        for obj_id, obj in self.objects.items():
            object_grid = self.to_grid(obj.pos)

            if object_grid[1] >= self.height:
                continue

            if object_grid[1] < 0:
                continue

            if object_grid[0] < 0:
                continue

            if object_grid[0] >= self.width:
                continue

            if obj.class_probs[target_class_index] > self.max_table_prob:
                self.max_table_prob = obj.class_probs[target_class_index]
                self.table_of_interest = obj_id

        self.max_table_prob = 1.0

        for obj_id in [self.table_of_interest]:
            obj = self.objects[obj_id]
            object_grid = self.to_grid([1.93611, -2.2])
            object_goal = np.zeros_like(self.grid_map_grey)
            # object_phase = np.zeros_like(self.grid_map_grey)
            cv2.circle(object_goal, object_grid, int(self.goal_radius/ self.resolution), 1, -1)
            cv2.circle(object_goal, object_grid, int((self.goal_radius - 0.2) / self.resolution), 0, -1)
            object_goal[unfree] = 0

            #goals = np.nonzero(object_goal)

            # object_phase -np.arctan2(object_grid[1]- goals[0] , object_grid[0]- goals[1])

            self.goal_map = object_goal
            # self.goal_phase = object_phase

        goal_is_reachable = np.any(self.goal_map)
        for target_class in self.max_class_probs:
            if not goal_is_reachable:
                self.max_class_probs[target_class] = 0

        goal_map = (self.goal_map*255).astype('uint8')
        #cv2.circle(sightings_map, center_coordinates, 3, 255, -1)
        image_message = self.bridge.cv2_to_imgmsg(goal_map, encoding="mono8")
        self.goal_map_pub.publish(image_message)


    def findFrontier(self):
        dx = [0, -1, -1, -1, 0, 1, 1, 1]
        dy = [1, 1, 0, -1, -1, -1, 0, 1]

        frontier_mat = cv2.Canny(self.grid_map_grey, 100, 200)

        grad_x = cv2.Scharr(self.grid_map_grey, cv2.CV_32F, 1, 0,
                            borderType=cv2.BORDER_ISOLATED)
        grad_y = cv2.Scharr(self.grid_map_grey, cv2.CV_32F, 0, 1,
                            borderType=cv2.BORDER_ISOLATED)

        free_pnts = np.asarray(np.where(frontier_mat == 255)).T.tolist()
        frontier_mat = np.zeros(np.shape(self.grid_map_grey), dtype=np.uint8)
        row, col = np.shape(self.grid_map_grey)

        for j in range(len(free_pnts)):
            r, c = free_pnts[j]
            if self.grid_map_grey[r, c] == 255:
                for i in range(8):
                    r1 = r + dx[i]
                    c1 = c + dy[i]

                    if 0 <= r1 < row and 0 <= c1 < col:
                        if self.grid_map_grey[r1, c1] == 128:
                            frontier_mat[r, c] = 255
                            break
            elif self.grid_map_grey[r, c] == 128:
                for i in range(8):
                    r1 = r + dx[i]
                    c1 = c + dy[i]

                    if 0 <= r1 < row and 0 <= c1 < col:
                        if self.grid_map_grey[r1, c1] == 255:
                            frontier_mat[r1, c1] = 255
                            # break

      

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_mat, 8)

        self.frontier_num_labels = numLabels
        self.frontier_centroid = centroids
        self.frontier_stat = stats

        s = int(math.ceil(self.agent_radius / self.resolution))
        kernel = np.zeros((2 * s + 1, 2 * s + 1), np.uint8)
        cv2.circle(kernel, (s, s), s, 1, -1)

        obstacle_grids = (self.grid_map_grey == 0).astype('uint8')
        obstacle_grids = cv2.dilate(obstacle_grids, kernel, iterations=1,
                                    borderType=cv2.BORDER_ISOLATED).astype(
            bool)
        self.grid_map_grey[obstacle_grids] = 0

        for i in range(1, numLabels):
            stat = stats[i]
            area = stat[cv2.CC_STAT_AREA]
            indices = (labels == i)
            if area <= 15:
                frontier_mat[indices] = 0
                labels[indices] = 0

        frontier_mat[obstacle_grids] = 0
        labels[obstacle_grids] = 0

        grad_x[frontier_mat == 0] = 0
        grad_y[frontier_mat == 0] = 0

        iteration = 0
        iter_max = 2
        kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).astype(np.uint8)
        frontier_new = np.zeros_like(frontier_mat)
        invalid_frontier = np.logical_or(obstacle_grids,
                                         self.grid_map_grey == 128)

        frontier_mat = labels.astype(float)
        while iteration < iter_max:
            frontier_new = cv2.dilate(frontier_mat, kernel, iterations=1,
                                      borderType=cv2.BORDER_ISOLATED)
            grad_x = cv2.filter2D(grad_x, ddepth=-1, kernel=kernel,
                                  borderType=cv2.BORDER_ISOLATED)
            grad_x = cv2.filter2D(grad_x, ddepth=-1, kernel=kernel,
                                  borderType=cv2.BORDER_ISOLATED)

            frontier_new[invalid_frontier] = 0
            grad_x[invalid_frontier] = 0
            grad_y[invalid_frontier] = 0

            iteration += 1
            if iteration == iter_max:
                frontier_new[np.logical_and(frontier_new > 0,
                                            frontier_mat > 0)] = 0
            frontier_mat = frontier_new

        # self.frontier_map = (frontier_mat == 255).astype(float)
        self.frontier_map = frontier_mat.astype(float)
        self.frontier_map[obstacle_grids] = 0
        self.frontier_map[self.frontier_map > 0] = 1
        for obj_id in self.sightings_map:
            self.sightings_map[obj_id][obstacle_grids] = 0

        self.grid_map_color[self.frontier_map > 0, :] = [48, 172, 119]
        self.frontier_phase = -cv2.phase(grad_x, grad_y) + np.pi

    def semantic_map_callback(self, msg):
        if self.pos_var is None:
            rospy.loginfo('Robot pose covariance is not set.')
            return

        if self.pos is None:
            rospy.loginfo('Robot pose is not set.')
            return

        if self.grid_map_color is None:
            rospy.loginfo('Grid map is not constructed.')
            return

        grid_map = self.grid_map_color.copy()
        startAngle = 0
        endAngle = 360

        pos_grid = self.to_grid(self.pos)

        axesLength = (
        max(int(round(5 * np.sqrt(self.pos_var) / self.resolution)), 1),
        max(int(round(5 * np.sqrt(self.pos_var) / self.resolution)), 1))
        angle = 0
        # Red color in BGR
        color = (0, 0, 255)
        # Line thickness of 5 px
        thickness = 1
        cv2.ellipse(grid_map, pos_grid, axesLength,
                    angle, startAngle, endAngle, color, thickness)

        colors = [(142, 47, 126), (0, 255, 0)]
        # grid_map = np.ones_like(grid_map)*255
        for obj_msg in msg.objects:
            obj_id = obj_msg.id
            covariance = np.reshape(obj_msg.covariance, (2, 2))
            obj_pose = np.asarray([obj_msg.x, obj_msg.y])
            class_probs = np.asarray(obj_msg.probability)

            if obj_id not in self.objects:
                self.objects[obj_id] = MapObject(obj_id, obj_pose, covariance,
                                                 class_probs)
            else:
                obj = self.objects[obj_id]
                obj.update(obj_pose, covariance, class_probs)

        total_text_width = 0
        total_text_height = 0
        text_color_bg = (255, 255, 255)
        text_color = (0, 0, 0)

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1  # 0.6
        font_thickness = 1

        for obj_id, obj in self.objects.items():
            if obj.visualized:
                text1 = 'object ID: ' + str(obj_id)

                text2 = 'pose: ' + str("%.2f" % obj.pos[0]) + ', ' + \
                        str("%.2f" % obj.pos[1])

                text3 = 'pose var: ' + str(
                    "%.2f" % (obj.pos_var[0, 0] * 10000.0)) + ', ' + \
                        str("%.2f" % (
                            obj.pos_var[0, 1] * 10000.0)) + ', ' + \
                        str("%.2f" % (
                            obj.pos_var[1, 0] * 10000.0)) + ', ' + \
                        str("%.2f" % (obj.pos_var[1, 1] * 10000.0))

                class_max = np.argmax(obj.class_probs)
                text4 = self.classes[class_max] + ': ' + str(
                    "%.2f" % obj.class_probs[class_max])

                text5 = ''

                texts = [text1, text2, text3, text4, text5]

                for text in texts:
                    text_size, _ = cv2.getTextSize(text, font, font_scale,
                                                   font_thickness)
                    text_w, text_h = text_size
                    total_text_height += (text_h + 1)
                    if text_w > total_text_width:
                        total_text_width = text_w

        if total_text_width and total_text_height:
            obj_info_image = np.zeros((total_text_height, total_text_width),
                                      dtype=np.uint8)
            text_coord = [0, 0]
            for obj_id, obj in self.objects.items():

                if obj.visualized:
                    text1 = 'object ID: ' + str(obj_id)

                    text2 = 'pose: ' + str("%.2f" % obj.pos[0]) + ', ' + \
                            str("%.2f" % obj.pos[1])

                    text3 = 'pose var: ' + str(
                        "%.2f" % (obj.pos_var[0, 0] * 10000.0)) + ', ' + \
                            str("%.2f" % (
                                obj.pos_var[0, 1] * 10000.0)) + ', ' + \
                            str("%.2f" % (
                                obj.pos_var[1, 0] * 10000.0)) + ', ' + \
                            str("%.2f" % (obj.pos_var[1, 1] * 10000.0))

                    class_max = np.argmax(obj.class_probs)
                    text4 = self.classes[class_max] + ': ' + str(
                        "%.2f" % obj.class_probs[class_max])

                    text5 = ''

                    texts = [text1, text2, text3, text4, text5]

                    for text in texts:
                        text_size, _ = cv2.getTextSize(text, font, font_scale,
                                                       font_thickness)
                        _, text_h = text_size
                        text_coord[1] = text_coord[1] + text_h + 1
                        cv2.putText(obj_info_image, text, tuple(text_coord),
                                    font,
                                    font_scale, 255, font_thickness)

            image_message = self.bridge.cv2_to_imgmsg(obj_info_image,
                                                      encoding="mono8")
            self.object_info_pub.publish(image_message)

        image_message = self.bridge.cv2_to_imgmsg(grid_map,
                                                  encoding="bgr8")
        image_message.header.seq = self.seq
        image_message.header.stamp = rospy.get_rostime()
        image_message.header.frame_id = 'map'
        self.image_pub.publish(image_message)
        self.seq += 1


if __name__ == '__main__':
    rospy.init_node("planner")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = Planner()
    try:
        whatever.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
