import numpy as np
import gym
import dartpy as dart
import xml.etree.ElementTree as ET

from core.dartHelper import buildFromFile
from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from numpy.linalg import inv
from numba import njit

from learning.ray_model import MuscleNN
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

import ray
import os
from copy import deepcopy

## Muscle Tuple : reduced_JtA, JtP, JtA

@njit
def mat_inv(mat):
    return inv(mat)

@njit
def compute_wp(updated_pos, ap_to_lbsps_i, ap_lbs_weight):
    return [np.dot(updated_pos[ap_to_lbsps_i[i]].T, ap_lbs_weight[i]) for i in range(len(ap_to_lbsps_i))]

class Env(gym.Env):
    def __init__(self, metadata):
        
        self.world = dart.simulation.World()
        
        self.skel_info = None
        self.new_skel_info = None
        self.root_name = None

        # self.muscle_path = None
        self.skel = None
        self.target_skel = None
        self.ground = None 
        self.step_counter = 0

        self.simulationHz = 480
        self.controlHz = 30
        self.bvhs = None
        self.bvh_idx = 0
        self.reward_bins = []

        self.ees_name = ["Head", "HandL", "HandR", "TalusL", "TalusR"]
        
        ## actuator Type 
        self.actuator_type = "pd_ref_residual"

        ## Muscle Configuration
        self.muscles = None

        self.muscle_info = None
        self.new_muscle_info = None

        self.muscle_pos = []
        self.muscle_control_type = "activation" # mass or activation or musclepd
        self.muscle_activation_levels = None
        self.muscle_nn = None
        self.muscle_buffer = [[],[],[]]

        self.kp = 0.0
        self.kv = 0.0
        self.learning_gain = False

        self.loading_xml(metadata)
        
        self.world.setTimeStep(1.0 / self.simulationHz)
        self.world.setGravity([0, -9.81, 0])
        
        self.target_pos = None
        self.target_vel = None
        
        self.target_displacement = np.zeros(self.skel.getNumDofs() - self.skel.getJoint(0).getNumDofs())
        
        self.cur_obs = None
        self.cur_reward = 0.0
        self.cumul_reward = 0.0
        self.start_frame = 0
        
        self.cur_root_T = None
        self.cur_root_T_inv = None
        self.pd_target = None

        self.reset()
        
        self.num_obs = len(self.get_obs())
        self.num_action = len(self.get_zero_action()) * (3 if self.learning_gain else 1)
        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_obs,))
        self.action_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_action,))
        self.action_scale = 0.04 # default
        

    def loading_xml(self, metadata):
        ## XML loading
        doc = ET.ElementTree(ET.fromstring(metadata))  # ET.parse(metadata)
        root = doc.getroot()
        bvh_info = None
        for child in root:
            if child.tag == "skeleton":
                self.skel_info, self.root_name, bvh_info = saveSkeletonInfo(child.text)
                self.new_skel_info = deepcopy(self.skel_info)
                self.skel = buildFromInfo(self.skel_info, self.root_name)
                # self.skel, bvh_info = buildFromFile(child.text)
                self.target_skel = self.skel.clone()
                self.world.addSkeleton(self.skel)
            elif child.tag == "ground":
                self.ground, _ = buildFromFile(child.text)
                self.ground.setMobile(False)
                self.world.addSkeleton(self.ground)
            elif child.tag == "simHz":
                self.simulationHz = int(child.text)
            elif child.tag == "controlHz":
                self.controlHz = int(child.text)
            elif child.tag == "bvh":
                Tframe = None 
                if "firstT" in child.attrib.keys():
                    Tframe = 1 if(child.attrib["firstT"].upper() == "TRUE") else None   
                if child.text[-3:] == "bvh":
                    self.bvhs = [MyBVH(child.text, bvh_info, self.skel, Tframe)]
                else:   
                    files = os.listdir(child.text)
                    self.bvhs = (MyBVH(f, bvh_info, self.skel) for f in files if f[-3:] == "bvh")

                for _ in range(self.bvhs[self.bvh_idx].num_frames):
                    self.reward_bins.append(0)

                # print(f'Size of reward bins is {len(self.reward_bins)}')
            elif child.tag == "action_scale":
                self.action_scale = float(child.text)
            elif child.tag == "muscle":
                # self.muscle_path = child.text
                # self.loading_muscle(child.text)

                self.muscle_info = self.saveMuscleInfo(child.text)
                self.loading_muscle_info(self.muscle_info)

                self.new_muscle_info = deepcopy(self.muscle_info)

            elif child.tag == "actuator" or child.tag == "actuactor":
                self.actuator_type = child.text
                if self.actuator_type == "pd": # should be deprecated
                    self.actuator_type = "pd_ref_residual"
            elif child.tag == "kp":
                self.kp = float(child.text) * np.ones(self.skel.getNumDofs())
            elif child.tag == "kv":
                self.kv = float(child.text) * np.ones(self.skel.getNumDofs())
            elif child.tag == "learningGain":
                self.learning_gain = True if child.text.upper() == "TRUE" else False

        # if self.kp is not numpy
        if type(self.kp) != np.ndarray:
            self.kp = 300.0 * np.ones(self.skel.getNumDofs()) 
        if type(self.kv) != np.ndarray:
            self.kv = 20.0 * np.ones(self.skel.getNumDofs())
        
        self.kp[:6] = 0.0
        self.kv[:6] = 0.0

        if self.actuator_type.find("mass") != -1:
            self.muscle_nn = MuscleNN(self.muscles.getNumMuscleRelatedDofs(), len(self.get_zero_action()), self.muscles.getNumMuscles()) ## Create Dummy Default Network    

    def set_muscle_network(self, nn_config = {"sizes" : [256, 256, 256], "learningStd" : False}):
        self.muscle_nn = MuscleNN(self.muscles.getNumMuscleRelatedDofs(), len(self.get_zero_action()), self.muscles.getNumMuscles(), config = {"sizes" : nn_config["sizes"], "learningStd" : nn_config["learningStd"]})

    def loading_muscle(self, path):
        ## Open XML
        doc = ET.parse(path)
        if doc is None:
            return
        self.muscles = dart.dynamics.Muscles(self.skel)
        root = doc.getroot()
        for child in root:
            if child.tag == "Unit":
                new_waypoints = []
                for waypoint in child:
                    if waypoint.tag == "Waypoint":
                        new_waypoints.append((waypoint.attrib["body"], np.array([float(p) for p in waypoint.attrib["p"].strip().split(" ")])))
                self.muscles.addMuscle(child.attrib["name"],[float(child.attrib["f0"]), float(child.attrib["lm"]), float(child.attrib["lt"]), float(child.attrib["pen_angle"]), float(child.attrib["lmax"]), 0.0], False, new_waypoints)
        self.muscle_activation_levels = np.zeros(self.muscles.getNumMuscles())

    def loading_muscle_info(self, muscle_info):
        self.muscles = dart.dynamics.Muscles(self.skel)

        for name, muscle in muscle_info.items():
            muscle_properties = muscle['muscle_properties']
            useVelocityForce = muscle['useVelocityForce']
            waypoints_info = muscle['waypoints']

            waypoints = []
            for waypoint in waypoints_info:
                waypoints.append((waypoint['body'], waypoint['p']))

            self.muscles.addMuscle(name, muscle_properties, useVelocityForce, waypoints)

        self.muscle_activation_levels = np.zeros(self.muscles.getNumMuscles())

    def saveMuscleInfo(self, path):
        muscle_info = {}

        for skel in self.skel_info.values():
            skel['muscles'] = []

        if path is not None:
            doc = ET.parse(path)
            if doc is None:
                print("File not found")
                return None

        root = doc.getroot()
        for child in root:
            muscle = {}

            name = child.attrib['name']
            
            waypoints = []
            for waypoint in child:
                if waypoint.tag == "Waypoint":
                    body = waypoint.attrib['body']
                    p = np.array([float(p) for p in waypoint.attrib["p"].strip().split(" ")])

                    skel = self.skel_info[body]

                    if not name in skel['muscles']:
                        skel['muscles'].append(name)

                    if skel.get('stretch') != "None":
                        body_t = skel['body_t']

                        stretch = skel['stretch']
                        if stretch == "x":
                            size = skel['size'][0]
                        elif stretch == "y":
                            size = skel['size'][1]
                        elif stretch == "z":
                            size = skel['size'][2]

                        stretch_axis = skel['stretch_axis']
                        
                        ratio = np.dot(p - body_t, stretch_axis) / (size * 0.5)
                        gap = p - (body_t + stretch_axis * ratio * size * 0.5)
                        waypoints.append({
                            'body': body,
                            'p': p,
                            'ratio': ratio,
                            'gap': gap,
                        })
                    else:
                        waypoints.append({
                            'body': body,
                            'p': p,
                        })

            # muscle['f0'] = float(child.attrib['f0'])
            # muscle['lm'] = float(child.attrib['lm'])
            # muscle['lt'] = float(child.attrib['lt'])
            # muscle['pen_angle'] = float(child.attrib['pen_angle'])
            # muscle['lmax'] = float(child.attrib['lmax'])
            # muscle['type1_fraction'] = 0.0

            type1_fraction = 0.0
            muscle_properties = [float(child.attrib["f0"]), float(child.attrib["lm"]), float(child.attrib["lt"]), float(child.attrib["pen_angle"]), float(child.attrib["lmax"]), type1_fraction]

            muscle['muscle_properties'] = muscle_properties
            muscle['useVelocityForce'] = False
            muscle['waypoints'] = waypoints

            # self.muscles.addMuscle(child.attrib["name"],[float(child.attrib["f0"]), float(child.attrib["lm"]), float(child.attrib["lt"]), float(child.attrib["pen_angle"]), float(child.attrib["lmax"]), 0.0], False, waypoints)

            muscle_info[name] = muscle

        # for name, skel in self.skel_info.items():
        #     print(name, skel['muscles'])

        self.new_skel_info = deepcopy(self.skel_info)

        return muscle_info

    def get_zero_action(self):
        if True:  # if using PD servo # self.actuator_type == "pd_ref_residual" or self.actuator_type == "mass":
            return np.zeros(self.skel.getNumDofs() - self.skel.getJoint(0).getNumDofs())

    def get_root_T(self, skel):
        root_y = np.array([0, 1, 0])
        rootbody_rotation = skel.getRootBodyNode().getWorldTransform().rotation()
        while np.isnan(rootbody_rotation).any():
            self.zero_reset()
            rootbody_rotation = skel.getRootBodyNode().getWorldTransform().rotation()
        root_z = rootbody_rotation @ np.array([0, 0, 1]); 

        root_z[1] = 0.0; 
        root_z = root_z / np.linalg.norm(root_z)
        root_x = np.cross(root_y, root_z)

        root_rot = np.array([root_x, root_y, root_z]).transpose()
        root_T = np.identity(4); 
        root_T[:3, :3] = root_rot; 
        rootbody_translation = skel.getRootBodyNode().getWorldTransform().translation()
        root_T[:3,  3] = rootbody_translation; 
        root_T[ 1,  3] = 0.0

        return root_T

    def get_obs(self):
        return self.cur_obs.copy()
        
    def update_target(self, time):
        self.target_pos = self.bvhs[self.bvh_idx].getPose(time)
        pos_next = self.bvhs[self.bvh_idx].getPose(time + 1.0 / self.controlHz)
        self.target_vel = self.skel.getPositionDifferences(pos_next, self.target_pos) * self.controlHz
        self.target_skel.setPositions(self.target_pos)
        self.target_skel.setVelocities(self.target_vel)

    def update_obs(self):
        w_bn_ang_vel = 0.1

        ## Skeleton Information 

        self.cur_root_T = self.get_root_T(self.skel)
        self.cur_root_T_inv = mat_inv(self.cur_root_T)

        bn_lin_pos = []
        bn_6d_orientation = []
        for bn in self.skel.getBodyNodes():
            p = np.ones(4)
            p[:3] = bn.getCOM()
            bn_lin_pos.append(p)
            bn_6d_orientation.append(bn.getWorldTransform().rotation())
        

        bn_lin_pos = (self.cur_root_T_inv @ np.array(bn_lin_pos).transpose()).transpose()[:,:3].flatten()
        bn_lin_vel = (self.cur_root_T_inv[:3,:3] @ np.array([bn.getCOMLinearVelocity() for bn in self.skel.getBodyNodes()]).transpose()).transpose().flatten()        
        bn_6d_orientation = (self.cur_root_T_inv[:3,:3] @ np.array(bn_6d_orientation).transpose()).transpose().reshape(len(bn_6d_orientation), -1)[:,:6].flatten()
        bn_ang_vel = (self.cur_root_T_inv[:3,:3] @ np.array([w_bn_ang_vel * bn.getAngularVelocity() for bn in self.skel.getBodyNodes()]).transpose()).transpose().flatten()

        # Target        
        target_root_T = self.get_root_T(self.target_skel)
        target_root_T_inv = mat_inv(target_root_T)
        
        target_bn_pos = []
        target_6d_orientation = []

        for bn in self.target_skel.getBodyNodes():
            p = np.ones(4)
            p[:3] = bn.getCOM()
            target_bn_pos.append(p)
            target_6d_orientation.append(bn.getWorldTransform().rotation())

        target_bn_pos = (target_root_T_inv @ np.array(target_bn_pos).transpose()).transpose()[:,:3].flatten()
        target_6d_orientation = (target_root_T_inv[:3,:3] @ np.array(target_6d_orientation).transpose()).transpose().reshape(len(target_6d_orientation), -1)[:,:6].flatten()

        # Root Displacement
        cur_to_target = mat_inv(self.cur_root_T) @ target_root_T
        cur_to_target = np.concatenate([cur_to_target[:3,:3].flatten()[:6], cur_to_target[:3,3]])

        self.cur_obs = np.concatenate([bn_lin_pos , bn_lin_vel , bn_6d_orientation , bn_ang_vel , cur_to_target, target_bn_pos, target_6d_orientation], dtype=np.float32) 
    
    def zero_reset(self):
        self.target_pos = np.zeros(self.skel.getNumDofs())
        self.target_vel = np.zeros(self.skel.getNumDofs())
        self.target_skel.setPositions(self.target_pos)
        self.target_skel.setVelocities(self.target_vel)

        solver = self.world.getConstraintSolver()
        # solver.setCollisionDetector(dart.collision.BulletCollisionDetector())
        solver.setCollisionDetector(dart.collision.FCLCollisionDetector())
        # solver.setCollisionDetector(dart.collision.DARTCollisionDetector())
        solver.clearLastCollisionResult()

        self.skel.setPositions(self.target_pos)
        self.skel.setVelocities(self.target_vel)

        self.skel.clearInternalForces()
        self.skel.clearExternalForces()
        self.skel.clearConstraintImpulses()

        self.world.setTime(0)
              
        self.update_obs()
        
        if self.muscles != None:
            self.muscles.update()
            self.muscle_pos = self.muscles.getMusclePositions()

        self.step_counter = 0
        self.pd_target = np.zeros(self.skel.getNumDofs())

        return self.get_obs()
    
    def reset(self, reset_time=None):
        # dynamics reset 
        # time = 0.0 # 
        if reset_time is None:
            time = (np.random.rand() % 1.0) * self.bvhs[self.bvh_idx].bvh_time
            # epsilon = 1e-6
            # inverse_rewards = [1.0 / (reward + epsilon) for reward in self.reward_bins]
            # total_inverse = sum(inverse_rewards)
            # probabilities = [inv_reward / total_inverse for inv_reward in inverse_rewards]
            # actions = list(range(len(self.reward_bins)))
            # chosen_frame = np.random.choice(actions, p=probabilities)
            # self.start_frame = chosen_frame

            # time = (np.random.rand() % 1.0) + chosen_frame
        else:
            time = reset_time
        self.update_target(time)

        solver = self.world.getConstraintSolver()
        # solver.setCollisionDetector(dart.collision.BulletCollisionDetector())
        solver.setCollisionDetector(dart.collision.FCLCollisionDetector())
        # solver.setCollisionDetector(dart.collision.DARTCollisionDetector())
        solver.clearLastCollisionResult()

        self.skel.setPositions(self.target_pos)
        self.skel.setVelocities(self.target_vel)

        self.skel.clearInternalForces()
        self.skel.clearExternalForces()
        self.skel.clearConstraintImpulses()

        self.world.setTime(time)
              
        self.update_obs()
        
        if self.muscles != None:
            self.muscles.update()
            self.muscle_pos = self.muscles.getMusclePositions()

        self.step_counter = 0
        self.pd_target = np.zeros(self.skel.getNumDofs())

        return self.get_obs()   

    def get_reward(self):
        # Joint reward
        q_diff = self.skel.getPositionDifferences(self.skel.getPositions(), self.target_pos)
        r_q = np.exp(-20.0 * np.inner(q_diff, q_diff) / len(q_diff))

        # COM reward 
        com_diff = self.skel.getCOM() - self.target_skel.getCOM()
        r_com = np.exp(-10 * np.inner(com_diff, com_diff) / len(com_diff))

        # EE reward 
        ee_diff = np.concatenate([(self.skel.getBodyNode(ee).getCOM() - self.target_skel.getBodyNode(ee).getCOM() - com_diff) for ee in self.ees_name])
        r_ee = np.exp(-40 * np.inner(ee_diff, ee_diff) / len(ee_diff))

        w_alive = 0.05

        self.cur_reward = (w_alive + r_q * (1.0 - w_alive)) * (w_alive + r_ee * (1.0 - w_alive)) * (w_alive + r_com * (1.0 - w_alive))
        return self.cur_reward

    def step(self, action):

        self.update_target(self.world.getTime())
        pd_target = np.zeros(self.skel.getNumDofs())
        if self.actuator_type.find("ref") != -1:
            pd_target = self.target_pos.copy()
        elif self.actuator_type.find("cur") != -1:
            pd_target = self.skel.getPositions().copy()

        displacement = np.zeros(self.skel.getNumDofs())
        displacement[6:] = self.action_scale * action[:len(displacement) - 6]

        # if not self.learning_gain: ## Vanila Setting
        #     displacement[6:] = self.action_scale * action 
        # else: ## Learning Gain
        #     displacement[6:] = self.action_scale * action[:len(action)//3]
        kp = self.kp
        kv = self.kv
        if self.learning_gain:
            kp[6:] = self.kp[6:] + 0.01 * action[len(action)//3:2*len(action)//3] * self.kp[6:]
            kv[6:] = self.kv[6:] + 0.01 * action[2*len(action)//3:] * self.kv[6:]
        
        pd_target = self.skel.getPositionDifferences(pd_target, -displacement)
        
        self.pd_target = pd_target
        
        mt = None

        rand_idx = np.random.randint(0, int(self.simulationHz//self.controlHz))
        for i in range(int(self.simulationHz//self.controlHz)):
            tau = self.skel.getSPDForce(pd_target, kp, kv)
            
            if self.actuator_type.find("pd") != -1:
                self.skel.setForces(tau)
            elif self.actuator_type.find("mass") != -1:
                mt = self.muscles.getMuscleTuples()
                # mt[0]: res_JtA_reduced
                self.muscle_activation_levels = self.muscle_nn.get_activation(mt[0], tau[6:])
                self.muscles.setActivations(self.muscle_activation_levels)
                self.muscles.applyForceToBody()
            self.world.step()

            if self.muscles != None:
                self.muscles.update()

            if self.actuator_type.find("mass") != -1 and rand_idx == i:
                self.muscle_buffer[0].append(mt[0]) # reduced_JtA
                self.muscle_buffer[1].append(tau[6:] - mt[1]) # net_tau_des
                self.muscle_buffer[2].append(mt[2]) # full_JtA
        self.step_counter += 1        
                # [self.muscle_buffer[mt_idx].append(mt[mt_idx]) for mt_idx in range(len(mt))]
                # self.muscle_buffer[-1].append(tau[6:])
        
        if self.muscles != None:
            self.muscle_pos = self.muscles.getMusclePositions()
        
        self.update_obs()
        self.get_reward()
        
        info = self.get_eoe_condition()

        return self.get_obs(), self.cur_reward, info["end"] != 0, info

    def get_eoe_condition(self):
        info = {}
        self.cumul_reward += self.cur_reward
        if self.cur_reward < 0.0001:
            info["end"] = 1
            self.reward_bins[self.start_frame] = min(self.cumul_reward, self.reward_bins[self.start_frame])
            self.cumul_reward = 0
        elif self.world.getTime() > 10.0: # elif self.step_counter > 299:
            info["end"] = 3
            self.reward_bins[self.start_frame] = min(self.cumul_reward, self.reward_bins[self.start_frame])
            self.cumul_reward = 0
        else:
            info["end"] = 0
        return info
        
    def load_muscle_model_weight(self, w):
        self.muscle_nn.load_state_dict(convert_to_torch_tensor(ray.get(w)))

    def get_muscle_tuples(self, idx):
        res = np.array(self.muscle_buffer[idx])
        self.muscle_buffer[idx] = []
        return res