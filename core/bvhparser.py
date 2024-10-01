## BVH Loader Class and transformation to rotation matrix 

import numpy as np
from numpy.linalg import matrix_power
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv as mat_inv

def inplaneXY(R):
    y = np.array([0, 1, 0])
    z = R[:3,:3] @ np.array([0, 0, 1])
    z[1] = 0
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)

    return np.array([x, y, z]).transpose()

class MyBVH():
    # bvh_file : map ( joint_name : str , bvh_joint_name : str )
    def __init__(self, bvh_file , bvh_info = None, skel = None, T_frame = None):

        self.bvh_file = bvh_file
        self.bvh_info = bvh_info

        self.joint_names = []
        self.parent_joint_name = {}
        self.joint_offsets = {} ## Not used here
        self.joint_channels = {}
        
        self.num_frames = 0
        self.frame_time = 1.0 / 30.0
        self.frame_size = 0
        self.frames_raw = []
        self.joint_names_to_index_in_frames = {}
        
        self.mocap_refs_mat = {}
        self.Tpose_mat = {}
        self.T_frame = T_frame
        self.root_T = np.identity(4)
        self.root_jn = None
        self.skel = skel
        self.mocap_refs = None
        
        self.bvh_time = None
        self.load_bvh(self.bvh_file)
        

    ## BVH function
    def load_bvh(self, bvh_file):    
        ## Loading Raw Data
        motion_reading = False; cur_parent = None; joint_name = None
        with open(bvh_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == "MOTION": ## Motion
                    motion_reading = True
                    for channels in self.joint_channels:
                        self.frame_size += len(channels)
                elif motion_reading: ## Motion Reading
                    if "Frames:" in line:
                        self.num_frames = int(line[len("Frames: "):].strip()) 
                    elif "Frame Time:" in line:
                        self.frame_time = float(line[len("Frame Time:"):].strip())
                    else:
                        self.frames_raw.append(np.array(line.split(" ")).astype(np.float32)) 
                else: 
                    if line == "HEIRARCHY": ## Hierarchy
                        cur_parent = None
                    if not motion_reading:
                        if "JOINT" in line or "ROOT" in line or "End" in line: 
                            cur_parent = joint_name
                            joint_name = line.split(" ")[1]
                            if "End" in line:
                                joint_name = "End_" + cur_parent
                                self.joint_channels[joint_name] = []
                            self.joint_names.append(joint_name)
                            self.parent_joint_name[joint_name] = cur_parent
                        elif "OFFSET" in line:
                            offset = line.split(" ")[1:]
                            offset = np.array([float(x) for x in offset])
                            self.joint_offsets[joint_name] = offset
                        elif "CHANNELS" in line:
                            channels = line.lower().split(" ")[2:]
                            self.joint_channels[joint_name] = channels
                        elif "}" in line:
                            joint_name = cur_parent
                            if cur_parent != None:    
                                cur_parent = self.parent_joint_name[cur_parent]
                            
        self.bvh_time = self.frame_time * self.num_frames

        ## Check Number of Frames
        if self.num_frames != len(self.frames_raw):
            print("Number of frames is not correct")
            return
        self.frames_raw = np.array(self.frames_raw)
        self.mocap_refs = np.zeros([self.num_frames, self.skel.getNumDofs()])
        
        ## Loading Finish
        self.bvh2rot()

    def bvh2rot(self):
        idx = 0
        for i in range(len(self.joint_names)):
            self.joint_names_to_index_in_frames[self.joint_names[i]] = idx
            idx += len(self.joint_channels[self.joint_names[i]])
        
        ## Set Transformation Matrix 
        for jn_name in self.joint_names:
            if "End" in jn_name:
                continue
            self.mocap_refs_mat[jn_name] = []
            joint_chs = self.joint_channels[jn_name]
            if len(joint_chs) == 6:
                xpos = joint_chs.index("xposition") # [ch_n for ch_n in joint_chs if ch_n.upper() == "XPOSITION"][0]
                ypos = joint_chs.index("yposition") # [ch_n for ch_n in joint_chs if ch_n.upper() == "YPOSITION"][0]
                zpos = joint_chs.index("zposition") # [ch_n for ch_n in joint_chs if ch_n.upper() == "ZPOSITION"][0]
                
                rots = [ch_idx for ch_idx in range(len(joint_chs)) if joint_chs[ch_idx].lower()[1:] == "rotation"]
                euler_order = "".join([ch_n[0].lower() for ch_n in joint_chs if ch_n.lower()[1:] == "rotation"])
                pos_idx = self.joint_names_to_index_in_frames[jn_name] + np.array([xpos, ypos, zpos])
                rot_idx = self.joint_names_to_index_in_frames[jn_name] + np.array(rots)
                
                pos = self.frames_raw[:, pos_idx] * 0.01
                rot = self.frames_raw[:, rot_idx]
                
                ## Using
                self.mocap_refs_mat[jn_name] = np.tile(np.identity(4), (self.num_frames,1,1))
                
                self.mocap_refs_mat[jn_name][:, :3, :3] = R.from_euler(euler_order.upper(), rot, degrees=True).as_matrix()
                self.mocap_refs_mat[jn_name][:, 0:3, 3] = pos

                self.Tpose_mat[jn_name] = self.mocap_refs_mat[jn_name][0 if self.T_frame == None else self.T_frame]
            elif len(joint_chs) == 3:
                rots = [ch_idx for ch_idx in range(len(joint_chs)) if joint_chs[ch_idx].lower()[1:] == "rotation"]
                euler_order = "".join([ch_n[0].lower() for ch_n in joint_chs if ch_n.lower()[1:] == "rotation"])
                rot_idx = self.joint_names_to_index_in_frames[jn_name] + np.array(rots)               
                rot = self.frames_raw[:, rot_idx]

                self.mocap_refs_mat[jn_name] = self.mocap_refs_mat[self.parent_joint_name[jn_name]][:,:3,:3] @ (R.from_euler(euler_order.upper(), rot, degrees=True).as_matrix())
                
                if self.T_frame != None:
                    self.Tpose_mat[jn_name] = self.mocap_refs_mat[jn_name][self.T_frame]

        self.bvh2sim()

    def bvh2sim(self): ## Change BVH motion to Skeleton motion
        for skel_jn_i in range(self.skel.getNumJoints()):
            skel_jn = self.skel.getJoint(skel_jn_i)
            if skel_jn.getName() in self.bvh_info.keys() and self.bvh_info[skel_jn.getName()] in self.mocap_refs_mat.keys():
                T = self.mocap_refs_mat[self.bvh_info[skel_jn.getName()]].copy()
                if T[0].shape == (4,4):
                    T[:,:3,3] -= self.Tpose_mat[self.bvh_info[skel_jn.getName()]][:3,3]
                    if self.T_frame != None:
                        T[:,:3,:3] = T[:,:3,:3] @ self.Tpose_mat[self.bvh_info[skel_jn.getName()]][:3,:3].T
                else:
                    if self.T_frame != None:
                        T = T @ self.Tpose_mat[self.bvh_info[skel_jn.getName()]].T

                T_parent = np.tile(np.identity(4), (self.num_frames,1,1))
                
                if self.parent_joint_name[self.bvh_info[skel_jn.getName()]] != None:
                    T_parent = self.mocap_refs_mat[self.parent_joint_name[self.bvh_info[skel_jn.getName()]]][:,:3,:3]
                    if self.T_frame != None:
                        T_parent = T_parent @ self.Tpose_mat[self.parent_joint_name[self.bvh_info[skel_jn.getName()]]][:3,:3].transpose()
                
                T_net = T_parent.transpose((0,2,1)) @ T

                if skel_jn.getNumDofs() == 6:
                    self.mocap_refs[:, skel_jn.getIndexInSkeleton(0):skel_jn.getIndexInSkeleton(0)+3] = (R.from_matrix(T_net[:,:3,:3]).as_rotvec())
                    self.mocap_refs[:, skel_jn.getIndexInSkeleton(0)+3:skel_jn.getIndexInSkeleton(0)+6] = T_net[:, :3, 3]
                elif skel_jn.getNumDofs() == 3:    
                    self.mocap_refs[:, skel_jn.getIndexInSkeleton(0):skel_jn.getIndexInSkeleton(0)+skel_jn.getNumDofs()] = (R.from_matrix(T_net[:,:3,:3]).as_rotvec())
                    # break
                elif skel_jn.getNumDofs() == 1:
                    self.mocap_refs[:, skel_jn.getIndexInSkeleton(0):skel_jn.getIndexInSkeleton(0)+skel_jn.getNumDofs()] = np.array([np.linalg.norm(R.from_matrix(T_net).as_rotvec(), axis=1)]).T


        self.root_jn = self.skel.getJoint(0)
        root_0 = self.root_jn.convertToTransform(self.mocap_refs[0,  0:self.root_jn.getNumDofs()]).matrix()
        root_0[:3,:3] = inplaneXY(root_0)
        
        self.root_T = self.root_jn.convertToTransform(self.mocap_refs[-1,  0:self.root_jn.getNumDofs()]).matrix()
        self.root_T[:3,:3] = inplaneXY(self.root_T)

        self.root_T = self.root_T @ mat_inv(root_0)
        self.root_T[1,3] = 0.0        
    
    def getLowerBoundPose(self, time): 
        iter = time // self.bvh_time
        net_time = time - self.bvh_time * iter
        frame = int(net_time / self.frame_time)

        res = self.mocap_refs[frame].copy()
        T = matrix_power(self.root_T, int(iter))
        res[:self.root_jn.getNumDofs()] = self.root_jn.convertToPositions(T @ self.root_jn.convertToTransform(res[:self.root_jn.getNumDofs()]).matrix())
        return res
    
    def getPose(self, time):
        return self.getLowerBoundPose(time)
        
        cur_pose = self.getLowerBoundPose(time)
        next_pose = self.getLowerBoundPose(time + self.mocap.frame_time)
        slerp_pose = np.zeros(cur_pose.shape)
        alpha = (time - ((time // self.mocap.frame_time) * self.mocap.frame_time)) / self.mocap.frame_time
        for i in range(self.skel.getNumJoints()):
            jn = self.skel.getJoint(i)
            if jn.getNumDofs() == 1:
                slerp_pose[jn.getIndexInSkeleton(0)] = cur_pose[jn.getIndexInSkeleton(0)] * (1.0 - alpha) + next_pose[jn.getIndexInSkeleton(0)] * alpha
            elif jn.getNumDofs() == 3:
                q1 = quaternion.from_rotation_vector(cur_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+3])
                q2 = quaternion.from_rotation_vector(next_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+3])
                slerp_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+3] = quaternion.as_rotation_vector(quaternion.slerp(q1, q2, 0.0, 1.0, alpha))
            elif jn.getNumDofs() == 6:
                q1 = quaternion.from_rotation_vector(cur_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+3])
                q2 = quaternion.from_rotation_vector(next_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+3])
                slerp_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+3] = quaternion.as_rotation_vector(quaternion.slerp(q1, q2, 0.0, 1.0, alpha))
                slerp_pose[jn.getIndexInSkeleton(3):jn.getIndexInSkeleton(3)+3] = cur_pose[jn.getIndexInSkeleton(3):jn.getIndexInSkeleton(3)+3] * (1.0 - alpha) + next_pose[jn.getIndexInSkeleton(3):jn.getIndexInSkeleton(3)+3] * alpha
                
        return slerp_pose
    
if __name__ == '__main__':
    bvh_file = "data/motion/lafan_walk1_subject1.bvh"
    import time
    start = time.perf_counter()
    bvh_loader = BVHLoader(bvh_file)
    end = time.perf_counter() - start
    print("Time : ", end)

    