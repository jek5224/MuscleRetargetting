## BVH Loader Class and transformation to rotation matrix 

import numpy as np
from numpy.linalg import matrix_power
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv as mat_inv
from OpenGL.GL import *
from OpenGL.GLUT import *


def inplaneXY(R):
    y = np.array([0, 1, 0])
    z = R[:3,:3] @ np.array([0, 0, 1])
    z[1] = 0
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)

    return np.array([x, y, z]).transpose()

class MySMPL():
    # bvh_file : map (joint_name : str, bvh_joint_idx : 0)
    def __init__(self, smpl_file , smpl_jn_info = None, skel = None, T_frame = None):
        np_npz = np.load(smpl_file)
        
        self.labels = np_npz['labels']
        
        self.poses = np_npz['pose_body']
        self.ori = np_npz['root_orient']
        self.trans = np_npz['trans']

        self.current_label_idx = 0

        self.skel = skel
        
        self.rot_root = R.from_euler('y', -90, degrees=True).as_matrix() @ R.from_euler('x', -90, degrees=True).as_matrix()

        ## Basic Option
        self.frame_time = 1.0 / float(np_npz['mocap_frame_rate'])
        self.num_frames = len(np_npz['markers'])
        self.markers = (self.rot_root @ np_npz['markers'].transpose(0,2,1)).transpose(0,2,1)
        
        ## Making Joint Angle's Sequence
        self.mocap_refs = np.zeros((self.num_frames, self.skel.getNumDofs()))
        self.mocap_refs[:, 3:6] = (self.rot_root @ np_npz['trans'].transpose()).transpose()     
        self.mocap_refs[:, :3] =  R.from_matrix(self.rot_root @ (R.from_rotvec(np_npz['root_orient']).as_matrix().transpose(1,2,0)).transpose(2,0,1)).as_rotvec()
        pos = self.skel.getPositions()
        self.skel.setPositions(self.mocap_refs[0])
        
        marker_labels = np_npz['labels']
        marker_tags = ['LBWT', 'RBWT', 'LFWT', 'RFWT']
        
        # Get index of marker_tags in marker_labels 
        marker_idx = [np.where(marker_labels == tag)[0][0] for tag in marker_tags]
        marker_com = np.mean(self.markers[0, marker_idx], axis=0)

        self.root_offset = marker_com - self.skel.getRootBodyNode().getTransform().translation()
        self.root_offset[1] -= 0.03

        self.skel.setPositions(pos)
        self.mocap_refs[:, 3:6] += self.root_offset

        ## Exclude Root Joint 
        for k, v in smpl_jn_info.items():
            if self.skel.getJoint(k).getNumDofs() == 3:
                self.mocap_refs[:, self.skel.getJoint(k).getIndexInSkeleton(0):self.skel.getJoint(k).getIndexInSkeleton(0) + 3] = np_npz['pose_body'][:, (v - 1) * 3 : v * 3]
            if self.skel.getJoint(k).getNumDofs() == 1:
                self.mocap_refs[:, self.skel.getJoint(k).getIndexInSkeleton(0)] = np.linalg.norm(np_npz['pose_body'][:, (v - 1) * 3 : v * 3], axis=-1)
        
        ## For Compativbility
        self.bvh_file = smpl_file
        self.bvh_time = self.frame_time * self.num_frames

        self.root_jn = self.skel.getJoint(0)

        root_0 = self.root_jn.convertToTransform(self.mocap_refs[0, 0:self.root_jn.getNumDofs()]).matrix()
        root_0[:3, :3] = inplaneXY(root_0)

        self.root_T = self.root_jn.convertToTransform(self.mocap_refs[-1, 0:self.root_jn.getNumDofs()]).matrix()
        self.root_T[:3, :3] = inplaneXY(self.root_T[:3, :3])

        self.root_T = self.root_T @ mat_inv(root_0)
        self.root_T[1,3] = 0.0

        labels = [] 
        for l in self.labels:
            labels.append(str(l))
        self.labels = labels
        

    ## Public Function
    def getLowerBoundPose(self, time): 
        iter = int(time // self.bvh_time)
        net_time = time - iter * self.bvh_time
        frame = int(net_time // self.frame_time)
        res = self.mocap_refs[frame % self.num_frames].copy()
        T = matrix_power(self.root_T, iter)
        res[:self.root_jn.getNumDofs()] = self.root_jn.convertToPositions(T @ self.root_jn.convertToTransform(res[:self.root_jn.getNumDofs()]).matrix()) 
        return res, float(frame)/self.num_frames
    
    def getPose(self, t):
        pos, phase = self.getLowerBoundPose(t)
        vel = self.skel.getPositionDifferences(self.getLowerBoundPose(t + self.frame_time)[0], pos) * (1.0 / self.frame_time)
        return pos, vel, phase

    def getSMPL(self, t):
        t = t // self.frame_time
        frame_idx = int(t % self.num_frames)
        return self.poses[frame_idx], self.ori[frame_idx], self.trans[frame_idx]

    def draw(self, t):
        frame_idx = int((t // self.frame_time) % self.num_frames)
        
        for m in self.markers[frame_idx]:    
            glPushMatrix()
            glTranslatef(m[0], m[1], m[2])
            glutSolidSphere(0.01, 20, 20)
            glPopMatrix()

        glColor3f(0.0,1.0,0.0)
        glPushMatrix()
        glTranslatef(self.markers[frame_idx][self.current_label_idx][0], self.markers[frame_idx][self.current_label_idx][1], self.markers[frame_idx][self.current_label_idx][2])
        glutSolidSphere(0.05, 20, 20)
        glPopMatrix()

    def set_current_label_idx(self, idx):
        self.current_label_idx = idx

if __name__ == '__main__':
    bvh_file = "data/motion/smpl/01_11_stageii.npz"
    import time
    start = time.perf_counter()
    bvh_loader = MySMPL(bvh_file)
    end = time.perf_counter() - start
    print("Time : ", end)

    