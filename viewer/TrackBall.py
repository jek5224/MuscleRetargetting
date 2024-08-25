## TrackBall implementation class
import numpy as np
import quaternion
from OpenGL.GL import *

class TrackBall(object):
    def __init__(self, center = np.array([0.0, 0.0]), radius = 1.0):
        self.center = center
        self.radius = radius
        self.start_pos = np.array([0.0, 0.0, 0.0])
        self.curr_quat = np.quaternion(1.0, 0.0, 0.0, 0.0)

    def mouse_on_sphere(self, x, y):
        point_on_sphere = np.array([0.0, 0.0, 0.0])
        point_on_sphere[0] = (x - self.center[0]) / self.radius
        point_on_sphere[1] = (y - self.center[1]) / self.radius

        mag = point_on_sphere[0] * point_on_sphere[0] + point_on_sphere[1] * point_on_sphere[1]

        if mag > 1.0:
            point_on_sphere[0] *= 1.0 / np.sqrt(mag)
            point_on_sphere[1] *= 1.0 / np.sqrt(mag)
            point_on_sphere[2] = 0.0
        else:
            point_on_sphere[2] = np.sqrt(1.0 - mag)
        return point_on_sphere

    def quat_from_vector(self, _from, _to):
        q = np.zeros(4)

        q[0] = _from[0] * _to[0] + _from[1] * _to[1] + _from[2] * _to[2]
        q[1] = _from[1] * _to[2] - _from[2] * _to[1]
        q[2] = _from[2] * _to[0] - _from[0] * _to[2]
        q[3] = _from[0] * _to[1] - _from[1] * _to[0]
        
        return q

    def start_ball(self, x, y):
        self.start_pos = self.mouse_on_sphere(x, y)

    def update_ball(self, x, y):
        to_pos = self.mouse_on_sphere(x, y)
        q = self.quat_from_vector(self.start_pos, to_pos)
        new_quat = np.quaternion(q[0], q[1], q[2], q[3])

        self.start_pos = to_pos
        self.curr_quat = new_quat * self.curr_quat

    def apply_gl_roatation(self):
        res = np.zeros((4, 4))
        res[3,3] = 1.0
        res[:3, :3] = quaternion.as_rotation_matrix(self.curr_quat)
        glMultMatrixd(res.transpose())
        return

    def set_trackball(self, center, radius):
        self.center = center
        self.radius = radius
    
    def set_center(self, center):
        self.center = center
    
    def set_radius(self, radius):
        self.radius = radius
    
    def set_quaternion(self, quat):
        self.curr_quat = quat
    
    




