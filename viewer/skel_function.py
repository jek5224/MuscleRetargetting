from skeleton_section import SKEL_dart_info

from skel.skel_model import SKEL
from skel.alignment.aligner import SkelFitter

# from data.face_info import male_right_faces, female_right_faces
from data.skel_info import male_skin_right_faces as male_right_faces
from data.skel_info import male_skin_left_faces as male_left_faces
from data.skel_info import female_skin_right_faces as female_right_faces
from data.skel_info import female_skin_left_faces as female_left_faces
from data.skel_info import skel_joint_edges
from skel.kin_skel import skel_joints_name, pose_param_names, pose_limits

class line_muscle:
    def __init__(self,
                 origin,
                 insertion,
                 waypoints,
                 waypoint_bodies):
        
        self.origin = origin['p']
        self.origin_mesh_index = origin['mesh_index'] 
        self.origin_face = origin['face']
        self.origin_barycentric = origin['barycentric']
        for info in SKEL_dart_info:
            if self.origin_mesh_index in info[0]:
                self.origin_body = info[1]
                break

        self.insertion = insertion['p']
        self.insertion_mesh_index = insertion['mesh_index']
        self.insertion_face = insertion['face']
        self.insertion_barycentric = insertion['barycentric']
        for info in SKEL_dart_info:
            if self.insertion_mesh_index in info[0]:
                self.insertion_body = info[1]
                break

        self.waypoints = waypoints
        self.waypoint_bodies = [self.insertion_body] * len(waypoints)

        print("Origin:", origin)
        print("Insertion:", insertion)
        print("Waypoints:", waypoints)

    def draw(self, color=np.array([0.5, 0, 0, 1])):
        points = [self.origin] + self.waypoints + [self.insertion]
        glColor4d(color[0], color[1], color[2], color[3])
        glBegin(GL_LINE_STRIP)
        for p in points:
            glVertex3f(p[0], p[1], p[2])
        glEnd()

        glBegin(GL_POINTS)
        for p in points:
            glVertex3f(p[0], p[1], p[2])
        glEnd()

class SKEL_muscle:
    def __init__(self,
                 name,
                 lines,
                 f0=1000.0,
                 lm=1.2,
                 lt=0.2,
                 pen_angle=0.0,
                 lmax=-0.1,
                 symmetry=False):
        self.name = name
        self.lines = lines
        self.f0 = f0
        self.lm = lm
        self.lt = lt
        self.pen_angle = pen_angle
        self.lmax = lmax

        self.symmetry = symmetry

    def draw(self, color=np.array([0.5, 0, 0, 1])):
        for line in self.lines:
            line.draw(color=color)