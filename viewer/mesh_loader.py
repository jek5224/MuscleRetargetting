# Mesh Loader for the viewer
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class MeshLoader(object):
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.faces = []
        self.texcoords = []

        self.faces_3 = []
        self.faces_4 = []

        self.new_vertices = []

        # self.joint_offset = None
        # self.axis = None

    def load(self, filename):
        # print('Loading mesh from', filename)
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    self.vertices.append(list(map(float, line[2:].split())))
                elif line.startswith('vn '):
                    self.normals.append(list(map(float, line[3:].split())))
                elif line.startswith('vt '):
                    self.texcoords.append(list(map(float, line[3:].split())))
                elif line.startswith('f '):
                    self.faces.append([list(map(int, face.split('/'))) for face in line[2:].split()])
        self.vertices = np.array(self.vertices).astype(np.float32) * 0.01
        self.normals = np.array(self.normals)
        self.texcoords = np.array(self.texcoords)

        for f in self.faces:
            if len(f) == 3:
                self.faces_3.append(f)
            elif len(f) == 4:
                self.faces_4.append(f)
            else:
                print('Invalid face:', f)

        self.faces_3 = np.array(self.faces_3) - 1
        self.faces_4 = np.array(self.faces_4) - 1

        ## Same Order
        self.vertices_3 = self.vertices[self.faces_3[:,:,0].flatten()]
        self.normals_3 = self.normals[self.faces_3[:,:,2].flatten()]
        
        self.vertices_4 = self.vertices[self.faces_4[:,:,0].flatten()]
        self.normals_4 = self.normals[self.faces_4[:,:,2].flatten()]

        self.new_vertices_3 = self.vertices_3.copy()
        self.new_vertices_4 = self.vertices_4.copy()


    def draw(self, color = np.array([0.5,0.5,0.5,1.0])):
        glPushMatrix()
        # glMultMatrixd(self.fromT.matrix().transpose())
        glColor4f(color[0], color[1], color[2], color[3])
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glVertexPointer(3, GL_FLOAT, 0, self.new_vertices_4)
        
        if len(self.normals) > 0:
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, self.normals_4)
        glDrawArrays(GL_QUADS, 0, len(self.new_vertices_4))

        if len(self.normals) > 0:
            glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()