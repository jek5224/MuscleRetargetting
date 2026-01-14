import numpy as np

def trackball_rotation(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        return np.identity(4)
    p1 = np.array([x1, y1, project_to_sphere(x1, y1)])
    p2 = np.array([x2, y2, project_to_sphere(x2, y2)])
    axis = -np.cross(p1, p2)  # Negate the axis to reverse the direction
    angle = np.arccos(np.clip(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)), -1.0, 1.0))
    return rotation_matrix_from_axis_angle(axis, angle)

def project_to_sphere(x, y):
    d = np.sqrt(x * x + y * y)
    r = 1.0
    if d < r * np.sqrt(0.5):
        return np.sqrt(r * r - d * d)
    else:
        t = r / np.sqrt(2)
        return t * t / d

def rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c), 0],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b), 0],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c, 0],
        [0, 0, 0, 1]
    ])