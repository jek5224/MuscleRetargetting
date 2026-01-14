import numpy as np

smpl_links = [
    (0, 1),
        (1, 4),
            (4, 7),
                (7, 10),    # Right Leg
    (0, 2),
        (2, 5),
            (5, 8),
                (8, 11),    # Left Leg
    (0, 3),
        (3, 6),
            (6, 9),
                (9, 12),
                    (12, 15),   # Neck
                (9, 13),
                    (13, 16),
                        (16, 18),
                            (18, 20),
                                (20, 22),   # Right Arm
                (9, 14),
                    (14, 17),
                        (17, 19),
                            (19, 21),
                                (21, 23)    # Left Arm
]

skel_joint_names = [
    "Pelvis",
    "FemurR",
    "TibiaR",
    "TalusR",
    "FootPinkyR",
    "FootThumbR",
    "FemurL",
    "TibiaL",
    "TalusL",
    "FootPinkyL",
    "FootThumbL",
    "Spine",
    "Torso",
    "Neck",
    "Head",
    "ShoulderR",
    "ArmR",
    "ForeArmR",
    "HandR",
    "ShoulderL",
    "ArmL",
    "ForeArmL",
    "HandL"
]

skel_joint_ts = np.array([
[0.0000, 0.9809, -0.0116], # Pelvis
[-0.0903, 0.9337, -0.0116], # FemurR
[-0.0995, 0.5387, -0.0103], # TibiaR
[-0.0800, 0.0776, -0.0419], # TalusR
[-0.1215, 0.0116, 0.0494], # FootPinkyR
[-0.0756, 0.0118, 0.0676], # FootThumbR
[0.0903, 0.9337, -0.0116], # FemurL
[0.0995, 0.5387, -0.0103], # TibiaL
[0.0800, 0.0776, -0.0419], # TalusL
[0.1215, 0.0116, 0.0494], # FootPinkyL
[0.0756, 0.0118, 0.0676], # FootThumbL
[0.0000, 1.0675, -0.0116], # Spine
[0.0000, 1.1761, -0.0116], # Torso
[0.0000, 1.4844, -0.0116], # Neck
[0.0000, 1.5652, -0.0116], # Head
[-0.0147, 1.4535, -0.0353], # ShoulderR
[-0.1995, 1.4350, -0.0353], # ArmR
[-0.5234, 1.4607, -0.0105], # ForeArmR
[-0.8102, 1.4694, 0.0194], # HandR
[0.0147, 1.4535, -0.0353], # ShoulderL
[0.1995, 1.4350, -0.0353], # ArmL
[0.5234, 1.4607, -0.0105], # ForeArmL
[0.8102, 1.4694, 0.0194], # HandL
])
skel_body_ts = np.array([
[0.0000, 0.9464, -0.0112], # Pelvis
[-0.0925, 0.6986, -0.0219], # FemurR
[-0.0895, 0.2912, -0.0329], # TibiaR
[-0.0893, 0.0389, -0.0233], # TalusR
[-0.1200, 0.0298, 0.0781], # FootPinkyR
[-0.0738, 0.0297, 0.0905], # FootThumbR
[0.0925, 0.6986, -0.0219], # FemurL
[0.0895, 0.2912, -0.0329], # TibiaL
[0.0893, 0.0389, -0.0233], # TalusL
[0.1200, 0.0298, 0.0781], # FootPinkyL
[0.0738, 0.0297, 0.0905], # FootThumbL
[0.0000, 1.0810, -0.0112], # Spine
[0.0000, 1.2766, -0.0112], # Torso
[0.0000, 1.4759, -0.0112], # Neck
[0.0000, 1.5945, -0.0083], # Head
[-0.0946, 1.4129, -0.0341], # ShoulderR
[-0.3452, 1.4011, -0.0130], # ArmR
[-0.6439, 1.4182, -0.0057], # ForeArmR
[-0.8503, 1.4125, 0.0304], # HandR
[0.0946, 1.4129, -0.0341], # ShoulderL
[0.3452, 1.4011, -0.0130], # ArmL
[0.6439, 1.4182, -0.0057], # ForeArmL
[0.8503, 1.4125, 0.0304], # HandL
])

SMPL_joint_ts = np.array([
[0.0000, 0.9464, -0.0112], # Pelvis
[0.0695, 0.8550, -0.0180], # L_Hip
[-0.0677, 0.8559, -0.0155], # R_Hip
[-0.0025, 1.0553, -0.0379], # Spine_01
[0.1038, 0.4798, -0.0225], # L_Knee
[-0.1060, 0.4733, -0.0244], # R_Knee
[0.0030, 1.1905, -0.0368], # Spine_02
[0.0902, 0.0818, -0.0662], # L_Ankle
[-0.0902, 0.0749, -0.0667], # R_Ankle
[0.0044, 1.2434, -0.0114], # Spine_03
[0.1166, 0.0260, 0.0531], # L_Toe
[-0.1156, 0.0267, 0.0567], # R_Toe
[0.0016, 1.4573, -0.0542], # Neck
[0.0833, 1.3652, -0.0455], # L_Collar
[-0.0773, 1.3623, -0.0500], # R_Collar
[0.0068, 1.5223, -0.0029], # Head
[0.1742, 1.3957, -0.0543], # L_Shoulder
[-0.1734, 1.3948, -0.0591], # R_Shoulder
[0.4338, 1.3829, -0.0818], # L_Elbow
[-0.4271, 1.3815, -0.0805], # R_Elbow
[0.6831, 1.3919, -0.0830], # L_Wrist
[-0.6824, 1.3893, -0.0861], # R_Wrist
[0.7671, 1.3837, -0.0979], # L_Palm
[-0.7670, 1.3831, -0.0964], # R_Palm
])
SMPL_joint_names = [
    "Pelvis", 
    "L_Hip", 
    "R_Hip", 
    "Spine_01", 
    "L_Knee", 
    "R_Knee", 
    "Spine_02", 
    "L_Ankle", 
    "R_Ankle", 
    "Spine_03", 
    "L_Toe", 
    "R_Toe",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Palm",
    "R_Palm"
]

# print(skel_joint_ts)
# print(skel_body_ts)
# print(SMPL_joint_ts)

# (Skel, SMPL)
joint_matches = [
    ("Pelvis", "Pelvis"),
    ("FemurR", "R_Hip"),
    ("TibiaR", "R_Knee"),
    ("TalusR", "R_Ankle"),
    # ("FootPinkyR", "R_Toe"),
    # ("FootThumbR", "R_Toe"),
    ("FemurL", "L_Hip"),
    ("TibiaL", "L_Knee"),
    ("TalusL", "L_Ankle"),
    # ("FootPinkyL", "L_Toe"),
    # ("FootThumbL", "L_Toe"),
    ("Spine", "Spine_01"),
    ("Torso", "Spine_02"),
    ("Neck", "Neck"),
    ("Head", "Head"),
    # ("ShoulderR", "R_Collar"),
    ("ArmR", "R_Shoulder"),
    ("ForeArmR", "R_Elbow"),
    ("HandR", "R_Wrist"),
    # ("ShoulderL", "L_Collar"),
    ("ArmL", "L_Shoulder"),
    ("ForeArmL", "L_Elbow"),
    ("HandL", "L_Wrist")
]

joint_match_indices = []
for joint_match in joint_matches:
    joint_match_indices.append((skel_joint_names.index(joint_match[0]), SMPL_joint_names.index(joint_match[1])))


