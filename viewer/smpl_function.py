if use_smpl:
    from models.smpl import SMPL

    smpl_joint_names = [
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