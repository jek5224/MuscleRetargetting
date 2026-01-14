import numpy as np
SKEL_dart_info = [
                    ([26, 27, 71, 72, 73], "Pelvis", "None", ["Mid", [71, 73]], False, False, None, None, "Free"),
                    
                    # Massive Vertebrae version
                    (range(21, 26), "Lumbar", "Pelvis", ["Mid", [26, 27, 71, 72, 73]], True, False, (-2/3*np.pi/4, -2/3*np.pi/4, -2/3*np.pi/4), (2/3*np.pi/4, 2/3*np.pi/4, 2/3*np.pi/4), "Ball"),
                    # (range(21, 26), "Lumbar", "Pelvis", ["Joint", 0], True, False, (-2/3*np.pi/4, -2/3*np.pi/4, -2/3*np.pi/4), (2/3*np.pi/4, 2/3*np.pi/4, 2/3*np.pi/4), "Ball"),
                    (range(9, 21), "Thorax", "Lumbar", ["Joint", 11], True, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    (range(2, 9), "Cervix", "Thorax", ["Joint", 12], True, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    ([0, 1], "Skull", "Cervix", ["Joint", 13], False, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    ([28, 29, 30], "Sternum", "Thorax", ["Mid", [10]], True, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),
                    (range(31, 51), "Left Ribs", "Sternum", ["Mid", [29]], False, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),
                    (range(51, 71), "Right Ribs", "Sternum", ["Mid", [29]], False, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),

                    # # All Vertebrae version
                    # ([25], "L5", "Pelvis", ["Mid", [26, 27, 71, 72, 73]], False, False, None, None, "Ball"),
                    # ([24], "L4", "L5", ["Face", list(range(37594, 39324)) + list(range(33930, 35622)), (37594, "L4")], False, False, None, None, "Ball"),
                    # ([23], "L3", "L4", ["Face", list(range(40958, 42838)) + list(range(37594, 39324)), (40958, "L3")], False, False, None, None, "Ball"),
                    # ([22], "L2", "L3", ["Face", list(range(39324, 40958)) + list(range(40958, 42838)), (39324, "L2")], False, False, None, None, "Ball"),
                    # ([21], "L1", "L2", ["Face", list(range(35622, 37594)) + list(range(39324, 40958)), (35622, "L1")], False, False, None, None, "Ball"),

                    # ([20], "T12", "L1", ["Face", list(range(50178, 51156)) + list(range(35622, 37594)), (50178, "T12")], False, False, None, None, "Ball"),
                    # ([19], "T11", "T12", ["Face", list(range(43662, 44616)) + list(range(50178, 51156)), (43662, "T11")], False, False, None, None, "Ball"),
                    # ([18], "T10", "T11", ["Face", list(range(48510, 50178)) + list(range(43662, 44616)), (48510, "T10")], False, False, None, None, "Ball"),
                    # ([17], "T9", "T10", ["Face", list(range(44616, 46208)) + list(range(48510, 50178)), (44616, "T9")], False, False, None, None, "Ball"),
                    # ([16], "T8", "T9", ["Face", list(range(42838, 43662)) + list(range(44616, 46208)), (42838, "T8")], False, False, None, None, "Ball"),
                    # ([15], "T7", "T8", ["Face", list(range(46208, 47724)) + list(range(42838, 43662)), (46208, "T7")], False, False, None, None, "Ball"),
                    # ([14], "T6", "T7", ["Face", list(range(47724, 48510)) + list(range(46208, 47724)), (47724, "T6")], False, False, None, None, "Ball"),
                    # ([13], "T5", "T6", ["Face", list(range(51156, 52592)) + list(range(47724, 48510)), (51156, "T5")], False, False, None, None, "Ball"),
                    # ([12], "T4", "T5", ["Face", list(range(53428, 54706)) + list(range(51156, 52592)), (53428, "T4")], False, False, None, None, "Ball"),
                    # ([11], "T3", "T4", ["Face", list(range(55902, 57054)) + list(range(53428, 54706)), (55902, "T3")], False, False, None, None, "Ball"),
                    # ([10], "T2", "T3", ["Face", list(range(54706, 55902)) + list(range(55902, 57054)), (54706, "T2")], False, False, None, None, "Ball"),
                    # ([9], "T1", "T2", ["Face", list(range(52592, 53428)) + list(range(54706, 55902)), (52592, "T1")], False, False, None, None, "Ball"),

                    # ([8], "C7", "T1", ["Face", list(range(92883, 93767)) + list(range(52592, 53428)), (92883, "C6")], False, False, None, None, "Ball"),
                    # ([7], "C6", "C7", ["Face", list(range(93767, 94777)) + list(range(92883, 93767)), (93767, "C6")], False, False, None, None, "Ball"),
                    # ([6], "C5", "C6", ["Face", list(range(91333, 92087)) + list(range(93767, 94777)), (91333, "C5")], False, False, None, None, "Ball"),
                    # ([5], "C4", "C5", ["Face", list(range(92087, 92883)) + list(range(91333, 92087)), (92087, "C4")], False, False, None, None, "Ball"),
                    # ([4], "C3", "C4", ["Face", list(range(94777, 95763)) + list(range(92087, 92883)), (94777, "C3")], False, False, None, None, "Ball"),
                    # ([3], "C2", "C3", ["Face", list(range(89521, 91333)) + list(range(94777, 95763)), (89521, "Axis, C2")], False, False, None, None, "Ball"),
                    # ([2], "C1", "C2", ["Face", range(88733, 91333), (88733, "Atlas, C1")], False, False, None, None, "Ball"),
                    # ([0, 1], "Skull", "C1", ["Joint", 13], False, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    # ([28, 29, 30], "Sternum", "T2", ["Mid", [10]], True, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),
                    # (range(31, 51), "Left Ribs", "Sternum", ["Mid", [29]], False, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),
                    # (range(51, 71), "Right Ribs", "Sternum", ["Mid", [29]], False, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),

                    
                    #   (range(28, 71), "Thoracic Cage", "T1", None, True),

                    ([74], "Femur_R", "Pelvis", ["Joint", 1], True, False, None, None, "Ball"),
                    ([75, 76], "Tibia_R", "Femur_R", ["Joint", 2], True, False, (0), (3/4*np.pi), "Revolute"),
                    ([77], "Talus_R", "Tibia_R", ["Joint", 3], True, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    (range(78, 89), "Toe_Metacarpals_R", "Talus_R", ["Joint", 4], False, True, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    (range(89, 103), "Toe_Phalanges_R", "Toe_Metacarpals_R", ["Joint", 5], False, True, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),

                    #   (range(78, 89), "Toe_Metacarpals_R", "Talus_R", ["Joint", 4], False, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    #   (range(89, 103), "Toe_Phalanges_R", "Toe_Metacarpals_R", ["Joint", 5], False, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                        
                    ([105], "Femur_L", "Pelvis", ["Joint", 6], True, False, None, None, "Ball"),
                    ([106, 107], "Tibia_L", "Femur_L", ["Joint", 7], True, False, (0), (3/4*np.pi), "Revolute"),
                    ([108], "Talus_L", "Tibia_L", ["Joint", 8], True, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    (range(109, 120), "Toe_Metacarpals_L", "Talus_L", ["Joint", 9], False, True, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    (range(120, 134), "Toe_Phalanges_L", "Toe_Metacarpals_L", ["Joint", 10], False, True, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),

                    #   (range(109, 120), "Toe_Metacarpals_L", "Talus_L", ["Joint", 9], False, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),
                    #   (range(120, 134), "Toe_Phalanges_L", "Toe_Metacarpals_L", ["Joint", 10], False, False, (-np.pi/4, -np.pi/4, -np.pi/4), (np.pi/4, np.pi/4, np.pi/4), "Ball"),

                    # [69477, 69478] / 69166, "Manubrium" / [101422, 100837] for Right Clavicle
                    # [95783, 95784, 95776, 95775] / 95763, "Right Scapula" / [180303] for Right Acromion

                    ([], "Clavicle_R", "Sternum", ["Face", [69477, 69478], (69166, "Manubrium")], False, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),
                    ([136], "Scapula_R", "Clavicle_R", ["Face", [95783, 95784, 95776, 95775], (95763, "Right Scapula")], True, False, None, None, "Ball"),
                    # ([136], "Scapula_R", "Sternum", ["Joint", 14], True, False, None, None, "Ball"),
                    ([137], "Humerus_R", "Scapula_R", ["Joint", 15], True, False, None, None, "Ball"),
                    ([138], "Ulna_R", "Humerus_R", ["Joint", 16], True, False, (-3/4*np.pi/2), (3/4*np.pi/2), "Revolute"),
                    ([139], "Radius_R", "Ulna_R", ["Joint", 17], True, False, (0), (3/4*np.pi), "Revolute"),

                    # [102941, 102942] / 102607, "Right 1st Finger Metacarpal" / [188089, 188102] for Thumb_R Metacarpal
                    # [107607, 107608] / 107575, "Right Thumb Proximal Phalanx" / [202952, 202938] Thumb_R Proximal
                    # [102011, 102012] / 101803, "Right Thumb Distal Phalanx" / [185684, 185672] Thumb_R Distal

                    # [106768, 106769] / 106511, "Right 2nd Finger Proximal Phalanx" / [199832, 199837] for 2nd Finger_R
                    # [105905, 105906] / 105735, "Right 2nd Finger Middle Phalanx" / [197420, 197450] 2nd Finger_R Middle
                    # [100943, 100944] / 100915, "Right 2nd Finger Distal Phalanx" / [182960, 182946] 2nd Finger_R Distal

                    # [107058, 107059] / 107011, "Right 3rd Finger Proximal Phalanx" / [201259, 201290] for 3rd Finger_R
                    # [106305, 106306] / 106133, "Right 3rd Finger Middle Phalanx" / [198587, 198626] 3rd Finger_R Middle
                    # [101400, 101402] / 101365, "Right 3rd Finger Distal Phalanx" / [184292, 184304] 3rd Finger_R Distal

                    # [107322, 107323] / 107299, "Right 4th Finger Proximal Phalanx" / [202092, 202097] for 4th Finger_R
                    # [106343, 106350] / 106321, "Right 4th Finger Middle Phalanx" / [199160] 4th Finger_R Middle
                    # [101789, 101790] / 101579, "Right 4th Finger Distal Phalanx" / [184982, 184991] 4th Finger_R Distal

                    # [106843, 106844] / 106791, "Right 5th Finger Proximal Phalanx" / [200629, 200652] for 5th Finger_R
                    # [105962, 105976] / 105931, "Right 5th Finger Middle Phalanx" / [198014] 5th Finger_R Middle
                    # [101182, 101183] / 101121, "Right 5th Finger Distal Phalanx" / [183653, 183666] 5th Finger_R Distal

                    # # One Hand
                    # (range(140, 167), "Hand_R", "Radius_R", ["Joint", 18], False, True, (0, -np.pi/4, -np.pi/2), (0, np.pi/4, np.pi/2), "Ball"),

                    # One box Fingers
                    (list(range(140, 148)) + [149, 150, 151, 152], "Carpals_R", "Radius_R", ["Joint", 18], False, True, (0, -np.pi/4, -np.pi/2), (0, np.pi/4, np.pi/2), "Ball"),
                    ([148, 153, 154], "Finger_1st_R", "Carpals_R", ["Face", [102941, 102942], (102607, "Right 1st Finger Metacarpal")], True, False, None, None, "Ball"),
                    ([155, 156, 157], "Finger_2nd_R", "Carpals_R", ["Face", [106768, 106769], (106511, "Right 2nd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    ([158, 159, 160], "Finger_3rd_R", "Carpals_R", ["Face", [107058, 107059], (107011, "Right 3rd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    ([161, 162, 163], "Finger_4th_R", "Carpals_R", ["Face", [107322, 107323], (107299, "Right 4th Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    ([164, 165, 166], "Finger_5th_R", "Carpals_R", ["Face", [106843, 106844], (106791, "Right 5th Finger Proximal Phalanx")], True, False, None, None, "Ball"),

                    # # Multi box Fingers
                    # (list(range(140, 148)) + [149, 150, 151, 152], "Carpals_R", "Radius_R", ["Joint", 18], False, True, (0, -np.pi/4, -np.pi/2), (0, np.pi/4, np.pi/2), "Ball"),
                    
                    # ([148], "Finger_1st_Metacarpal_R", "Carpals_R", ["Face", [102941, 102942], (102607, "Right 1st Finger Metacarpal")], True, False, None, None, "Ball"),
                    # ([153], "Finger_1st_Proximal_R", "Finger_1st_Metacarpal_R", ["Face", [107607, 107608], (107575, "Right Thumb Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([154], "Finger_1st_Distal_R", "Finger_1st_Proximal_R", ["Face", [102011, 102012], (101803, "Right Thumb Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([155], "Finger_2nd_Proximal_R", "Carpals_R", ["Face", [106768, 106769], (106511, "Right 2nd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([156], "Finger_2nd_Middle_R", "Finger_2nd_Proximal_R", ["Face", [105905, 105906], (105735, "Right 2nd Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([157], "Finger_2nd_Distal_R", "Finger_2nd_Middle_R", ["Face", [100943, 100944], (100915, "Right 2nd Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([158], "Finger_3rd_Proximal_R", "Carpals_R", ["Face", [107058, 107059], (107011, "Right 3rd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([159], "Finger_3rd_Middle_R", "Finger_3rd_Proximal_R", ["Face", [106305, 106306], (106133, "Right 3rd Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([160], "Finger_3rd_Distal_R", "Finger_3rd_Middle_R", ["Face", [101400, 101402], (101365, "Right 3rd Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([161], "Finger_4th_Proximal_R", "Carpals_R", ["Face", [107058, 107059], (107299, "Right 4th Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([162], "Finger_4th_Middle_R", "Finger_4th_Proximal_R", ["Face", [106343, 106350], (106321, "Right 4th Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([163], "Finger_4th_Distal_R", "Finger_4th_Middle_R", ["Face", [101789, 101790], (101579, "Right 4th Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([164], "Finger_5th_Proximal_R", "Carpals_R", ["Face", [106843, 106844], (106791, "Right 5th Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([165], "Finger_5th_Middle_R", "Finger_5th_Proximal_R", ["Face", [105962, 105976], (105931, "Right 5th Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([166], "Finger_5th_Distal_R", "Finger_5th_Middle_R", ["Face", [101182, 101183], (101121, "Right 5th Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # [69437, 69907] / 69166, "Manubrium"  / [100621, 100807] for Left Clavicle
                    # [114659, 114660, 114667, 114668] / 114647, "Left Scapula" / [224083] for Left Acromion

                    ([], "Clavicle_L", "Sternum", ["Face", [69437, 69907], (69166, "Manubrium")], False, False, (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), "Ball"),
                    ([167], "Scapula_L", "Clavicle_L", ["Face", [114659, 114660, 114667, 114668], (114647, "Left Scapula")], True, False, None, None, "Ball"),
                    # ([167], "Scapula_L", "Sternum", ["Joint", 19], True, False, None, None, "Ball"),
                    ([168], "Humerus_L", "Scapula_L", ["Joint", 20], True, False, None, None, "Ball"),
                    ([169], "Ulna_L", "Humerus_L", ["Joint", 21], True, False, (-np.pi/2), (np.pi/2), "Revolute"),
                    ([170], "Radius_L", "Ulna_L", ["Joint", 22], True, False, (0), (3/4*np.pi), "Revolute"),

                    # [121537, 121538] / (121491, "Left 1st Finger Metacarpal") / [231868, 231870] for Thumb_L
                    # [126490, 126491] / (126459, "Left Thumb Proximal Phalanx") / [246718, 246729] 
                    # [120895, 120877] / (120687, "Left Thumb Distal Phalanx") / [229452] 

                    # [125653, 125654] / (125395, "Left 2nd Finger Proximal Phalanx") / [243618, 243612] for 2nd Finger_L
                    # [124789, 124790] / (124619, "Left 2nd Finger Middle Phalanx") / [241230, 241200] 
                    # [119831, 119828] / (119799, "Left 2nd Finger Distal Phalanx") / [226727] 

                    # [125942, 125943] / (125895, "Left 3rd Finger Proximal Phalanx") / [245070, 245039] for 3rd Finger_L
                    # [125189, 125190] / (125017, "Left 3rd Finger Middle Phalanx") / [242406, 242367] 
                    # [120284, 120286] / (120249, "Left 3rd Finger Distal Phalanx") / [] 

                    # [126205, 126206] / (126183, "Left 4th Finger Proximal Phalanx")/ [245872, 245874] for 4th Finger_L
                    # [125234, 125236] / (125205, "Left 4th Finger Middle Phalanx") / [] 
                    # [120673, 120674] / (120463, "Left 4th Finger Distal Phalanx") / [] 

                    # [125727, 125728] / (125675, "Left 5th Finger Proximal Phalanx") / [244432, 244409] for 5th Finger_L
                    # [124860, 125005] / (124815, "Left 5th Finger Middle Phalanx") / [] 
                    # [120058, 120067] / (120005, "Left 5th Finger Distal Phalanx") / [] 

                    # # One Hand
                    # (range(171, 198), "Hand_L", "Radius_L", ["Joint", 23], True, True, (0, -np.pi/4, -np.pi/2), (0, np.pi/4, np.pi/2), "Ball"),

                    # One box Fingers
                    (list(range(171, 179)) + [180, 181, 182, 183], "Carpals_L", "Radius_L", ["Joint", 23], False, True, (-np.pi/2, 0, -np.pi/4), (np.pi/2, 0, np.pi/4), "Ball"),
                    ([179, 184, 185], "Finger_1st_L", "Carpals_L", ["Face", [121537, 121538], (121491, "Left 1st Finger Metacarpal")], True, False, None, None, "Ball"),
                    ([186, 187, 188], "Finger_2nd_L", "Carpals_L", ["Face", [125653, 125654], (125395, "Left 2nd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    ([189, 190, 191], "Finger_3rd_L", "Carpals_L", ["Face", [125942, 125943], (125895, "Left 3rd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    ([192, 193, 194], "Finger_4th_L", "Carpals_L", ["Face", [126205, 126206], (126183, "Left 4th Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    ([195, 196, 197], "Finger_5th_L", "Carpals_L", ["Face", [125727, 125728], (125675, "Left 5th Finger Proximal Phalanx")], True, False, None, None, "Ball"),             

                    # # Multi box Fingers
                    # (list(range(171, 179)) + [180, 181, 182, 183], "Carpals_L", "Radius_L", ["Joint", 23], False, True, (-np.pi/2, 0, -np.pi/4), (np.pi/2, 0, np.pi/4), "Ball"),
                    # ([179], "Finger_1st_Metacarpal_L", "Carpals_L", ["Face", [121537, 121538], (121491, "Left 1st Finger Metacarpal")], True, False, None, None, "Ball"),
                    # ([184], "Finger_1st_Proximal_L", "Finger_1st_Metacarpal_L", ["Face", [126490, 126491], (126459, "Left Thumb Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([185], "Finger_1st_Distal_L", "Finger_1st_Proximal_L", ["Face", [120895, 120877], (120687, "Left Thumb Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([186], "Finger_2nd_Proximal_L", "Carpals_L", ["Face", [125653, 125654], (125395, "Left 2nd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([187], "Finger_2nd_Middle_L", "Finger_2nd_Proximal_L", ["Face", [124789, 124790], (124619, "Left 2nd Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([188], "Finger_2nd_Distal_L", "Finger_2nd_Middle_L", ["Face", [119831, 119828], (119799, "Left 2nd Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([189], "Finger_3rd_Proximal_L", "Carpals_L", ["Face", [125942, 125943], (125895, "Left 3rd Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([190], "Finger_3rd_Middle_L", "Finger_3rd_Proximal_L", ["Face", [125189, 125190], (125017, "Left 3rd Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([191], "Finger_3rd_Distal_L", "Finger_3rd_Middle_L", ["Face", [120284, 120286], (120249, "Left 3rd Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([192], "Finger_4th_Proximal_L", "Carpals_L", ["Face", [126205, 126206], (126183, "Left 4th Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([193], "Finger_4th_Middle_L", "Finger_4th_Proximal_L", ["Face", [125234, 125236], (125205, "Left 4th Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([194], "Finger_4th_Distal_L", "Finger_4th_Middle_L", ["Face", [120673, 120674], (120463, "Left 4th Finger Distal Phalanx")], True, False, None, None, "Ball"),

                    # ([195], "Finger_5th_Proximal_L", "Carpals_L", ["Face", [125727, 125728], (125675, "Left 5th Finger Proximal Phalanx")], True, False, None, None, "Ball"),
                    # ([196], "Finger_5th_Middle_L", "Finger_5th_Proximal_L", ["Face", [124860, 125005], (124815, "Left 5th Finger Middle Phalanx")], True, False, None, None, "Ball"),
                    # ([197], "Finger_5th_Distal_L", "Finger_5th_Middle_L", ["Face", [120058, 120067], (120005, "Left 5th Finger Distal Phalanx")], True, False, None, None, "Ball"),
                ]
SKEL_dart_dofs = [
    "pelvis_r_x",
    "pelvis_r_y",
    "pelvis_r_z",
    "pelvis_t_x",
    "pelvis_t_y",
    "pelvis_t_z",
    "lumbar_x",
    "lumbar_y",
    "lumbar_z",
    "thorax_lateral",
    "thorax_axial",
    "thorax_extension",
    "cervix_lateral",
    "cervix_axial",
    "cervix_extension",
    "skull_lateral",
    "skull_axial",
    "skull_extension",
    "sternum_x",
    "sternum_y",
    "sternum_z",
    "left_rib_x",
    "left_rib_y",
    "left_rib_z",
    "right_rib_x",
    "right_rib_y",
    "right_rib_z",
    "right_femur_adduction",
    "right_femur_medial",
    "right_femur_flexion",
    "right_tibia_flexion",
    "right_talus_inversion",
    "right_talus_medial",
    "right_talus_flexion",
    "right_calcaneus_inversion",
    "right_calcaneus_medial",
    "right_calcaneus_flexion",
    "right_toe_x",
    "right_toe_y",
    "right_toe_z",
    "left_femur_abduction",
    "left_femur_lateral",
    "left_femur_flexion"
    "left_tibia_flexion",
    "left_talus_eversion",
    "left_talus_lateral",
    "left_talus_flexion",
    "left_calcaneus_eversion",
    "left_calcaneus_lateral",
    "left_calcaneus_flexion",
    "left_toe_x",
    "left_toe_y",
    "left_toe_z",
    "right_clavicle_x",
    "right_clavicle_y",
    "right_clavicle_z",
    "right_scapula_elevation",
    "right_scapula_protraction",
    "right_scapula_downward",
    "right_humerus_adduction",
    "right_humerus_medial",
    "right_humerus_flexion",
    "right_ulna_flexion",
    "right_radius_axial",
    "right_hand_flexion",
    "right_hand_axial (X)",
    "right_hand_deviation",

    "left_clavicle_x",
    "left_clavicle_y",
    "left_clavicle_z",
    "left_scapula_elevation",
    "left_scapula_retraction",
    "left_scapula_downward",
    "left_humerus_abduction",
    "left_humerus_lateral",
    "left_humerus_flexion",
    "left_ulna_extension",
    "left_radius_pronation",
    "left_hand_extension",
    "left_hand_axial (X)",
    "left_hand_deviation",
]

SKEL_face_index_male = {
    # In Anatomical Order
    
    # Axial Skeleton
    ## Vertebral Column
    "Cranium and Upper Jaw": [79230, 85817],                # 127
    "Lower Jaw": [85817, 88733],                            # 128
    
    "Atlas, C1": [88733, 89521],                            # 129
    "Axis, C2": [89521, 91333],                             # 130
    "C3": [94777, 95763],                                   # 135
    "C4": [92087, 92883],                                   # 132
    "C5": [91333, 92087],                                   # 131
    "C6": [93767, 94777],                                   # 134
    "C7": [92883, 93767],                                   # 133
    
    "T1": [52592, 53428],                                   # 80
    "T2": [54706, 55902],                                   # 82
    "T3": [55902, 57054],                                   # 83
    "T4": [53428, 54706],                                   # 81
    "T5": [51156, 52592],                                   # 79
    "T6": [47724, 48510],                                   # 76
    "T7": [46208, 47724],                                   # 75
    "T8": [42838, 43662],                                   # 72
    "T9": [44616, 46208],                                   # 74
    "T10": [48510, 50178],                                  # 77
    "T11": [43662, 44616],                                  # 73
    "T12": [50178, 51156],                                  # 78
    
    "L1": [35622, 37594],                                   # 68
    "L2": [39324, 40958],                                   # 70
    "L3": [40958, 42838],                                   # 71
    "L4": [37594, 39324],                                   # 69
    "L5": [33930, 35622],                                   # 67

    "Sacrum": [3094, 9046],                                 # 4
    "Coccyx": [0, 322],                                     # 0
    
    # Thoracic Cage
    ## Sternum
    "Manubrium": [69166, 69918],                            # 106
    "Sternum Body": [57054, 59580],                         # 84
    "Xiphoid Process": [59580, 59854],                      # 85

    ## Rib
    ### Left Rib
    "Left rib 1": [62630, 63364],                           # 92
    "Left rib 1 cartilage": [62420, 62630],                 # 91    
   
    "Left rib 2": [64856, 65460],                           # 97
    "Left rib 2 cartilage": [64648, 64856],                 # 96

    "Left rib 3": [68126, 68668],                           # 104
    "Left rib 3 cartilage": [67930, 68126],                 # 103

    "Left rib 4": [63558, 64116],                           # 94
    "Left rib 4 cartilage": [63364, 63558],                 # 93

    "Left rib 5": [61844, 62420],                           # 90
    "Left rib 5 cartilage": [61626, 61844],                 # 89
    
    "Left rib 6": [66900, 67482],                           # 101
    "Left rib 6 cartilage": [66452, 66900],                 # 100
    
    "Left rib 7": [65910, 66452],                           # 99
    "Left rib 7 cartilage": [65460, 65910],                 # 98
    
    "Left rib 8": [60626, 61102],                           # 87
    "Left rib 9": [64116, 64648],                           # 95
    "Left rib 10": [67482, 67930],                          # 102
    "Left 8910 costal cartilage": [59854, 60626],            # 86
    
    "Left rib 11": [61102, 61626],                          # 88
    "Left rib 12": [68668, 69166],                          # 105
    
    ### Right Rib
    "Right rib 1": [72694, 73428],                          # 113
    "Right rib 1 cartilage": [72484, 72694],                # 112

    "Right rib 2": [74920, 75524],                          # 118
    "Right rib 2 cartilage": [74712, 74920],                # 117

    "Right rib 3": [78190, 78732],                          # 125
    "Right rib 3 cartilage": [77994, 78190],                # 124

    "Right rib 4": [73622, 74180],                          # 115
    "Right rib 4 cartilage": [73428, 73622],                # 114

    "Right rib 5": [71908, 72484],                          # 111
    "Right rib 5 cartilage": [71690, 71908],                # 110

    "Right rib 6": [76964, 77546],                          # 122
    "Right rib 6 cartilage": [76516, 76964],                # 121

    "Right rib 7": [75974, 76516],                          # 120
    "Right rib 7 cartilage": [75524, 75974],                # 119
    
    "Right rib 8": [70690, 71166],                          # 108
    "Right rib 9": [74180, 74712],                          # 116
    "Right rib 10": [77546, 77994],                         # 123
    "Right 8910 costal cartilage": [69918, 70690],           # 107
    
    "Right rib 11": [71166, 71690],                         # 109
    "Right rib 12": [78732, 79230],                         # 126

    # Appendicular Skeleton
    ## Lower Limbs
    ### Pelvic Girdle
    "Left Hip": [322, 1634],                                # 1
    "Pubic Symphysis": [1634, 1782],                        # 2
    "Right Hip": [1782, 3094],                              # 3

    ### Right Lower Limb
    "Right Femur": [9046, 10698],                           # 5
    "Right Fibula": [10698, 11508],                         # 6
    "Right Tibia": [11508, 12380],                          # 7

    #### Right Foot
    ##### Right Carpals
    "Right Talus": [12380, 12574],                          # 8
    "Right Calcaneus": [12574, 13258],                      # 9
    "Right Navicular": [17708, 18168],                      # 19
    "Right Cuboid": [13258, 13802],                         # 10
    "Right Medial Cuneiform": [16088, 16672],               # 16
    "Right Intermediate Cuneiform": [15340, 15670],         # 14
    "Right Lateral Cuneiform": [15670, 16088],              # 15

    ##### Right Toe Metacarpals
    "Right 1st Toe Metacarpal": [14288, 14932],                 # 12
    "Right 2nd Toe Metacarpal": [16672, 17232],                 # 17
    "Right 3rd Toe Metacarpal": [17232, 17708],                 # 18
    "Right 4th Toe Metacarpal": [14932, 15340],                 # 13
    "Right 5th Toe Metacarpal": [13802, 14288],                 # 11

    ##### Right Phalanges
    "Right 1st Toe Proximal Phalanx": [20274, 20588],       # 31
    "Right 2nd Toe Proximal Phalanx": [21062, 21286],       # 34
    "Right 3rd Toe Proximal Phalanx": [21286, 21488],       # 35
    "Right 4th Toe Proximal Phalanx": [20588, 20808],       # 32
    "Right 5th Toe Proximal Phalanx": [20808, 21062],       # 33
    "Right 2nd Toe Middle Phalanx": [19854, 20076],         # 29
    "Right 3rd Toe Middle Phalanx": [20076, 20274],         # 30
    "Right 4th Toe Middle Phalanx": [19444, 19658],         # 27
    "Right 5th Toe Middle Phalanx": [19658, 19854],         # 28
    "Right 1st Toe Distal Phalanx": [18354, 18614],         # 22
    "Right 2nd Toe Distal Phalanx": [19028, 19246],         # 25
    "Right 3rd Toe Distal Phalanx": [19246, 19444],         # 26
    "Right 4th Toe Distal Phalanx": [18614, 18822],         # 23
    "Right 5th Toe Distal Phalanx": [18822, 19028],         # 24

    "Right 1st Sesamoid": [18168, 18288],                   # 20
    "Right 2nd Sesamoid": [18288, 18354],                   # 21
    
    ### Left Lower Limb
    "Left Femur": [21488, 23140],                           # 36
    "Left Fibula": [23140, 23950],                          # 37
    "Left Tibia": [23950, 24822],                           # 38
    
    #### Left Foot
    ##### Left Carpals
    "Left Talus": [24822, 25016],                           # 39
    "Left Calcaneus": [25016, 25700],                       # 40
    "Left Navicular": [30150, 30610],                       # 50
    "Left Cuboid": [25700, 26244],                          # 41
    "Left Medial Cuneiform": [28530, 29114],                # 47
    "Left Intermediate Cuneiform": [27782, 28112],          # 45
    "Left Lateral Cuneiform": [28112, 28530],               # 46

    ##### Left Metacarpals
    "Left 1st Toe Metacarpal": [26730, 27374],                  # 43
    "Left 2nd Toe Metacarpal": [29114, 29674],                  # 48
    "Left 3rd Toe Metacarpal": [29674, 30150],                  # 49
    "Left 4th Toe Metacarpal": [27374, 27782],                  # 44
    "Left 5th Toe Metacarpal": [26244, 26730],                  # 42

    ##### Left Phalanges
    "Left 1st Toe Proximal Phalanx": [32716, 33030],        # 62
    "Left 2nd Toe Proximal Phalanx": [33504, 33728],        # 65
    "Left 3rd Toe Proximal Phalanx": [33728, 33930],        # 66
    "Left 4th Toe Proximal Phalanx": [33030, 33250],        # 63
    "Left 5th Toe Proximal Phalanx": [33250, 33504],        # 64
    "Left 2nd Toe Middle Phalanx": [32296, 32518],          # 60
    "Left 3rd Toe Middle Phalanx": [32518, 32716],          # 61
    "Left 4th Toe Middle Phalanx": [31886, 32100],          # 58
    "Left 5th Toe Middle Phalanx": [32100, 32296],          # 59
    "Left 1st Toe Distal Phalanx": [30796, 31056],          # 53
    "Left 2nd Toe Distal Phalanx": [31470, 31688],          # 56
    "Left 3rd Toe Distal Phalanx": [31688, 31886],          # 57
    "Left 4th Toe Distal Phalanx": [31056, 31264],          # 54
    "Left 5th Toe Distal Phalanx": [31264, 31470],          # 55
    
    "Left 1st Sesamoid": [30610, 30730],                    # 51
    "Left 2nd Sesamoid": [30730, 30796],                    # 52


    ## Upper Limbs
    ### Right Upper Limb
    "Right Scapula": [95763, 97737],                        # 136
    "Right Humerus" : [97737, 99891],                       # 137
    "Right Ulna" : [99891, 100447],                         # 138
    "Right Radius" : [100447, 100915],                      # 139
    
    #### Right Hand
    ##### Right Carpals
    "Right Capitate": [102019, 102333],                     # 145
    "Right Hamate": [103275, 103557],                       # 149
    "Right Lunate": [103557, 103747],                       # 150
    "Right Pisiform": [103747, 104077],                     # 151
    "Right Scaphold": [104077, 104259],                     # 152
    "Right Trapezium": [105065, 105295],                    # 155
    "Right Trapezoid": [105295, 105493],                    # 156
    "Right Triquetral": [105493, 105735],                   # 157

    ##### Right Finger Metacarpals
    "Right 1st Finger Metacarpal": [102607, 102951],               # 147
    "Right 2nd Finger Metacarpal": [104259, 104645],               # 153
    "Right 3rd Finger Metacarpal": [104645, 105065],               # 154
    "Right 4th Finger Metacarpal": [102951, 103275],               # 148
    "Right 5th Finger Metacarpal": [102333, 102607],               # 146
    
    ##### Right Phalanges
    "Right Thumb Proximal Phalanx": [107575, 107781],       # 166
    "Right Thumb Distal Phalanx": [101803, 102019],         # 144
    "Right 2nd Finger Proximal Phalanx": [106511, 106791],  # 162
    "Right 2nd Finger Middle Phalanx": [105735, 105931],    # 158
    "Right 2nd Finger Distal Phalanx": [100915, 101121],    # 140
    "Right 3rd Finger Proximal Phalanx": [107011, 107299],  # 164
    "Right 3rd Finger Middle Phalanx": [106133, 106321],    # 160
    "Right 3rd Finger Distal Phalanx": [101365, 101579],    # 142
    "Right 4th Finger Proximal Phalanx": [107299, 107575],  # 165
    "Right 4th Finger Middle Phalanx": [106321, 106511],    # 161
    "Right 4th Finger Distal Phalanx": [101579, 101803],    # 143
    "Right 5th Finger Proximal Phalanx": [106791, 107011],  # 163
    "Right 5th Finger Middle Phalanx": [105931, 106133],    # 159
    "Right 5th Finger Distal Phalanx": [101121, 101365],    # 141

    ### Left Upper Limb
    "Left Scapula" : [114647, 116621],                      # 167
    "Left Humerus" : [116621, 118775],                      # 168
    "Left Ulna" : [118775, 119331],                         # 169
    "Left Radius" : [119331, 119799],                       # 170

    #### Left Hand
    ##### Left Carpals
    "Left Capitate": [120903, 121217],                      # 176
    "Left Hamate": [122159, 122441],                        # 180
    "Left Lunate": [122441, 122631],                        # 181
    "Left Pisiform": [122631, 122961],                      # 182
    "Left Scaphold": [122961, 123143],                      # 183
    "Left Trapezium": [123949, 124179],                     # 186
    "Left Trapezoid": [124179, 124377],                     # 187
    "Left Triquetral": [124377, 124619],                    # 188

    ##### Left Finger Metacarpals
    "Left 1st Finger Metacarpal": [121491, 121835],                # 178
    "Left 2nd Finger Metacarpal": [123143, 123529],                # 184
    "Left 3rd Finger Metacarpal": [123529, 123949],                # 185
    "Left 4th Finger Metacarpal": [121835, 122159],                # 179
    "Left 5th Finger Metacarpal": [121217, 121491],                # 177
    
    ##### Left Phalanges
    "Left Thumb Proximal Phalanx": [126459, 126665],        # 197
    "Left Thumb Distal Phalanx": [120687, 120903],          # 175
    "Left 2nd Finger Proximal Phalanx": [125395, 125675],   # 193
    "Left 2nd Finger Middle Phalanx": [124619, 124815],     # 189
    "Left 2nd Finger Distal Phalanx": [119799, 120005],     # 171
    "Left 3rd Finger Proximal Phalanx": [125895, 126183],   # 195
    "Left 3rd Finger Middle Phalanx": [125017, 125205],     # 191
    "Left 3rd Finger Distal Phalanx": [120249, 120463],     # 173
    "Left 4th Finger Proximal Phalanx": [126183, 126459],   # 196
    "Left 4th Finger Middle Phalanx": [125205, 125395],     # 192
    "Left 4th Finger Distal Phalanx": [120463, 120687],     # 174
    "Left 5th Finger Proximal Phalanx": [125675, 125895],   # 194
    "Left 5th Finger Middle Phalanx": [124815, 125017],     # 190    
    "Left 5th Finger Distal Phalanx": [120005, 120249],     # 172

    # # In Ascending Order
    # "Coccyx": [0, 322],                                     # 0
    # "Left Hip": [322, 1634],                                # 1
    # "Pubic Symphysis": [1634, 1782],                        # 2
    # "Right Hip": [1782, 3094],                              # 3
    # "Sacrum": [3094, 9046],                                 # 4
    # "Right Femur": [9046, 10698],                           # 5

    # # "Right Tibia": [10698, 12380],

    # "Right Fibula": [10698, 11508],                         # 6
    # "Right Tibia": [11508, 12380],                          # 7

    # # "Right Talus": [12380, 13258],
    # # "Right Calcaneus": [13258, 18354],

    # "Right Talus": [12380, 12574],                          # 8
    # "Right Calcaneus": [12574, 13258],                      # 9
    # "Right Cuboid": [13258, 13802],                         # 10
    # "Right 5th Toe Metacarpal": [13802, 14288],                 # 11
    # "Right 1st Toe Metacarpal": [14288, 14932],                 # 12
    # "Right 4th Toe Metacarpal": [14932, 15340],                 # 13
    # "Right Intermediate Cuneiform": [15340, 15670],         # 14
    # "Right Lateral Cuneiform": [15670, 16088],              # 15
    # "Right Medial Cuneiform": [16088, 16672],               # 16
    # "Right 2nd Toe Metacarpal": [16672, 17232],                 # 17
    # "Right 3rd Toe Metacarpal": [17232, 17708],                 # 18
    # "Right Navicular": [17708, 18168],                      # 19
    # "Right 1st Sesamoid": [18168, 18288],                   # 20
    # "Right 2nd Sesamoid": [18288, 18354],                   # 21

    # # "Right Toe": [18354, 21488],

    # "Right 1st Toe Distal Phalanx": [18354, 18614],         # 22
    # "Right 4th Toe Distal Phalanx": [18614, 18822],         # 23
    # "Right 5th Toe Distal Phalanx": [18822, 19028],         # 24
    # "Right 2nd Toe Distal Phalanx": [19028, 19246],         # 25
    # "Right 3rd Toe Distal Phalanx": [19246, 19444],         # 26
    # "Right 4th Toe Middle Phalanx": [19444, 19658],         # 27
    # "Right 5th Toe Middle Phalanx": [19658, 19854],         # 28
    # "Right 2nd Toe Middle Phalanx": [19854, 20076],         # 29
    # "Right 3rd Toe Middle Phalanx": [20076, 20274],         # 30
    # "Right 1st Toe Proximal Phalanx": [20274, 20588],       # 31
    # "Right 4th Toe Proximal Phalanx": [20588, 20808],       # 32
    # "Right 5th Toe Proximal Phalanx": [20808, 21062],       # 33
    # "Right 2nd Toe Proximal Phalanx": [21062, 21286],       # 34
    # "Right 3rd Toe Proximal Phalanx": [21286, 21488],       # 35
    
    # "Left Femur": [21488, 23140],                           # 36

    # # "Left Tibia": [23140, 24822],

    # "Left Fibula": [23140, 23950],                          # 37
    # "Left Tibia": [23950, 24822],                           # 38
    
    # # "Left Talus": [24822, 25700],
    # # "Left Calcaneus": [25700, 30796],

    # "Left Talus": [24822, 25016],                           # 39
    # "Left Calcaneus": [25016, 25700],                       # 40
    # "Left Cuboid": [25700, 26244],                          # 41
    # "Left 5th Toe Metacarpal": [26244, 26730],                  # 42
    # "Left 1st Toe Metacarpal": [26730, 27374],                  # 43
    # "Left 4th Toe Metacarpal": [27374, 27782],                  # 44
    # "Left Intermediate Cuneiform": [27782, 28112],          # 45
    # "Left Lateral Cuneiform": [28112, 28530],               # 46
    # "Left Medial Cuneiform": [28530, 29114],                # 47
    # "Left 2nd Toe Metacarpal": [29114, 29674],                  # 48
    # "Left 3rd Toe Metacarpal": [29674, 30150],                  # 49
    # "Left Navicular": [30150, 30610],                       # 50
    # "Left 1st Sesamoid": [30610, 30730],                    # 51
    # "Left 2nd Sesamoid": [30730, 30796],                    # 52

    # # "Left Toe": [30796, 33930],

    # "Left 1st Toe Distal Phalanx": [30796, 31056],          # 53
    # "Left 4th Toe Distal Phalanx": [31056, 31264],          # 54
    # "Left 5th Toe Distal Phalanx": [31264, 31470],          # 55
    # "Left 2nd Toe Distal Phalanx": [31470, 31688],          # 56
    # "Left 3rd Toe Distal Phalanx": [31688, 31886],          # 57
    # "Left 4th Toe Middle Phalanx": [31886, 32100],          # 58
    # "Left 5th Toe Middle Phalanx": [32100, 32296],          # 59
    # "Left 2nd Toe Middle Phalanx": [32296, 32518],          # 60
    # "Left 3rd Toe Middle Phalanx": [32518, 32716],          # 61
    # "Left 1st Toe Proximal Phalanx": [32716, 33030],        # 62
    # "Left 4th Toe Proximal Phalanx": [33030, 33250],        # 63
    # "Left 5th Toe Proximal Phalanx": [33250, 33504],        # 64
    # "Left 2nd Toe Proximal Phalanx": [33504, 33728],        # 65
    # "Left 3rd Toe Proximal Phalanx": [33728, 33930],        # 66

    # "L5": [33930, 35622],                                   # 67
    # "L1": [35622, 37594],                                   # 68
    # "L4": [37594, 39324],                                   # 69
    # "L2": [39324, 40958],                                   # 70
    # "L3": [40958, 42838],                                   # 71

    # "T8": [42838, 43662],                                   # 72
    # "T11": [43662, 44616],                                  # 73
    # "T9": [44616, 46208],                                   # 74
    # "T7": [46208, 47724],                                   # 75
    # "T6": [47724, 48510],                                   # 76
    # "T10": [48510, 50178],                                  # 77
    # "T12": [50178, 51156],                                  # 78
    # "T5": [51156, 52592],                                   # 79
    # "T1": [52592, 53428],                                   # 80
    # "T4": [53428, 54706],                                   # 81
    # "T2": [54706, 55902],                                   # 82
    # "T3": [55902, 57054],                                   # 83

    # # "Sternum": [57054, 59854],
    # "Sternum Body": [57054, 59580],                         # 84
    # "Xiphoid Process": [59580, 59854],                      # 85

    # "Left 8910 costal cartilage": [59854, 60626],            # 86
    # "Left rib 8": [60626, 61102],                           # 87
    # "Left rib 11": [61102, 61626],                          # 88
    # # "Left rib 5": [61626, 62420],
    # "Left rib 5 cartilage": [61626, 61844],                 # 89
    # "Left rib 5": [61844, 62420],                           # 90

    # # "Left rib 1": [62420, 63364],
    # "Left rib 1 cartilage": [62420, 62630],                 # 91    
    # "Left rib 1": [62630, 63364],                           # 92

    # # "Left rib 4": [63364, 64116],
    # "Left rib 4 cartilage": [63364, 63558],                 # 93
    # "Left rib 4": [63558, 64116],                           # 94

    # "Left rib 9": [64116, 64648],                           # 95

    # # "Left rib 2": [64648, 65460],
    # "Left rib 2 cartilage": [64648, 64856],                 # 96
    # "Left rib 2": [64856, 65460],                           # 97

    # # "Left rib 7": [65460, 66452],
    # "Left rib 7 cartilage": [65460, 65910],                 # 98
    # "Left rib 7": [65910, 66452],                           # 99

    # # "Left rib 6": [66452, 67482],
    # "Left rib 6 cartilage": [66452, 66900],                 # 100
    # "Left rib 6": [66900, 67482],                           # 101

    # "Left rib 10": [67482, 67930],                          # 102

    # # "Left rib 3": [67930, 68668],
    # "Left rib 3 cartilage": [67930, 68126],                 # 103
    # "Left rib 3": [68126, 68668],                           # 104

    # "Left rib 12": [68668, 69166],                          # 105

    # "Manubrium": [69166, 69918],                            # 106
    # "Right 8910 costal cartilage": [69918, 70690],           # 107
    # "Right rib 8": [70690, 71166],                          # 108
    # "Right rib 11": [71166, 71690],                         # 109

    # # "Right rib 5": [71690, 72484],
    # "Right rib 5 cartilage": [71690, 71908],                # 110
    # "Right rib 5": [71908, 72484],                          # 111

    # # "Right rib 1": [72484, 73428],
    # "Right rib 1 cartilage": [72484, 72694],                # 112
    # "Right rib 1": [72694, 73428],                          # 113
            
    # # "Right rib 4": [73428, 74180],
    # "Right rib 4 cartilage": [73428, 73622],                # 114
    # "Right rib 4": [73622, 74180],                          # 115

    # "Right rib 9": [74180, 74712],                          # 116

    # # "Right rib 2": [74712, 75524],
    # "Right rib 2 cartilage": [74712, 74920],                # 117
    # "Right rib 2": [74920, 75524],                          # 118

    # # "Right rib 7": [75524, 76516],
    # "Right rib 7 cartilage": [75524, 75974],                # 119
    # "Right rib 7": [75974, 76516],                          # 120

    # # "Right rib 6": [76516, 77546],
    # "Right rib 6 cartilage": [76516, 76964],                # 121
    # "Right rib 6": [76964, 77546],                          # 122

    # "Right rib 10": [77546, 77994],                         # 123

    # # "Right rib 3": [77994, 78732],
    # "Right rib 3 cartilage": [77994, 78190],                # 124
    # "Right rib 3": [78190, 78732],                          # 125

    # "Right rib 12": [78732, 79230],                         # 126

    # # "Skull": [79230, 88733],
    # "Cranium and Upper Jaw": [79230, 85817],                # 127
    # "Lower Jaw": [85817, 88733],                            # 128
    
    # "Atlas, C1": [88733, 89521],                            # 129
    # "Axis, C2": [89521, 91333],                             # 130
    # "C5": [91333, 92087],                                   # 131
    # "C4": [92087, 92883],                                   # 132
    # "C7": [92883, 93767],                                   # 133
    # "C6": [93767, 94777],                                   # 134
    # "C3": [94777, 95763],                                   # 135

    # "Right Scapula": [95763, 97737],                        # 136
    # "Right Humerus" : [97737, 99891],                       # 137
    # "Right Ulna" : [99891, 100447],                         # 138
    # "Right Radius" : [100447, 100915],                      # 139

    # # "Right Hand" : [100915, 114647],
    # "Right 2nd Finger Distal Phalanx": [100915, 101121],    # 140
    # "Right 5th Finger Distal Phalanx": [101121, 101365],    # 141
    # "Right 3rd Finger Distal Phalanx": [101365, 101579],    # 142
    # "Right 4th Finger Distal Phalanx": [101579, 101803],    # 143
    # "Right Thumb Distal Phalanx": [101803, 102019],         # 144

    # "Right Capitate": [102019, 102333],                     # 145
    # "Right 5th Finger Metacarpal": [102333, 102607],               # 146
    # "Right 1st Finger Metacarpal": [102607, 102951],               # 147
    # "Right 4th Finger Metacarpal": [102951, 103275],               # 148
    # "Right Hamate": [103275, 103557],                       # 149
    # "Right Lunate": [103557, 103747],                       # 150
    # "Right Pisiform": [103747, 104077],                     # 151
    # "Right Scaphold": [104077, 104259],                     # 152
    # "Right 2nd Finger Metacarpal": [104259, 104645],               # 153
    # "Right 3rd Finger Metacarpal": [104645, 105065],               # 154
    # "Right Trapezium": [105065, 105295],                    # 155
    # "Right Trapezoid": [105295, 105493],                    # 156
    # "Right Triquetral": [105493, 105735],                   # 157
    # "Right 2nd Finger Middle Phalanx": [105735, 105931],    # 158
    # "Right 5th Finger Middle Phalanx": [105931, 106133],    # 159
    # "Right 3rd Finger Middle Phalanx": [106133, 106321],    # 160
    # "Right 4th Finger Middle Phalanx": [106321, 106511],    # 161
    # "Right 2nd Finger Proximal Phalanx": [106511, 106791],  # 162
    # "Right 5th Finger Proximal Phalanx": [106791, 107011],  # 163
    # "Right 3rd Finger Proximal Phalanx": [107011, 107299],  # 164
    # "Right 4th Finger Proximal Phalanx": [107299, 107575],  # 165
    # "Right Thumb Proximal Phalanx": [107575, 107781],       # 166

    # # "Right 2nd Finger Distal Phalanx": [107781, 107987],
    # # "Right 5th Finger Distal Phalanx": [107987, 108231],
    # # "Right 3rd Finger Distal Phalanx": [108231, 108445],
    # # "Right 4th Finger Distal Phalanx": [108445, 108669],
    # # "Right Thumb Distal Phalanx": [108669, 108885],

    # # "Right Capitate": [108885, 109199],
    # # "Right 5th Finger Metacarpal": [109199, 109473],
    # # "Right 1st Finger Metacarpal": [109473, 109817],
    # # "Right 4th Finger Metacarpal": [109817, 110141],
    # # "Right Hamate": [110141, 110423],
    # # "Right Lunate": [110423, 110613],
    # # "Right Pisiform": [110613, 110943],
    # # "Right Scaphold": [110943, 111125],
    # # "Right 2nd Finger Metacarpal": [111125, 111511],
    # # "Right 3rd Finger Metacarpal": [111511, 111931],
    # # "Right Trapezium": [111931, 112161],
    # # "Right Trapezoid": [112161, 112359],
    # # "Right Triquetral": [112359, 112601],
    # # "Right 2nd Finger Middle Phalanx": [112601, 112797],
    # # "Right 5th Finger Middle Phalanx": [112797, 112999],
    # # "Right 3rd Finger Middle Phalanx": [112999, 113187],
    # # "Right 4th Finger Middle Phalanx": [113187, 113377],
    # # "Right 2nd Finger Proximal Phalanx": [113377, 113657],
    # # "Right 5th Finger Proximal Phalanx": [113657, 113877],
    # # "Right 3rd Finger Proximal Phalanx": [113877, 114165],
    # # "Right 4th Finger Proximal Phalanx": [114165, 114441],
    # # "Right Thumb Proximal Phalanx": [114441, 114647],

    # "Left Scapula" : [114647, 116621],                      # 167
    # "Left Humerus" : [116621, 118775],                      # 168
    # "Left Ulna" : [118775, 119331],                         # 169
    # "Left Radius" : [119331, 119799],                       # 170

    # # "Left Hand" : [119799, 126664],
    # "Left 2nd Finger Distal Phalanx": [119799, 120005],     # 171
    # "Left 5th Finger Distal Phalanx": [120005, 120249],     # 172
    # "Left 3rd Finger Distal Phalanx": [120249, 120463],     # 173
    # "Left 4th Finger Distal Phalanx": [120463, 120687],     # 174
    # "Left Thumb Distal Phalanx": [120687, 120903],          # 175

    # "Left Capitate": [120903, 121217],                      # 176
    # "Left 5th Finger Metacarpal": [121217, 121491],                # 177
    # "Left 1st Finger Metacarpal": [121491, 121835],                # 178
    # "Left 4th Finger Metacarpal": [121835, 122159],                # 179
    # "Left Hamate": [122159, 122441],                        # 180
    # "Left Lunate": [122441, 122631],                        # 181
    # "Left Pisiform": [122631, 122961],                      # 182
    # "Left Scaphold": [122961, 123143],                      # 183
    # "Left 2nd Finger Metacarpal": [123143, 123529],                # 184
    # "Left 3rd Finger Metacarpal": [123529, 123949],                # 185
    # "Left Trapezium": [123949, 124179],                     # 186
    # "Left Trapezoid": [124179, 124377],                     # 187
    # "Left Triquetral": [124377, 124619],                    # 188
    # "Left 2nd Finger Middle Phalanx": [124619, 124815],     # 189
    # "Left 5th Finger Middle Phalanx": [124815, 125017],     # 190
    # "Left 3rd Finger Middle Phalanx": [125017, 125205],     # 191
    # "Left 4th Finger Middle Phalanx": [125205, 125395],     # 192
    # "Left 2nd Finger Proximal Phalanx": [125395, 125675],   # 193
    # "Left 5th Finger Proximal Phalanx": [125675, 125895],   # 194
    # "Left 3rd Finger Proximal Phalanx": [125895, 126183],   # 195
    # "Left 4th Finger Proximal Phalanx": [126183, 126459],   # 196
    # "Left Thumb Proximal Phalanx": [126459, 126665],        # 197
}

SKEL_face_index_female = {
    "Coccyx": [0, 322],
    "Left Hip": [322, 1634],
    "Pubic Symphysis": [1634, 1782],
    "Right Hip": [1782, 3094],
    "Sacrum": [3094, 9046],
    "Right Femur": [9046, 10698],
    # "Right Tibia": [10698, 12380],

    "Right Fibula": [10698, 11508],
    "Right Tibia": [11508, 12380],

    # "Right Talus": [12380, 13258],
    # "Right Calcaneus": [13258, 18354],

    "Right Talus": [12380, 12574],
    "Right Calcaneus": [12574, 13258],
    "Right Cuboid": [13258, 13802],
    "Right 5th Toe Metacarpal": [13802, 14288],
    "Right 1st Toe Metacarpal": [14288, 14932],
    "Right 4th Toe Metacarpal": [14932, 15340],
    "Right Intermediate Cuneiform": [15340, 15670],
    "Right Lateral Cuneiform": [15670, 16088],
    "Right Medial Cuneiform": [16088, 16672],
    "Right 2nd Toe Metacarpal": [16672, 17232],
    "Right 3rd Toe Metacarpal": [17232, 17708],
    "Right Navicular": [17708, 18168],
    "Right 1st Sesamoid": [18168, 18288],
    "Right 2nd Sesamoid": [18288, 18354],

    # "Right Toe": [18354, 21488],

    "Right 1st Toe Distal Phalanx": [18354, 18614],
    "Right 4th Toe Distal Phalanx": [18614, 18822],
    "Right 5th Toe Distal Phalanx": [18822, 19028],
    "Right 2nd Toe Distal Phalanx": [19028, 19246],
    "Right 3rd Toe Distal Phalanx": [19246, 19444],
    "Right 4th Toe Middle Phalanx": [19444, 19658],
    "Right 5th Toe Middle Phalanx": [19658, 19854],
    "Right 2nd Toe Middle Phalanx": [19854, 20076],
    "Right 3rd Toe Middle Phalanx": [20076, 20274],
    "Right 1st Toe Proximal Phalanx": [20274, 20588],
    "Right 4th Toe Proximal Phalanx": [20588, 20808],
    "Right 5th Toe Proximal Phalanx": [20808, 21062],
    "Right 2nd Toe Proximal Phalanx": [21062, 21286],
    "Right 3rd Toe Proximal Phalanx": [21286, 21488],

    "Left Femur": [21488, 23140],
    # "Left Tibia": [23140, 24822],

    "Left Fibula": [23140, 23950],
    "Left Tibia": [23950, 24822],
    
    # "Left Talus": [24822, 25700],
    # "Left Calcaneus": [25700, 30796],

    "Left Talus": [24822, 25016],
    "Left Calcaneus": [25016, 25700],
    "Left Cuboid": [25700, 26244],
    "Left 5th Toe Metacarpal": [26244, 26730],
    "Left 1st Toe Metacarpal": [26730, 27374],
    "Left 4th Toe Metacarpal": [27374, 27782],
    "Left Intermediate Cuneiform": [27782, 28112],
    "Left Lateral Cuneiform": [28112, 28530],
    "Left Medial Cuneiform": [28530, 29114],
    "Left 2nd Toe Metacarpal": [29114, 29674],
    "Left 3rd Toe Metacarpal": [29674, 30150],
    "Left Navicular": [30150, 30610],
    "Left 1st Sesamoid": [30610, 30730],
    "Left 2nd Sesamoid": [30730, 30796],

    # "Left Toe": [30796, 33930],

    "Left 1st Toe Distal Phalanx": [30796, 31056],
    "Left 4th Toe Distal Phalanx": [31056, 31264],
    "Left 5th Toe Distal Phalanx": [31264, 31470],
    "Left 2nd Toe Distal Phalanx": [31470, 31688],
    "Left 3rd Toe Distal Phalanx": [31688, 31886],
    "Left 4th Toe Middle Phalanx": [31886, 32100],
    "Left 5th Toe Middle Phalanx": [32100, 32296],
    "Left 2nd Toe Middle Phalanx": [32296, 32518],
    "Left 3rd Toe Middle Phalanx": [32518, 32716],
    "Left 1st Toe Proximal Phalanx": [32716, 33030],
    "Left 4th Toe Proximal Phalanx": [33030, 33250],
    "Left 5th Toe Proximal Phalanx": [33250, 33504],
    "Left 2nd Toe Proximal Phalanx": [33504, 33728],
    "Left 3rd Toe Proximal Phalanx": [33728, 33930],
    
    "L5": [33930, 35622],
    "L1": [35622, 37594],
    "L4": [37594, 39324],
    "L2": [39324, 40958],
    "L3": [40958, 42838],
    "T8": [42838, 43662],
    "T11": [43662, 44616],
    "T9": [44616, 46208],
    "T7": [46208, 47724],
    "T6": [47724, 48510],
    "T10": [48510, 50178],
    "T12": [50178, 51156],
    "T5": [51156, 52592],
    "T1": [52592, 53428],
    "T4": [53428, 54706],
    "T2": [54706, 55902],
    "T3": [55902, 57054],

    # "Sternum": [57054, 59854],
    "Sternum Body": [57054, 59580],
    "Xiphoid Process": [59580, 59854],

    "Left 8910 costal cartilage": [59854, 60626],
    "Left rib 8": [60626, 61102],
    "Left rib 11": [61102, 61626],
    # "Left rib 5": [61626, 62420],
    "Left rib 5 cartilage": [61626, 61844],
    "Left rib 5": [61844, 62420],

    # "Left rib 1": [62420, 63364],
    "Left rib 1 cartilage": [62420, 62630],
    "Left rib 1": [62630, 63364],

    # "Left rib 4": [63364, 64116],
    "Left rib 4 cartilage": [63364, 63558],
    "Left rib 4": [63558, 64116],

    "Left rib 9": [64116, 64648],

    # "Left rib 2": [64648, 65460],
    "Left rib 2 cartilage": [64648, 64856], 
    "Left rib 2": [64856, 65460],

    # "Left rib 7": [65460, 66452],
    "Left rib 7 cartilage": [65460, 65910],
    "Left rib 7": [65910, 66452],

    # "Left rib 6": [66452, 67482],
    "Left rib 6 cartilage": [66452, 66900],
    "Left rib 6": [66900, 67482],

    "Left rib 10": [67482, 67930],

    # "Left rib 3": [67930, 68668],
    "Left rib 3 cartilage": [67930, 68126],
    "Left rib 3": [68126, 68668],

    "Left rib 12": [68668, 69166],

    "Manubrium": [69166, 69918],
    "Right 8910 costal cartilage": [69918, 70690],
    "Right rib 8": [70690, 71166],
    "Right rib 11": [71166, 71690],

    # "Right rib 5": [71690, 72484],
    "Right rib 5 cartilage": [71690, 71908],
    "Right rib 5": [71908, 72484],

    # "Right rib 1": [72484, 73428],
    "Right rib 1 cartilage": [72484, 72694],
    "Right rib 1": [72694, 73428],

    # "Right rib 4": [73428, 74180],
    "Right rib 4 cartilage": [73428, 73622],
    "Right rib 4": [73622, 74180],

    "Right rib 9": [74180, 74712],

    # "Right rib 2": [74712, 75524],
    "Right rib 2 cartilage": [74712, 74920],
    "Right rib 2": [74920, 75524],

    # "Right rib 7": [75524, 76516],
    "Right rib 7 cartilage": [75524, 75974],
    "Right rib 7": [75974, 76516],

    # "Right rib 6": [76516, 77546],
    "Right rib 6 cartilage": [76516, 76964],
    "Right rib 6": [76964, 77546],

    "Right rib 10": [77546, 77994],

    # "Right rib 3": [77994, 78732],
    "Right rib 3 cartilage": [77994, 78190],
    "Right rib 3": [78190, 78732],

    "Right rib 12": [78732, 79230],

    "Atlas, C1": [79230, 80018],
    "Axis, C2": [80018, 81830],
    "C5": [81830, 82590],
    "C4": [82590, 83380],
    "C7": [83380, 84264],
    "C6": [84264, 85274],
    "C3": [85274, 86260],

    # "Skull": [86260, 95763],
    "Lower Jaw": [86260, 89176],
    "Cranium and Upper Jaw": [89176, 95763],
    
    "Right Scapula": [95763, 97737],
    "Right Humerus" : [97737, 99891],
    "Right Ulna" : [99891, 100447],
    "Right Radius" : [100447, 100915],

    # "Right Hand" : [100915, 114647],
    "Right 2nd Finger Distal Phalanx": [100915, 101121],
    "Right 5th Finger Distal Phalanx": [101121, 101365],
    "Right 3rd Finger Distal Phalanx": [101365, 101579],
    "Right 4th Finger Distal Phalanx": [101579, 101803],
    "Right Thumb Distal Phalanx": [101803, 102019],

    "Right Capitate": [102019, 102333],
    "Right 5th Finger Metacarpal": [102333, 102607],
    "Right 1st Finger Metacarpal": [102607, 102951],
    "Right 4th Finger Metacarpal": [102951, 103275],
    "Right Hamate": [103275, 103557],
    "Right Lunate": [103557, 103747],
    "Right Pisiform": [103747, 104077],
    "Right Scaphold": [104077, 104259],
    "Right 2nd Finger Metacarpal": [104259, 104645],
    "Right 3rd Finger Metacarpal": [104645, 105065],
    "Right Trapezium": [105065, 105295],
    "Right Trapezoid": [105295, 105493],
    "Right Triquetral": [105493, 105735],
    "Right 2nd Finger Middle Phalanx": [105735, 105931],
    "Right 5th Finger Middle Phalanx": [105931, 106133],
    "Right 3rd Finger Middle Phalanx": [106133, 106321],
    "Right 4th Finger Middle Phalanx": [106321, 106511],
    "Right 2nd Finger Proximal Phalanx": [106511, 106791],
    "Right 5th Finger Proximal Phalanx": [106791, 107011],
    "Right 3rd Finger Proximal Phalanx": [107011, 107299],
    "Right 4th Finger Proximal Phalanx": [107299, 107575],
    "Right Thumb Proximal Phalanx": [107575, 107781],

    # "Right 2nd Finger Distal Phalanx": [107781, 107987],
    # "Right 5th Finger Distal Phalanx": [107987, 108231],
    # "Right 3rd Finger Distal Phalanx": [108231, 108445],
    # "Right 4th Finger Distal Phalanx": [108445, 108669],
    # "Right Thumb Distal Phalanx": [108669, 108885],

    # "Right Capitate": [108885, 109199],
    # "Right 5th Finger Metacarpal": [109199, 109473],
    # "Right 1st Finger Metacarpal": [109473, 109817],
    # "Right 4th Finger Metacarpal": [109817, 110141],
    # "Right Hamate": [110141, 110423],
    # "Right Lunate": [110423, 110613],
    # "Right Pisiform": [110613, 110943],
    # "Right Scaphold": [110943, 111125],
    # "Right 2nd Finger Metacarpal": [111125, 111511],
    # "Right 3rd Finger Metacarpal": [111511, 111931],
    # "Right Trapezium": [111931, 112161],
    # "Right Trapezoid": [112161, 112359],
    # "Right Triquetral": [112359, 112601],
    # "Right 2nd Finger Middle Phalanx": [112601, 112797],
    # "Right 5th Finger Middle Phalanx": [112797, 112999],
    # "Right 3rd Finger Middle Phalanx": [112999, 113187],
    # "Right 4th Finger Middle Phalanx": [113187, 113377],
    # "Right 2nd Finger Proximal Phalanx": [113377, 113657],
    # "Right 5th Finger Proximal Phalanx": [113657, 113877],
    # "Right 3rd Finger Proximal Phalanx": [113877, 114165],
    # "Right 4th Finger Proximal Phalanx": [114165, 114441],
    # "Right Thumb Proximal Phalanx": [114441, 114647],

    "Left Scapula" : [114647, 116621],
    "Left Humerus" : [116621, 118775],
    "Left Ulna" : [118775, 119331],
    "Left Radius" : [119331, 119799],

    # "Left Hand" : [119799, 126664],
    "Left 2nd Finger Distal Phalanx": [119799, 120005],
    "Left 5th Finger Distal Phalanx": [120005, 120249],
    "Left 3rd Finger Distal Phalanx": [120249, 120463],
    "Left 4th Finger Distal Phalanx": [120463, 120687],
    "Left Thumb Distal Phalanx": [120687, 120903],

    "Left Capitate": [120903, 121217],
    "Left 5th Finger Metacarpal": [121217, 121491],
    "Left 1st Finger Metacarpal": [121491, 121835],
    "Left 4th Finger Metacarpal": [121835, 122159],
    "Left Hamate": [122159, 122441],
    "Left Lunate": [122441, 122631],
    "Left Pisiform": [122631, 122961],
    "Left Scaphold": [122961, 123143],
    "Left 2nd Finger Metacarpal": [123143, 123529],
    "Left 3rd Finger Metacarpal": [123529, 123949],
    "Left Trapezium": [123949, 124179],
    "Left Trapezoid": [124179, 124377],
    "Left Triquetral": [124377, 124619],
    "Left 2nd Finger Middle Phalanx": [124619, 124815],
    "Left 5th Finger Middle Phalanx": [124815, 125017],
    "Left 3rd Finger Middle Phalanx": [125017, 125205],
    "Left 4th Finger Middle Phalanx": [125205, 125395],
    "Left 2nd Finger Proximal Phalanx": [125395, 125675],
    "Left 5th Finger Proximal Phalanx": [125675, 125895],
    "Left 3rd Finger Proximal Phalanx": [125895, 126183],
    "Left 4th Finger Proximal Phalanx": [126183, 126459],
    "Left Thumb Proximal Phalanx": [126459, 126665],
}

# glColor3f(1, 0, 0)
# glPushMatrix()
# # coccyx
# for i in range(322):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 1, 0)
# glPushMatrix()
# # Left Hip
# for i in range(322, 1636):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # pubic symphysis
# for i in range(1636, 1782):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 1, 0)
# glPushMatrix()
# # Right Hip
# for i in range(1782, 3094):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 1)
# glPushMatrix()
# # Sacrum
# for i in range(3094, 9046):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 1, 1)
# glPushMatrix()
# # Femur_R
# for i in range(9046, 10698):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0.5, 0)
# glPushMatrix()
# # Tibia_R
# for i in range(10698, 12380):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.5, 1, 0)
# glPushMatrix()
# # Talus_R
# for i in range(12380, 13258):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 1, 0.5)
# glPushMatrix()
# # calcaneus_R
# for i in range(13258, 18354):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0.5, 1)
# glPushMatrix()
# # toe_R
# for i in range(18354, 21488):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.5, 0, 1)
# glPushMatrix()
# # Femur_L
# for i in range(21488, 23140):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0.5)
# glPushMatrix()
# # Tibia_L
# for i in range(23140, 24822):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.5, 1, 1)
# glPushMatrix()
# # Talus_L
# for i in range(24822, 25700):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0.5, 1)
# glPushMatrix()
# # calcaneus_L
# for i in range(25700, 30796):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 1, 0.5)
# glPushMatrix()
# # toe_L
# for i in range(30796, 33930):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 0)
# glPushMatrix()
# # L5
# for i in range(33930, 35622):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.4, 0.4, 0.4)
# glPushMatrix()
# # L1
# for i in range(35622, 37594):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.1, 0.1, 0.1)
# glPushMatrix()
# # L4
# for i in range(37594, 39324):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.3, 0.3, 0.3)
# glPushMatrix()
# # L2
# for i in range(39324, 40958):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0.2, 0.2, 0.2)
# glPushMatrix()
# # L3
# for i in range(40958, 42838):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T8
# for i in range(42838, 43662):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T11
# for i in range(43662, 44616):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T9
# for i in range(44616, 46208):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T7
# for i in range(46208, 47724):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T6
# for i in range(47724, 48510):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T10
# for i in range(48510, 50178):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T12
# for i in range(50178, 51156):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T5
# for i in range(51156, 52592):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T1
# for i in range(52592, 53428):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T4
# for i in range(53428, 54706):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T2
# for i in range(54706, 55902):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(0, 0, 1)
# glPushMatrix()
# # T3
# for i in range(55902, 57054):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # sternum
# for i in range(57054, 59854):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left 8910 costal cartilage
# for i in range(59854, 60626):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 8
# for i in range(60626, 61102):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 11
# for i in range(61102, 61626):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 5
# for i in range(61626, 62420):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 1
# for i in range(62420, 63364):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 4
# for i in range(63364, 64116):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 9
# for i in range(64116, 64648):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 2
# for i in range(64648, 65460):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 7
# for i in range(65460, 66452):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 6
# for i in range(66452, 67482):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 10
# for i in range(67482, 67930):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 3
# for i in range(67930, 68668):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Left rib 12
# for i in range(68668, 69166):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Manubrium
# for i in range(69166, 69918):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right Costal Cartilage
# for i in range(69918, 70690):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 8
# for i in range(70690, 71166):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 11
# for i in range(71166, 71690):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 1
# for i in range(71690, 72484):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 1
# for i in range(72484, 73428):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 4
# for i in range(73428, 74180):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 9
# for i in range(74180, 74712):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 2
# for i in range(74712, 75524):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 7
# for i in range(75524, 76516):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 6
# for i in range(76516, 77546):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 10
# for i in range(77546, 77994):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 3
# for i in range(77994, 78732):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Right rib 12
# for i in range(78732, 79230):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Skull
# for i in range(79230, 88733):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Atlas, C1
# for i in range(88733, 89521):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # Axis, C2
# for i in range(89521, 91333):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # C5
# for i in range(91333, 92087):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # C7
# for i in range(92087, 93767):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # C6
# for i in range(93767, 94777):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # C4
# for i in range(94777, 95763):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # right scapula
# for i in range(95763, 97737):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # right humerus
# for i in range(97737, 99891):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # right ulna
# for i in range(99891, 100447):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # right radius
# for i in range(100447, 100915):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # right hand
# for i in range(100915, 114647):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # left scapula
# for i in range(114647, 116621):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # left humerus
# for i in range(116621, 118775):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # left ulna
# for i in range(118775, 119331):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # left radius
# for i in range(119331, 119799):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()

# glColor3f(1, 0, 0)
# glPushMatrix()
# # left hand
# for i in range(119799, 126664):
#     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]] + root_dif
#     glBegin(GL_TRIANGLES)
#     glVertex3f(v0[0], v0[1], v0[2])
#     glVertex3f(v1[0], v1[1], v1[2])
#     glVertex3f(v2[0], v2[1], v2[2])
#     glEnd()
# glPopMatrix()