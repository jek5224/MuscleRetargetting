import dartpy as dart
import xml.etree.ElementTree as ET
import numpy as np
import os

def MakeWeldJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint):
    joint_prop = dart.dynamics.WeldJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    return joint_prop

def MakeFreeJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint, damping):
    joint_prop = dart.dynamics.FreeJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = False
    joint_prop.mVelocityLowerLimits = np.ones(6) * -100
    joint_prop.mVelocityUpperLimits = np.ones(6) * 100
    joint_prop.mDampingCoefficients = np.ones(6) * damping
    return joint_prop

def MakeFixedFreeJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint, damping):
    joint_prop = dart.dynamics.FreeJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = True
    joint_prop.mPositionLowerLimits = np.ones(6) * -0.00001
    joint_prop.mPositionUpperLimits = np.ones(6) *  0.00001
    joint_prop.mVelocityLowerLimits = np.ones(6) * -100
    joint_prop.mVelocityUpperLimits = np.ones(6) *  100
    joint_prop.mDampingCoefficients = np.ones(6) * damping
    return joint_prop

def MakeRevoluteJointProperties(name, axis, T_ParentBodyToJoint, T_ChildBodyToJoint, lower, upper, damping, friction, stiffness):
    joint_prop = dart.dynamics.RevoluteJointProperties()
    joint_prop.mName = name
    joint_prop.mAxis = axis
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = True
    joint_prop.mPositionLowerLimits = np.ones(1) * lower
    joint_prop.mPositionUpperLimits = np.ones(1) * upper

    joint_prop.mVelocityLowerLimits = np.ones(1) * -100
    joint_prop.mVelocityUpperLimits = np.ones(1) * 100

    joint_prop.mForceLowerLimits = np.ones(1) * -10000.0
    joint_prop.mForceUpperLimits = np.ones(1) * 10000.0

    joint_prop.mDampingCoefficients = np.ones(1) * damping
    joint_prop.mFrictions = np.ones(1) * friction
    joint_prop.mSpringStiffnesses = np.ones(1) * stiffness
    return joint_prop

def MakeBallJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint, lower, upper, damping, friction, stiffness = np.zeros(3)):
    joint_prop = dart.dynamics.BallJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = True
    joint_prop.mPositionLowerLimits = lower
    joint_prop.mPositionUpperLimits = upper

    joint_prop.mVelocityLowerLimits = np.ones(3) * -100
    joint_prop.mVelocityUpperLimits = np.ones(3) * 100

    joint_prop.mForceLowerLimits = np.ones(3) * -10000.0
    joint_prop.mForceUpperLimits = np.ones(3) * 10000.0

    joint_prop.mDampingCoefficients = np.ones(3) * damping
    joint_prop.mFrictions = np.ones(3) * friction
    joint_prop.mSpringStiffnesses = stiffness
    return joint_prop

def MakeBodyNode(skel, parent, joint_properties, joint_type, intertia):
    if joint_type == "Free":
        [joint, body] = skel.createFreeJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "FixedFree":
        [joint, body] = skel.createFreeJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "Revolute":
        [joint, body] = skel.createRevoluteJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "Ball":
        [joint, body] = skel.createBallJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "Weld":
        [joint, body] = skel.createWeldJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))

    body.setInertia(intertia)
    return body

def Orthonormalize(T):

    v0 = T.rotation()[:, 0]
    v1 = T.rotation()[:, 1]
    v2 = T.rotation()[:, 2]

    u0 = v0
    u1 = v1 - np.dot(u0, v1) / np.dot(u0, u0) * u0
    u2 = v2 - np.dot(u0, v2) / np.dot(u0, u0) * u0 - np.dot(u1, v2) / np.dot(u1, u1) * u1

    res = np.zeros([3,3])
    res[:, 0] = u0 / np.linalg.norm(u0)
    res[:, 1] = u1 / np.linalg.norm(u1)
    res[:, 2] = u2 / np.linalg.norm(u2)

    T.set_rotation(res)
    return T

def saveSkeletonInfo(path=None, defaultDamping = 0.2):
    skel_info = {}

    if path is not None:
        doc = ET.parse(path)
        if doc is None:
            print("File not found")
            return None
        
    root = doc.getroot()
    root_name = root.attrib['name']

    bvh_info = {}
    smpl_info = {}
    joints_pd_gain = []
    mesh_info = {}

    for node in root:
        skel = {}
        # skel['name'] = node.attrib['name']
        name = node.attrib['name']
        parent = node.attrib['parent']
        skel['parent_str'] = parent

        body = node.find('Body')
        skel['type'] = body.attrib['type']
        skel['mass'] = float(body.attrib['mass'])

        obj = None
        if "obj" in body.attrib:
            obj = body.attrib['obj']
        skel['obj'] = obj

        if obj and obj != "None":
            if "Zygote" in obj:
                obj_path = obj
            else:
                obj_path = os.getcwd() + "/data/OBJ/" + obj
            mesh_info[name] = obj_path

        skel['size'] = np.array(body.attrib['size'].strip().split(' ')).astype(np.float32)

        ## contact
        skel['contact'] = body.attrib['contact'] == 'On'
        color = np.ones(4) * 0.2
        if 'color' in body.attrib:
            color = np.array(body.attrib['color'].split(' ')).astype(np.float32)
        skel['color'] = color

        # T_body = dart.math.Isometry3().Identity()
        # trans = body.find('Transformation')
        # T_body.set_rotation(np.array(trans.attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3))
        # T_body.set_translation(np.array(trans.attrib['translation'].strip().split(' ')).astype(np.float32))
        # T_body = Orthonormalize(T_body)
        # skel['T_body'] = T_body
        
        trans = body.find('Transformation')
        skel['body_r'] = np.array(trans.attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3)
        skel['body_t'] = np.array(trans.attrib['translation'].strip().split(' ')).astype(np.float32)
        
        joint = node.find("Joint")
        joint_type = joint.attrib['type']
        skel['joint_type'] = joint_type
        if 'bvh' in joint.attrib:
            bvh_info[name] = joint.attrib['bvh']
            skel['bvh'] = joint.attrib['bvh']

        if 'smpl_jidx' in joint.attrib:
            smpl_info[name] = int(joint.attrib['smpl_jidx'])
            skel['smpl_jidx'] = int(joint.attrib['smpl_jidx'])

        # T_joint = dart.math.Isometry3().Identity()
        # trans = joint.find('Transformation')
        # T_joint.set_rotation(np.array(trans.attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3))
        # T_joint.set_translation(np.array(trans.attrib['translation'].strip().split(' ')).astype(np.float32))
        # T_joint = Orthonormalize(T_joint)
        # skel['T_joint'] = T_joint

        trans = joint.find('Transformation')
        skel['joint_r'] = np.array(trans.attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3)
        skel['joint_t'] = np.array(trans.attrib['translation'].strip().split(' ')).astype(np.float32)

        if body.get('stretch') is not None:
            stretches = np.array(body.attrib['stretch'].strip().split(' ')).astype(np.int32)
            skel['stretches'] = stretches

            stretch_axises = []
            gaps = []

            for i in stretches:
                if i == 0:
                    axis = np.array([1, 0, 0])
                elif i == 1:
                    axis = np.array([0, 1, 0])
                elif i == 2:
                    axis = np.array([0, 0, 1])
                
                size = skel['size'][i]

                stretch_axis = skel['body_r'] @ axis
                if np.dot(stretch_axis, skel['body_t'] - skel['joint_t'] ) < 0:
                    stretch_axis = -stretch_axis

                stretch_axises.append(stretch_axis)
                gap = skel['body_t'] - (skel['joint_t'] + stretch_axis * size * 0.5)

                gaps.append(gap)

            skel['stretch_axises'] = stretch_axises
            skel['gaps'] = gaps

            '''
            <----- size ----->
            ------------------                             ------------------
            |                |                             |     parent     |
            |      body      |<- --gap ----O<--------------|      body      |    O    
            |                |           joint  gap_parent |                |  parent
            ------------------                             ------------------   joint

            '''

            if skel['parent_str'] != "None":
                parent_info = skel_info[skel['parent_str']]
                parent_stretches = parent_info['stretches']
                parent_joint_t = parent_info['joint_t']

                gaps_parent = []

                for i in range(len(parent_stretches)):
                    parent_stretch = parent_stretches[i]
                    parent_size = parent_info['size'][parent_stretch]

                    parent_stretch_axis = parent_info['stretch_axises'][i]
                    parent_gap = parent_info['gaps'][i]

                    gap_parent_cand1 = skel['joint_t'] - (parent_joint_t + parent_gap + parent_stretch_axis * parent_size)
                    gap_parent_cand2 = skel['joint_t'] - (parent_joint_t + parent_gap)

                    if np.linalg.norm(gap_parent_cand1) < np.linalg.norm(gap_parent_cand2):
                        gap_parent = gap_parent_cand1
                        same_direction_to_parent = True
                    else:
                        gap_parent = gap_parent_cand2
                        same_direction_to_parent = False

                    gaps_parent.append([gap_parent, same_direction_to_parent])

                skel['gaps_parent'] = gaps_parent

        if joint_type == "Free":
            damping = defaultDamping
            if 'damping' in joint.attrib:
                damping = float(joint.attrib['damping'])
            skel['damping'] = damping
        elif joint_type == "FixedFree":
            damping = defaultDamping
            if 'damping' in joint.attrib:
                damping = float(joint.attrib['damping'])
            skel['damping'] = damping
        elif joint_type == "Ball":
            skel['lower'] = np.array(joint.attrib['lower'].strip().split(' ')).astype(np.float32)
            skel['upper'] = np.array(joint.attrib['upper'].strip().split(' ')).astype(np.float32)

            damping = defaultDamping
            if 'damping' in joint.attrib:
                damping = float(joint.attrib['damping'])
            skel['damping'] = damping

            friction = 0.0
            if 'friction' in joint.attrib:
                friction = float(joint.attrib['friction'])
            skel['friction'] = friction

            stiffness = np.zeros(3)
            if 'stiffness' in joint.attrib:
                stiffness = np.array(joint.attrib['stiffness'].strip().split(' ')).astype(np.float32)
            skel['stiffness'] = stiffness
        elif joint_type == "Revolute":
            skel['axis'] = np.array(joint.attrib['axis'].strip().split(' ')).astype(np.float32)
            skel['lower'] = float(joint.attrib['lower'])
            skel['upper'] = float(joint.attrib['upper'])

            damping = defaultDamping
            if 'damping' in joint.attrib:
                damping = float(joint.attrib['damping'])
            skel['damping'] = damping

            friction = 0.0
            if 'friction' in joint.attrib:
                friction = float(joint.attrib['friction'])
            skel['friction'] = friction
            
            stiffness = 0.0
            if 'stiffness' in joint.attrib:
                stiffness = float(joint.attrib['stiffness'])
            skel['stiffness'] = stiffness
        elif joint_type == "Weld":
            pass
        else:
            print("Not implemented")
            return None
        
        pd_gain = None
        if 'pd_gain' in joint.attrib:
            pd_gain = [np.array(joint.attrib['pd_gain'].strip().split(' ')).astype(np.float32)]
            if 'kv' in joint.attrib:
                pd_gain.append(np.array(joint.attrib['kv'].strip().split(' ')).astype(np.float32))
            else:
                pd_gain.append(np.sqrt(pd_gain[0]) * 2)

        joints_pd_gain.append(pd_gain)
        
        skel_info[name] = skel

    children = {}
    for name in skel_info.keys():
        children[name] = []
    for name, info in skel_info.items():
        if info['parent_str'] != "None":
            children[info['parent_str']].append(name)
    for name in reversed(children.keys()):
        if len(children[name]) > 0:
            for child in children[name]:
                if len(children[child]) > 0:
                    for grandchild in children[child]:
                        if not grandchild in children[name]:
                            children[name].append(grandchild)

    for name in skel_info.keys():
        skel_info[name]['children'] = children[name]

    return skel_info, root_name, bvh_info, joints_pd_gain, mesh_info, smpl_info

# XML file to Skeleton
def buildFromFile(path = None, defaultDamping = 0.2):
    if path is not None:
        doc = ET.parse(path)
        # Error handling
        if doc is None:
            print("File not found")
            return None
        
        root = doc.getroot()
        skel = dart.dynamics.Skeleton(root.attrib['name'])

        bvh_info = {}

        for node in root:
            name = node.attrib['name']
            parent_str = node.attrib['parent']
            parent = None
            if parent_str != "None":
                parent = skel.getBodyNode(parent_str)
        
            type = node.find("Body").attrib['type']
            mass = float(node.find("Body").attrib['mass'])
            
            shape = None
            if type == "Box":
                size = np.array(node.find("Body").attrib['size'].strip().split(' ')).astype(np.float32)
                shape = dart.dynamics.BoxShape(size)
            else:
                print("Not implemented")
                return None
        
            ## contact
            contact = node.find("Body").attrib['contact'] == "On"
            color = np.ones(4) * 0.2
            if 'color' in node.find("Body").attrib:
                color = np.array(node.find("Body").attrib['color'].split(' ')).astype(np.float32)

            inertia = dart.dynamics.Inertia()
            inertia.setMoment(shape.computeInertia(mass))
            inertia.setMass(mass)
            
            T_body = dart.math.Isometry3().Identity()
            T_body.set_rotation(np.array(node.find("Body").find("Transformation").attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3))
            T_body.set_translation(np.array(node.find("Body").find("Transformation").attrib['translation'].strip().split(' ')).astype(np.float32))
            T_body = Orthonormalize(T_body)
            
            joint = node.find("Joint")
            type = joint.attrib['type']
            if 'bvh' in joint.attrib:
                bvh_info[name] = joint.attrib['bvh']

            T_joint = dart.math.Isometry3().Identity()
            T_joint.set_rotation(np.array(node.find("Joint").find("Transformation").attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3))
            T_joint.set_translation(np.array(node.find("Joint").find("Transformation").attrib['translation'].strip().split(' ')).astype(np.float32))
            T_joint = Orthonormalize(T_joint)
            
            parent_to_joint = T_joint
            if parent != None:
                parent_to_joint = parent.getTransform().inverse().multiply(T_joint)
            
            child_to_joint = T_body.inverse().multiply(T_joint)

            props = None
            if type == "Free":
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                props = MakeFreeJointProperties(name, parent_to_joint, child_to_joint, damping)
            elif type == "FixedFree":
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                props = MakeFixedFreeJointProperties(name, parent_to_joint, child_to_joint, damping)
            elif type == "Ball":
                lower = np.array(joint.attrib['lower'].strip().split(' ')).astype(np.float32)
                upper = np.array(joint.attrib['upper'].strip().split(' ')).astype(np.float32)
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                friction = 0.0
                if 'friction' in joint.attrib:
                    friction = float(joint.attrib['friction'])
                stiffness = np.zeros(3)
                if 'stiffness' in joint.attrib:
                    stiffness = np.array(joint.attrib['stiffness'].strip().split(' ')).astype(np.float32)
                props = MakeBallJointProperties(name, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness)
            elif type == "Revolute":
                axis = np.array(joint.attrib['axis'].strip().split(' ')).astype(np.float32)
                lower = float(joint.attrib['lower'])
                upper = float(joint.attrib['upper'])
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                friction = 0.0
                if 'friction' in joint.attrib:
                    friction = float(joint.attrib['friction'])
                stiffness = 0.0
                if 'stiffness' in joint.attrib:
                    stiffness = float(joint.attrib['stiffness'])
                props = MakeRevoluteJointProperties(name, axis, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness)
            elif type == "Weld":
                props = MakeWeldJointProperties(name, parent_to_joint, child_to_joint)
            else:
                print("Joint Prop Not implemented")
                return None
            
            bn = MakeBodyNode(skel, parent, props, type, inertia)
            shape_node = bn.createShapeNode(shape)
            shape_node.createVisualAspect().setColor(color)
            shape_node.createDynamicsAspect()
            if contact:    
                shape_node.createCollisionAspect()

        return skel, bvh_info
    
# skel info to Skeleton
def buildFromInfo(skel_info, root_name):
    skel = dart.dynamics.Skeleton(root_name)

    for name, info in skel_info.items():
        parent = None
        parent_str = info['parent_str']
        if parent_str != "None":
            parent = skel.getBodyNode(parent_str)

        type = info['type']
        shape = None
        if type == "Box":
            size = info['size']
            shape = dart.dynamics.BoxShape(size)
        else:
            print("Not implemented")
            return None
    
        ## contact
        contact = info['contact']
        color = info['color']
        if 'obj' in info:
            obj = info['obj']
        else:
            obj = None

        mass = info['mass']
        inertia = dart.dynamics.Inertia()
        inertia.setMoment(shape.computeInertia(mass))
        inertia.setMass(mass)
        
        T_body = dart.math.Isometry3().Identity()
        body_r = info['body_r']
        body_t = info['body_t']
        T_body.set_rotation(body_r)
        T_body.set_translation(body_t)
        T_body = Orthonormalize(T_body)

        T_joint = dart.math.Isometry3().Identity()
        joint_r = info['joint_r']
        joint_t = info['joint_t']
        T_joint.set_rotation(joint_r)
        T_joint.set_translation(joint_t)
        T_joint = Orthonormalize(T_joint)

        joint_type = info['joint_type']  
        
        parent_to_joint = T_joint
        if parent != None:
            parent_to_joint = parent.getTransform().inverse().multiply(T_joint)
        
        child_to_joint = T_body.inverse().multiply(T_joint)

        props = None
        if joint_type == "Free":
            damping = info['damping']
            props = MakeFreeJointProperties(name, parent_to_joint, child_to_joint, damping)
        elif joint_type == "FixedFree":
            damping = info['damping']
            props = MakeFixedFreeJointProperties(name, parent_to_joint, child_to_joint, damping)
        elif joint_type == "Ball":
            lower = info['lower']
            upper = info['upper']
            damping = info['damping']
            friction = info['friction']
            stiffness = info['stiffness']
            friction = 0.0

            props = MakeBallJointProperties(name, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness)
        elif joint_type == "Revolute":
            axis = info['axis']
            lower = info['lower']
            upper = info['upper']
            damping = info['damping']
            friction = info['friction']
            stiffness = info['stiffness']

            props = MakeRevoluteJointProperties(name, axis, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness)
        elif joint_type == "Weld":
            props = MakeWeldJointProperties(name, parent_to_joint, child_to_joint)
        else:
            print("Not implemented")
            return None
        
        bn = MakeBodyNode(skel, parent, props, joint_type, inertia)
        shape_node = bn.createShapeNode(shape)
        shape_node.createVisualAspect().setColor(color)
        shape_node.createDynamicsAspect()
        if contact:    
            shape_node.createCollisionAspect()
            
    return skel

def exportSkeleton(skel_info, root_name, filename):
    def tw(file, string, tabnum):
        for _ in range(tabnum):
            file.write("\t")
        file.write(string + "\n")

    f = open(f"data/{filename}", 'w')
    tw(f, "<Skeleton name=\"%s\">" % (root_name), 0)

    for name, info in skel_info.items():
        tw(f, "<Node name=\"%s\" parent=\"%s\">" % (name, info['parent_str']), 1)

        tw(f, "<Body type=\"%s\" mass=\"%f\" size=\"%s\" contact=\"%s\" color=\"%s\" obj=\"%s\" stretch=\"%s\">" % 
           (info['type'], 
            info['mass'], 
            " ".join(info['size'].astype(str)), 
            "On" if info['contact'] else "Off", 
            " ".join(info['color'].astype(str)), 
            info['obj'],
            " ".join([str(i) for i in info['stretches']]),
            ), 
            2)
        
        tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" % 
           (" ".join(info['body_r'].astype(str).flatten()), 
            " ".join(info['body_t'].astype(str))
            ), 
            3)

        tw(f, "</Body>", 2)

        if info['joint_type'] == "Free":
            if 'bvh' in info:
                tw(f, "<Joint type=\"Free\" bvh=\"%s\" smpl_jidx=\"%s\">" % 
                   (info['bvh'], info['smpl_jidx']), 2)
            else:
                tw(f, "<Joint type=\"Free\" smpl_jidx=\"%s\">" % 
                   (info['smpl_jidx']), 2)
        elif info['joint_type'] == "Ball":
            if 'bvh' in info:
                tw(f, "<Joint type=\"Ball\" bvh=\"%s\" lower=\"%s\" upper=\"%s\" smpl_jidx=\"%s\">" %
                (info['bvh'],
                    " ".join(info['lower'].astype(str)),
                    " ".join(info['upper'].astype(str)),
                    info['smpl_jidx']
                    ),
                    2)
            else:
                tw(f, "<Joint type=\"Ball\" lower=\"%s\" upper=\"%s\" smpl_jidx=\"%s\">" %
                (" ".join(info['lower'].astype(str)),
                    " ".join(info['upper'].astype(str)),
                    info['smpl_jidx']
                    ),
                    2)
        elif info['joint_type'] == "Revolute":
            if 'bvh' in info:
                tw(f, "<Joint type=\"Revolute\" bvh=\"%s\" axis=\"%s\" lower=\"%s\" upper=\"%s\" smpl_jidx=\"%s\">" %
                (info['bvh'],
                    " ".join(info['axis'].astype(str)),
                    info['lower'],
                    info['upper'],
                    info['smpl_jidx']
                    ),
                    2)
            else:
                tw(f, "<Joint type=\"Revolute\" axis=\"%s\" lower=\"%s\" upper=\"%s\" smpl_jidx=\"%s\">" %
                (" ".join(info['axis'].astype(str)),
                    info['lower'],
                    info['upper'],
                    info['smpl_jidx']
                    ),
                    2)
        
        tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" %
                (" ".join(info['joint_r'].astype(str).flatten()),
                " ".join(info['joint_t'].astype(str))),
                3)
        
        tw(f, "</Joint>", 2)

        tw(f, "</Node>", 1)
    tw(f, "</Skeleton>", 0)
    f.close()

def exportBoundingBoxes(skeleton_meshes, root_name='Skeleton', filename='zygote_skel.xml'):
    def tw(file, string, tabnum):
        for _ in range(tabnum):
            file.write("\t")
        file.write(string + "\n")

    from scipy.spatial.transform import Rotation as R

    def writeBoundingBoxes(name, mesh):
        for i in range(len(mesh.corners)):
            if i == 0:
                if mesh.is_root:
                    parent_name = None
                else:
                    parent_name = mesh.parent_name + "0"
                    # if len(mesh.parent_mesh.corners) > 1:
                    #     parent_name = parent_name + "0"
                obj = mesh.obj
            else:
                parent_name = name + "0"
                obj = None
            tw(f, "<Node name=\"%s\" parent=\"%s\">" % (name+str(i), parent_name), 1)

            # tw(f, "<Body type=\"%s\" mass=\"%f\" size=\"%s\" contact=\"%s\" color=\"%s\" obj=\"%s\" stretch=\"%s\">" % 
            #    (info['type'], 
            #     info['mass'], 
            #     " ".join(info['size'].astype(str)), 
            #     "On" if info['contact'] else "Off", 
            #     " ".join(info['color'].astype(str)), 
            #     info['obj'],
            #     " ".join([str(i) for i in info['stretches']]),
            #     ), 
            #     2)

            contact = "On" if mesh.is_contact else "Off"
            # sacrum 0.5 kg for bounding box with size (0.11064, 0.134131, 0.059748)
            mass = np.prod(mesh.sizes[i]) * 0.5 / (0.11064 * 0.134131 * 0.059748)
            tw(f, "<Body type=\"%s\" mass=\"%f\" size=\"%s\" contact=\"%s\" color=\"%s %s\" obj=\"%s\">" % 
                ('Box', 
                np.round(mass, 6), 
                " ".join(np.round(mesh.sizes[i], 6).astype(str)), 
                contact, 
                " ".join(mesh.color.astype(str)), 
                mesh.transparency,
                obj,
                ), 
                2)
            
            tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" % 
                (" ".join(np.round(mesh.body_rs[i], 6).astype(str).flatten()), 
                " ".join(np.round(mesh.body_ts[i], 6).astype(str)),
                ), 
                3)
            
            tw(f, "</Body>", 2)

            joint_type = 'Free' if (mesh.is_root and i == 0) else 'Ball'#'Ball'
            if joint_type == "Free":
                tw(f, "<Joint type=\"Free\">", 2)
            elif joint_type == "Ball":
                if i > 0 or mesh.is_weld:
                    joint_type = 'Weld'
                    # axis = np.array([1.001, 0.001, 0.001,])

                    # tw(f, "<Joint type=\"Revolute\" axis=\"%s\" lower=\"-0.00001\" upper=\"0.00001\">" %
                    #         (" ".join(axis.astype(str))),
                    #         2)
                    tw(f, "<Joint type=\"Weld\">", 2)
                else:
                    lower = np.array([np.round(-np.pi/2, 2), 
                                      np.round(-np.pi/2, 2), 
                                      np.round(-np.pi/2, 2)])
                    upper = np.array([np.round(np.pi/2, 2), 
                                      np.round(np.pi/2, 2), 
                                      np.round(np.pi/2, 2)])
                
                    tw(f, "<Joint type=\"Ball\" lower=\"%s\" upper=\"%s\">" %
                            (" ".join(lower.astype(str)),
                            " ".join(upper.astype(str)),
                            ),
                            2)
            elif joint_type == "Weld":
                tw(f, "<Joint type=\"Weld\">", 2)

            # elif joint_type == "Revolute":
            #     if 'bvh' in info:
            #         tw(f, "<Joint type=\"Revolute\" bvh=\"%s\" axis=\"%s\" lower=\"%s\" upper=\"%s\" smpl_jidx=\"%s\">" %
            #         (info['bvh'],
            #             " ".join(info['axis'].astype(str)),
            #             info['lower'],
            #             info['upper'],
            #             info['smpl_jidx']
            #             ),
            #             2)
            #     else:
            #         tw(f, "<Joint type=\"Revolute\" axis=\"%s\" lower=\"%s\" upper=\"%s\" smpl_jidx=\"%s\">" %
            #         (" ".join(info['axis'].astype(str)),
            #             info['lower'],
            #             info['upper'],
            #             info['smpl_jidx']
            #             ),
            #             2)
            
            joint_r = np.eye(3)
            if joint_type == 'Free':
                tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" %
                    (" ".join(joint_r.astype(str).flatten()),
                    " ".join(np.round(np.mean(mesh.vertices, axis=0), 6).astype(str))),
                    3)
            else:
                if i > 0:
                    joint_t = mesh.weld_joints[i - 1]
                else:
                    joint_t = mesh.joint_to_parent
                tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" %
                        (" ".join(joint_r.astype(str).flatten()),
                        " ".join(np.round(joint_t, 6).astype(str))),
                        3)
            
            tw(f, "</Joint>", 2)

            tw(f, "</Node>", 1)

        for child_name in mesh.children_names:
            child_mesh = skeleton_meshes[child_name]
            writeBoundingBoxes(child_name, child_mesh)

    root_mesh = None
    for name, mesh in skeleton_meshes.items():
        if mesh.is_root:
            root_name = name
            root_mesh = mesh
            break

    if root_mesh is None:
        print("xml file not saved) No root mesh found")
        return

    f = open(f"data/{filename}", 'w')

    tw(f, "<Skeleton name=\"%s\">" % (root_name), 0)
    writeBoundingBoxes(root_name, root_mesh)

    tw(f, "</Skeleton>", 0)
    f.close()
    return

def exportMuscleWaypoints(muscle_meshes, skeleton_names, filename='zygote_muscle.xml'):
    def tw(file, string, tabnum):
        for _ in range(tabnum):
            file.write("\t")
        file.write(string + "\n")

    f0 = 1000.000000
    lm = 1.200000
    lt = 0.200000
    pen_angle = 0.000000
    lmax = -0.100000

    # body = "Sacrum0"
    with open(f"data/{filename}", "w") as file:
        tw(file, "<Muscle>", 0)
        
        for name, muscle in muscle_meshes.items():
            if len(muscle.waypoints) == 0:
                continue
            
            body = "Sacrum0"
            tw(file, f"<Unit name=\"{name}\" f0=\"{f0}\" lm=\"{lm}\" lt=\"{lt}\" pen_angle=\"{pen_angle}\" lmax=\"{lmax}\">", 1)
            for group_i, waypoint_group in enumerate(muscle.waypoints):
                # for waypoint in waypoint_group:
                # tw(file, f"<Waypoint body=\"{body}\" p=\"{' '.join([str(np.round(p, 6)) for p in fiber])}\"/>", 2)

                origin = skeleton_names[muscle.attach_skeletons[group_i][0]] + str(muscle.attach_skeletons_sub[group_i][0])
                insertion = skeleton_names[muscle.attach_skeletons[group_i][1]] + str(muscle.attach_skeletons_sub[group_i][1])
                for i in range(len(waypoint_group[0])):
                    tw(file, "<Fiber>", 2)    
                    for waypoints_i in range(len(waypoint_group)):
                        # if waypoints_i < len(waypoint_group) / 2:
                        #     body = origin
                        # else:
                        #     body = insertion
                        if waypoints_i == 0:
                            body = origin
                        else:
                            body = insertion
                        tw(file, f"<Waypoint body=\"{body}\" p=\"{' '.join(np.round(waypoint_group[waypoints_i][i], 6).astype(str))}\"/>", 3)
                    tw(file, "</Fiber>", 2)
            tw(file, "</Unit>", 1)
        tw(file, "</Muscle>", 0)