import dartpy as dart
import xml.etree.ElementTree as ET
import numpy as np

def MakeWeldJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint):
    joint_prop = dart.dynamics.getWeldJointProperties()
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

    for node in root:
        skel = {}
        # skel['name'] = node.attrib['name']
        name = node.attrib['name']
        parent = node.attrib['parent']
        skel['parent_str'] = parent

        body = node.find('Body')
        skel['type'] = body.attrib['type']
        skel['mass'] = float(body.attrib['mass'])

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

                    gap_parent = skel['joint_t'] - (parent_joint_t + parent_gap + parent_stretch_axis * parent_size)
                    gaps_parent.append(gap_parent)

                skel['gaps_parent'] = gaps_parent

        if joint_type == "Free":
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
        else:
            print("Not implemented")
            return None
        
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

    # print(children)
    for name in skel_info.keys():
        skel_info[name]['children'] = children[name]
    
    return skel_info, root_name, bvh_info

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
                print("Not implemented")
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