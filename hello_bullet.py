import pybullet as p
import time
import pybullet_data
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)


cubeOrn = R.from_euler('xyz', [0, 0, 0], degrees=True)
cubeOrnVec = cubeOrn.as_rotvec()
distance = 0.75
cubeStartPos = cubeOrnVec * distance
cubeStartOrientation = cubeOrn.as_quat()
robotId = p.loadURDF("hand_urdf.urdf",cubeStartPos, cubeStartOrientation, 
                   # useMaximalCoordinates=1, ## New feature in Pybullet
                   flags=p.URDF_USE_INERTIA_FROM_FILE)

for i in range (10000):
    p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, targetPosition=1)
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()

