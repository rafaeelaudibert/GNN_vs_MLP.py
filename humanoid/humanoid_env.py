import gym
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import math
import os
from gym import error, spaces, utils
from gym.utils import seeding

# Enums
X, Y, Z = range(3)

# Constants
FORCE_FACTOR = 0.082
MOTOR_NAMES =  ["abdomen_z", "abdomen_y", "abdomen_x", \
                    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee", \
                    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", \
                    "right_shoulder1", "right_shoulder2", "right_elbow", \
                    "left_shoulder1", "left_shoulder2", "left_elbow"]
MOTOR_POWERS = [100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25]
LINEAR_VEL_MAX = 4
 
class HumanoidEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50
  }
 
  def __init__(self, envStepCounterLimit = 2_048, connection=p.DIRECT):
    """
      HumanoidEnv initialization method, which creates the self.action_space and the self.observation_space variables,
      as well as connect to the pyBullet physics client
    """
    self.action_space = spaces.Box(-2, +2, (17, ))          # Actions can be exerced in the [-2, 2] range
    self.observation_space = spaces.Box(-10, +10, (43, ))   # Observations will be taken in a [-10, 10] range

    self._observation = []                                  # Array which stores the observations
    self._envStepCounterLimit = envStepCounterLimit         # Maximum frames which the env will try to run, before it's done (default 10_000)

    self.physicsClient = p.connect(connection)                   # Create the pyBullet physics client
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Used to load some default models from pyBullet (such as the field)

    self._seed()                                            # Configure the seed (random number generator)
 
  def step(self, action):
    """
      Method which is called whenever a new action should be taken by the HumanoidEnv
    """
    self._assign_throttle(action)
    p.stepSimulation()
    self._observation = self._compute_observation()
    reward = self._compute_reward()
    done = self._compute_done()
 
    self._envStepCounter += 1
    return np.array(self._observation), reward, done, {}

  def reset(self):
    self.initial_position = None
    self._actions = np.zeros((17,))
    self._envStepCounter = 0
  
    p.resetSimulation()
    p.setGravity(0, 0, -10) # m/s^2
    p.setTimeStep(0.01) # sec
  
    self.plane = p.loadURDF("plane.urdf")
    self.botId = p.loadMJCF(os.path.join(os.path.abspath(os.path.dirname(__file__)), "humanoid.xml"),
                            flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0]

    self.ordered_joints = []
    self.ordered_joint_indices = []
    joint_dict = {}
    for j in range(p.getNumJoints(self.botId)):

        # Grab JointInformation
        info = p.getJointInfo(self.botId, j)
        link_name = info[12].decode("ascii")
        self.ordered_joint_indices.append(j)

        # Configure foots (used for collision with the terrain)
        if link_name == "left_foot": self.left_foot_index = j
        if link_name == "right_foot": self.right_foot_index = j

        if info[2] == p.JOINT_REVOLUTE:
            joint_name = info[1].decode("ascii")
            joint_dict[joint_name] = j
            self.ordered_joints.append( (j, info[8], info[9]) ) # (Index, Joint Lower Limit, Joint Higher Limit)
            
            # Reset forces
            p.setJointMotorControl2(self.botId, j, controlMode=p.VELOCITY_CONTROL, force=0)

    self._motors_power = MOTOR_POWERS
    self._motors_index = [joint_dict[name] for name in MOTOR_NAMES]
  
    self._observation = self._compute_observation()
    return np.array(self._observation)

  # PyBullet handles this
  def _render(self, mode='human', close=False):
    pass
  
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]  


  ####
  ## Helpers ##
  def _assign_throttle(self, actions):
      actions = [self._motors_power[idx] * action * FORCE_FACTOR for idx, action in enumerate(actions)]
      self._actions = np.clip(self._actions + actions, -5, 5)
      p.setJointMotorControlArray(self.botId, self._motors_index, controlMode=p.TORQUE_CONTROL, forces=self._actions)

  def _compute_observation(self):
    jointStates = p.getJointStates(self.botId, self.ordered_joint_indices)
    joints = np.array([self._current_relative_position(jointStates, *jtuple) for jtuple in self.ordered_joints]).flatten()
    
    rcont = p.getContactPoints(self.botId, -1, self.right_foot_index, -1)
    lcont = p.getContactPoints(self.botId, -1, self.left_foot_index, -1)
    feet_contact = np.array([len(rcont) > 0, len(lcont) > 0])

    body_xyz, body_quaternion = p.getBasePositionAndOrientation(self.botId)
    (qx, qy, qz) = p.getEulerFromQuaternion(body_quaternion)
    if self.initial_position == None:
        self.initial_position = body_xyz

    (vx, vy, vz), _ = p.getBaseVelocity(self.botId)        
    extra_info = np.array([body_xyz[Z] - self.initial_position[Z], 0.1*vx, 0.1*vy, 0.1*vz, qx, qy, qz])

    return np.clip(np.concatenate([joints] + [feet_contact] + [extra_info]), -10, +10)

  def _current_relative_position(self, jointStates, j, lower, upper):
    state  = jointStates[j]
    pos, vel = state[0], state[1]
    pos_mid = 0.5 * (lower + upper)

    return (2 * (pos - pos_mid) / (upper - lower), 0.1 * vel)



  def _compute_reward(self):
    cubePos, quaternOrn = p.getBasePositionAndOrientation(self.botId)
    cubeOrn = p.getEulerFromQuaternion(quaternOrn)
    linearVel, _ = p.getBaseVelocity(self.botId)

    self._reward = min(linearVel[X], LINEAR_VEL_MAX) - (0.005 * (linearVel[X] ** 2 + linearVel[Y] ** 2)) - (0.05 * cubePos[Y] ** 2) \
                    - (0.02 * sum([c ** 2 for c in cubeOrn])) + 0.02 
    return self._reward

  def _compute_done(self):
    cubePos, _ = p.getBasePositionAndOrientation(self.botId)

    # Enough steps or below ground, or too high
    done = self._envStepCounter >= self._envStepCounterLimit or cubePos[Z] < -0.1 or cubePos[Z] > 3 
    return done 
