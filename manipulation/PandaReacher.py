
from os.path import dirname, join, abspath
import numpy as np
from gym import spaces
import math
import roboticstoolbox as rtb
import pdb
import random
import spatialmath as sm
import spatialgeometry as sg
import swift

SCENE_FILE = join(dirname(abspath(__file__)), 'scene_reinforcement_learning_env_custom.ttt')
POS_MIN, POS_MAX = [0.2, -0.7, 0.1], [0.6, 0.7, 0.7]
ANG_MIN, ANG_MAX = [50, 50, 5], [120,120,60]


class ReacherEnv(object):

    def __init__(self, max_episode_steps, dense, **kwargs):
        self.env = swift.Swift()
        self.env.launch()
        #self.agent = Panda()
        self.panda = rtb.models.Panda()
        self.panda_utils = rtb.models.Panda()
        # self.gripper = PandaGripper()
        #self.agent.set_control_loop_enabled(False)
        #self.agent.set_motor_locked_at_zero_velocity(True)

        self.target = sg.Sphere(radius=0.02, pose=sm.SE3(-0.6, -0.2, 0.0))
        self.target_goal = self.panda.fkine(self.panda.q)
        self.target_goal.A[:3, 3] = self.target.T[:3, -1]

        self.goal_axes = sg.Axes(length=.1, base=self.target_goal)
        # todo: get end effector position
        self.panda_ee_tip = self.panda.q[:3]
        #self.agent_ee_tip = self.agent.get_tip()
        self.panda.q = self.panda.qr

        self.env.add(self.panda)
        self.env.add(self.target)
        self.env.add(self.goal_axes)

        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1, 1, 1]), dtype=np.float64)
        self.action_limits = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61])
        self.observation_space = spaces.Box(low=-1, high=1, shape=[20])
        self.nSubtasks = 1
        self.eps_since_reset = 0
        self._max_episode_steps = max_episode_steps
        self.n = 7
        self.qlim = np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                            [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]])
        
        self.rang = self.qlim[1, :] - self.qlim[0, :]
        self.reward_dense = dense


    def _reinit(self):
        self.env.close()
        self.env = swift.Swift()
        self.env.launch()

        self.panda.q = self.panda.qr
        # todo: get end effector position

        self.panda_ee_tip = self.panda.q[:3]
        self.initial_joint_positions = self.panda.q
        self.eps_since_reset = 0

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # todo: get end effector position
        e_pos = self.panda.fkine(self.panda.q).A[:3, -1]

        t_pos = self.target_goal.A[:3, -1]
        return np.concatenate([self.panda.q,
                               self.panda.qd,
                               e_pos,
                               t_pos-e_pos])

    def reset(self):
        # remove current target marker
        self.env.remove(self.target)
        self.env.remove(self.goal_axes)

        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        rot = list(np.deg2rad(np.random.uniform(ANG_MIN, ANG_MAX)))

        # set pose for new target
        self.target_goal.A[:3, -1] = pos
        rot = sm.SE3.RPY(rot)

        self.target_goal.A[:3, :3] = rot.A[:3, :3]

        self.panda.q = self.panda.qr

        # set new target marker
        self.target = sg.Sphere(radius=0.02, pose=sm.SE3(pos))
        self.goal_axes = sg.Axes(length=.1, pose = sm.SE3(pos))
        self.env.add(self.target)
        self.env.add(self.goal_axes)

        self.ep_step = 0
        self.eps_since_reset += 1
        if self.eps_since_reset >= 100:
            self._reinit()
        return self._get_state()

    def gen_reward_dense(self):
        # Reward is negative distance to target
        dist = self.get_dist()
        reward = math.exp(-3*dist)
        return reward
    
    def gen_reward_sparse(self):
    
        dist = self.get_dist()
        joint_config = self.panda.q
        reward_manip = self.panda.manipulability(joint_config)
        if dist < 0.03:
            reward = 1 + (reward_manip * 10)
        else:
            reward = reward_manip
        return reward

    def get_dist(self):
        ePos = self.panda_ee_tip
        tPos = self.target.T[:3,-1]
        # Reward is negative distance to target
        dist = np.linalg.norm(ePos-tPos)
        return dist

    def step(self, action):
        # fixme: this probably won't work
        self.panda.qd = action # Execute action on arm
        # self.agent.set_joint_target_velocities(action)  # Execute action on arm
        # self.pr.step()  # Step the physics simulation
        self.env.step() # Step the physics simulation

        self.ep_step += 1
        
        # Generate reward
        reward = self.gen_reward_sparse()

        done = True if (self.ep_step >= self._max_episode_steps) else False

        info = {"reward0":reward,
                "overallReward":reward,
                "subtask":0,
                "success":1 if reward > 0.95 else 0}

        return self._get_state(), reward, done, info

    def _rand_q(self, k=0.2):
        q = np.zeros(self.n)
        for i in range(self.n):
            off = k * self.rang[i]
            q[i] = random.uniform(self.qlim[0, i] + off, self.qlim[1, i] - off)

        return q
        

    def _find_pose(self):
        q = self._rand_q()
        return self.panda_utils.fkine(q)

    def _check_limit(self):
        off = 0.2

        joint_config = self.panda.q


        for i in range(7):
            if joint_config[i] <= (self.qlim[0, i] + off):
                print('Joint limit hit')
                return True
            
            elif joint_config[i] >= (self.qlim[1, i] - off):
                print('Joint limit hit')
                return True

        return False
    
    def shutdown(self):
        self.env.close()


    def seed(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        self.action_space.np_random.seed(seed)
        return


    
class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


EPISODES = 50
EPISODE_LENGTH = 200

if __name__ == '__main__':
    env = ReacherEnv(headless=False, max_ep_steps=1000)
    agent = Agent()
    replay_buffer = []

    for e in range(EPISODES):

        print('Starting episode %d' % e)
        state = env.reset()
        for i in range(EPISODE_LENGTH):
            action = agent.act(state)
            
            reward, next_state,_,_ = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            state = next_state
            agent.learn(replay_buffer)

    print('Done!')
    env.shutdown()