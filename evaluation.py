
import gym
import matplotlib.pyplot as plt
import pickle

import os
import torch
from torch.autograd import Variable
from policy import Policy
import arguements

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape  # modify scene
from pyrep.const import PrimitiveShape
import numpy as np
import time



def init_pos(pr):
	agent = UR10()
	ee_init_pos = np.array([1, 0.2, 1])
	# Get a path to the target (rotate so z points down)

	path = agent.get_path(
		position=ee_init_pos, euler=[-2.2, 3.1415,
									 0])  # generate path given position and euler angles. NOTE: sometime the end-eff knock over the obj, why?

	done = False
	while not done:
		done = path.step()  # how does step works?
		pr.step()

	target = Shape.create(type=PrimitiveShape.CUBOID,  # the cuboid
						  size=[0.05, 0.05, 0.4],
						  mass=0.1,
						  smooth=False,
						  color=[1.0, 0.1, 0.1],
						  static=False, respondable=True)
	target.set_position(np.array([1.0, 0.2, 1.0]))  # initial position of the target

	time.sleep(0.5)

	return agent, target


def move(dy, dz, omega, ee_pos, ee_orient, pr, agent):
	# print('omega:', omega)
	# ee's x,y,z of the next step --
	ee_pos[1] += dy
	ee_pos[2] += dz
	# ee's orientation of the next step --
	ee_orient[0] += omega

	ee_pos[1] = np.clip(ee_pos[1], 0.2, 0.7)  # position limit
	ee_pos[2] = np.clip(ee_pos[2], 1.0, 1.5)
	ee_orient[0] = np.clip(ee_orient[0], 0.8, 2.8)  # orientation limit

	'''normally it won't get to the desired point'''

	new_joint_angles = agent.solve_ik(ee_pos, euler=ee_orient)  # get the joint angles of the robot by doing IK --

	# agent.set_joint_target_velocities([1, 1, 1, 1, 1, 1])   # not sure how to use this --?

	agent.set_joint_target_positions(new_joint_angles)  # set the joint angles as the result of IK above

	pr.step()  # Step the physics simulation

	# get the actual  position and orientation of the ee after pr.step()
	ee = agent.get_tip()
	ee_pos = ee.get_position()
	ee_orient = ee.get_orientation()

	return ee_pos, ee_orient, new_joint_angles


def is_stable(ee, target):
	ee_pos_0 = ee.get_position()
	tar_pos = target.get_position()
	pos_shift = np.linalg.norm(ee_pos_0 - tar_pos)

	ee_orient = ee.get_orientation()
	tar_orient = target.get_orientation()
	orient_shift = abs(ee_orient[0] - tar_orient[0] - 0.9)

	if pos_shift < 0.3 and orient_shift < 0.2:
		return True
	else:
		return False


def get_reward(fl, target, ee, args):

	# fl: 0 fall, 1 stay, 2 success
	'''
		3.Rewards
        r1: time spent penalty: -1
        r2: if the obj falls, ends the simulation. -20;
			if stay on, (z-1)*100 + 20;
			if the obj reach at the goal stably. +100
	'''

	'''
    ee-->end effector
    start ee pos: [1, 0.2, 1.0]
    ee goal pos: [1, 0.7 1.5]
    '''
	tar_pos = target.get_position()

	r1 = -1

	if fl == 0:
		r2 = -20
	elif fl == 1:
		r2 = (tar_pos[2] - 1.0) * 100 + 20  # 20 for not falling,
	else:
		r2 = 1000

	# print('r2:', r2)
	r = r1 + r2

	# easy reward --------------------------
	# if fl == 0:
	# 	r = -20
	# if fl == 2:
	# 	r = 100

	# r3 = np.linalg.norm((tar_pos[1:] - np.array([0.7, 1.5]))) * 10
	# r3 = (tar_pos[2] - 1.0) * 100

	return r


def sample(policy):
	args = arguements.achieve_args()

	# SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_003.ttt')
	SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_002.ttt')
	pr = PyRep()
	pr.launch(SCENE_FILE, headless=False)  # lunch the ttt file
	pr.start()

	agent = UR10()

	starting_joint_positions = [-1.5547982454299927, -0.11217942088842392, 2.505795478820801, 0.7483376860618591,
								1.587110161781311, 4.083085536956787]  # these angles correspond to [1.0, 0.2, 1.0]

	agent.set_joint_positions(starting_joint_positions)

	# agent.set_control_loop_enabled(False)
	agent.set_motor_locked_at_zero_velocity(True)

	# ee_pos = np.array([1.0, 0.2, 1.0])
	success_num = 0


	traj_num = 0
	avg_reward = []
	while traj_num < 20:

		agent, target = init_pos(pr)  # init agent and target

		ee = agent.get_tip()
		ee_pos = ee.get_position()
		ee_orient = ee.get_orientation()
		# print('initial_ee_pos:', ee_pos)
		# print('initial_ee_orient:', ee_orient)

		traj_reward = 0
		traj_num += 1
		for i in range(100):  # 100 steps max
			# print('step:', i)

			y = ee_pos[1]

			z = ee_pos[2]
			# print('y:', y)

			# action = policy.select_action(Variable(torch.Tensor([y]).unsqueeze(0)))[0]   # add noise to actuib
			action = policy(Variable(torch.Tensor([y]).unsqueeze(0)))[0]               # no noise

			action = np.squeeze(action.detach().numpy())
			v = action[0]
			omega = action[1]
			# print('v:', v)
			# print('omega:', omega)

			# v = 0.5  # velocity along y axis, cont here, can be change to s(t)

			# print('action:', )
			dy = 0.07 * v  # the step length along y axis
			# print('dy:', dy)
			y_ = y + dy  # estimated next y pos
			z_ = y_ ** 2 - 0.4 * y_ + 1.04  # estimated next z pos

			dz = z_ - z
			# print('dz:', dz)
			# print('omega:', omega)
			# print('ee_orient:', ee_orient)

			ee_pos, ee_orient, curr_joint_angles = move(dy, dz, omega, ee_pos, ee_orient, pr,
														agent)  # move the ee for 20 mini steps


			# check each step after ee_orient > 2.6, if stable, success, break, if not
			if ee_orient[0] > 2.6:  # 2.2 is largest angle of th ee
				for _ in range(5):
					agent.set_joint_target_positions(curr_joint_angles)  # wait for 5 loops to see if it's really stable

				if is_stable(ee, target) is True:
					print('success!')
					success_num += 1
					time.sleep(0.5)  # for observation
					# target.set_position([-10, -10, -10])   #
					r = get_reward(2, target, ee, args)    # success
					traj_reward += r
					target.remove()
					break
				else:
					r = get_reward(0, target, ee, args)  # fall
					traj_reward += r
					target.remove()
					break

			else:
				# check each step before ee_orient > 2.2, if stable, continue, if not break
				if is_stable(ee, target) is True:
					r = get_reward(1, target, ee, args)  # going on
					traj_reward += r

				else:
					r = get_reward(0, target, ee, args)  # fall
					traj_reward += r
					target.remove()
					break

		# print('traj length:', i)

		avg_reward.append(traj_reward)

	pr.stop()  # Stop the simulation
	pr.shutdown()  # Close the application

	success_rate = success_num / traj_num
	avg_reward = np.mean(avg_reward)
	print('success_rate:', success_rate)
	print('avg_reward:', avg_reward)

	return success_rate, avg_reward


def plot_rewards():
	# read saved rewards and combine them ---
	args = arguements.achieve_args()
	# n = 109  # change this
	# gap = 10 # change this
	tot_r = []
	for idx in range(10, 1000, 10):
		# idx = int((i+1) * gap)
		rewards_name = 'rewards_' + args.model_name + '_from' + str(idx - 10) + 'to' + str(idx) + '.txt'
		print('reward name:', rewards_name)
		with open(rewards_name, "rb") as fp:  # Unpickling
			temp_r = pickle.load(fp)
		print('rewards:', temp_r)

		tot_r = tot_r + temp_r

	print('--- saving all rewards ---')  # save rewards each 10 iters, one saving only contains 10 rewards
	rewards_name = 'rewards.txt'
	with open(rewards_name, "wb") as fp:  # Pickling
		pickle.dump(tot_r, fp)
	# print('rewards in the last several iters:', tot_r)

	print('tot_r:', tot_r)

	# now you have the total rewards, plot as you like it ---
	iter = list(range(len(tot_r)))
	plt.plot(iter, tot_r)
	plt.title('Walker')
	plt.xlabel('iterations')
	plt.ylabel('Rewards')
	plt.savefig('miao.png')
	plt.show()


def eval_model(policy_mdl):

	print('model:', policy_mdl)

	POLICY = testload(policy_mdl)

	print('----- in evaluation -----')

	success_rate, avg_reward = sample(POLICY)

	return success_rate, avg_reward


def testload(policy_mdl):

	s_dim = 1
	a_dim = 2

	print('model:', policy_mdl)

	POLICY = Policy(s_dim, a_dim)

	if os.path.exists(policy_mdl):
		POLICY.load_state_dict(torch.load(policy_mdl))

		print('loaded checkpoint from file:', policy_mdl)
	return POLICY


if __name__ == "__main__":
	eval_model('params007_900.pol.mdl')