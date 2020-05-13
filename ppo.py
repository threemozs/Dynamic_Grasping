import os
import torch
from torch import optim
from torch.autograd import Variable
from policy import Policy
from value import Value
import arguements


from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape  # modify scene
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import numpy as np
import time


''' An indirect way to control the speed along path

    One pr.step() cost ~0.07 seconds,
    We assume that every step cost 0.07 second, delta is the step length along a certain axis.
    Then the speed = delta / 0.07, 
    delta = 0.07 * speed.

    Fixed parabola: z = y^2 - 0.4y + 1.04, y(0) = 0.2, z(0) = 1.0
  
    dummy network: (y-y0) --> (v_y, w)

'''


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
	batchsz = args.sample_point_num

	SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_003.ttt')
	# SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_002.ttt')
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

	data = {'state': [], 'action': [], 'reward': [], 'done': [], }
	sample_num = 0
	avg_reward = []
	while sample_num < batchsz:

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
			data['state'].append(y)
			z = ee_pos[2]
			# print('y:', y)

			action = policy.select_action(Variable(torch.Tensor([y]).unsqueeze(0)))[0]   # add noise to actuib
			# action = policy(Variable(torch.Tensor([y]).unsqueeze(0)))[0]               # no noise

			action = np.squeeze(action.detach().numpy())
			v = action[0]
			omega = action[1]
			# print('v:', v)
			# print('omega:', omega)

			# v = 0.5  # velocity along y axis, cont here, can be change to s(t)
			data['action'].append(np.squeeze(np.asarray([v, omega])))
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
			sample_num += 1

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
					data['reward'].append(r)
					data['done'].append(0)
					target.remove()
					break
				else:
					r = get_reward(0, target, ee, args)  # fall
					traj_reward += r
					data['reward'].append(r)
					data['done'].append(0)

					target.remove()
					break

			else:
				# check each step before ee_orient > 2.2, if stable, continue, if not break
				if is_stable(ee, target) is True:
					r = get_reward(1, target, ee, args)  # going on
					traj_reward += r
					data['reward'].append(r)
					data['done'].append(1)  # continue

				else:
					r = get_reward(0, target, ee, args)  # fall
					traj_reward += r
					data['reward'].append(r)
					data['done'].append(0)

					target.remove()
					break


		print('traj length:', i)

		avg_reward.append(traj_reward)

	pr.stop()  # Stop the simulation
	pr.shutdown()  # Close the application

	# print('success_num:', success_num)
	print('success_rate:', success_num / traj_num)
	# print('avg_reward:', np.mean(avg_reward))
	# print('data:', data)

	return data, np.mean(avg_reward)


class PPO:

	def __init__(self):
		"""

		:param env_cls: env class or function, not instance, as we need to create several instance in class.
		:param thread_num:
		"""

		self.args = arguements.achieve_args()
		self.gamma = self.args.gamma
		self.lr = self.args.lr
		self.epsilon = self.args.epsilon
		self.tau = self.args.tau

		# construct policy and value network
		self.policy = Policy(1, 2)
		self.value = Value(1)

		self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
		self.value_optim = optim.Adam(self.value.parameters(), lr=self.lr)

	def est_adv(self, r, v, mask):
		"""
		we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
		:param r: reward, Tensor, [b]
		:param v: estimated value, Tensor, [b]
		:param mask: indicates ending for 0 otherwise 1, Tensor, [b]
		:return: A(s, a), V-target(s), both Tensor
		"""
		batchsz = v.size(0)

		# v_target is worked out by Bellman equation.
		v_target = torch.Tensor(batchsz)
		delta = torch.Tensor(batchsz)
		A_sa = torch.Tensor(batchsz)

		prev_v_target = 0
		prev_v = 0
		prev_A_sa = 0
		for t in reversed(range(batchsz)):
			# mask here indicates a end of trajectory
			# this value will be treated as the target value of value network.
			# mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
			# formula: V(s_t) = r_t + gamma * V(s_t+1)
			v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

			# formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
			delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

			# formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
			A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

			# update previous
			prev_v_target = v_target[t]
			prev_v = v[t]
			prev_A_sa = A_sa[t]

		# normalize A_sa
		A_sa = (A_sa - A_sa.mean()) / A_sa.std()

		return A_sa, v_target

	def update(self):
		"""
		firstly sample batchsz items and then perform optimize algorithms.
		:param batchsz:
		:return:
		"""
		# 1. sample data asynchronously

		# batch = self.sample_original(self.args.sample_point_num)
		batch, avg_reward = sample(self.policy)
		self.avg_reward = avg_reward

		s = torch.from_numpy(np.stack(batch['state'])).view(-1, 1)
		a = torch.from_numpy(np.array(batch['action']))
		r = torch.from_numpy(np.array(batch['reward']))
		mask = torch.from_numpy(np.array(batch['done']))
		batchsz = s.size(0)

		# print('s:', s)
		# print(s.size())
		# print('a:', a)
		# print(a.size())
				# print('r:', r)
		# print(r.size())
		#
		# print('mask:', mask)
		# print(mask.size())
		#
		# print('---- batchsz:-----', batchsz)

		# exit()

		# 2. get estimated V(s) and PI_old(s, a),
		# v: [b, 1] => [b]
		v = self.value(Variable(s)).data.squeeze()
		log_pi_old_sa = self.policy.get_log_prob(Variable(s), Variable(a)).data

		# 3. estimate advantage and v_target according to GAE and Bellman Equation
		A_sa, v_target = self.est_adv(r, v, mask)


		# 4. backprop.

		v_target = Variable(v_target)
		A_sa = Variable(A_sa)
		s = Variable(s)
		a = Variable(a)
		log_pi_old_sa = Variable(log_pi_old_sa)

		for _ in range(self.args.epoch_num):

			# 4.1 shuffle current batch
			perm = torch.randperm(batchsz)
			# shuffle the variable for mutliple optimize
			v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
			                                                               log_pi_old_sa[perm]

			# 4.2 get mini-batch for optimizing
			optim_batchsz = self.args.optim_batchsz
			optim_chunk_num = int(np.ceil(batchsz / optim_batchsz))
			# chunk the optim_batch for total batch
			v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
			                                                               torch.chunk(A_sa_shuf, optim_chunk_num), \
			                                                               torch.chunk(s_shuf, optim_chunk_num), \
			                                                               torch.chunk(a_shuf, optim_chunk_num), \
			                                                               torch.chunk(log_pi_old_sa_shuf,
			                                                                           optim_chunk_num)
			# 4.3 iterate all mini-batch to optimize
			for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
			                                                         log_pi_old_sa_shuf):
				# print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
				# 1. update value network
				v_b = self.value(s_b)
				loss = torch.pow(v_b - v_target_b, 2).mean()
				self.value_optim.zero_grad()
				loss.backward()
				self.value_optim.step()

				# 2. update policy network by clipping
				# [b, 1]
				log_pi_sa = self.policy.get_log_prob(s_b, a_b)
				# ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
				# [b, 1] => [b]
				ratio = torch.exp(log_pi_sa - log_pi_old_sa_b).squeeze(1)
				surrogate1 = ratio * A_sa_b
				surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
				surrogate = - torch.min(surrogate1, surrogate2).mean()

				# backprop
				self.policy_optim.zero_grad()
				surrogate.backward(retain_graph=True)
				# gradient clipping, for stability
				torch.nn.utils.clip_grad_norm(self.policy.parameters(), 10)
				self.policy_optim.step()
		return self.avg_reward, self.policy

	def save(self, i, filename='ppo'):

		torch.save(self.value.state_dict(), filename + str(i) + '.val.mdl')
		torch.save(self.policy.state_dict(), filename + str(i) + '.pol.mdl')

		print('saved network to mdl')

	def load(self, filename='ppo'):
		# value_mdl = 'params005_base.val.mdl'
		# policy_mdl = 'params005_base.pol.mdl'

		# value_mdl = 'params006_480.val.mdl'
		# policy_mdl = 'params006_480.pol.mdl'

		value_mdl = 'params007_900.val.mdl'
		policy_mdl = 'params007_900.pol.mdl'

		if os.path.exists(value_mdl):
			self.value.load_state_dict(torch.load(value_mdl))
			print('loaded checkpoint from file:', value_mdl)
		if os.path.exists(policy_mdl):
			self.policy.load_state_dict(torch.load(policy_mdl))

			print('loaded checkpoint from file:', policy_mdl)
		return self.value, self.policy

