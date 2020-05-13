from ppo import PPO
import torch
import matplotlib.pyplot as plt
import arguements
import pickle


def main():
	torch.set_default_tensor_type('torch.DoubleTensor')
	args = arguements.achieve_args()

	ppo = PPO()

	ignore01, ignore02 = ppo.load()

	avg_rewards = []
	saved_rewards = []

	for i in range(10000):

		avg_reward, POLICY = ppo.update()
		avg_rewards.append(avg_reward)
		print('avg_rewards:', avg_rewards)
		saved_rewards.append(avg_reward)

		# saving model each 10 iterations
		if i % 10 == 0 and i != 0:
			pass
			idx = i  # MUST CHANGE THIS WHEN RESUME TRAINING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			print('--- saving models ---')
			ppo.save(idx, filename=args.model_name)

			print('--- saving rewards ---')   # save rewards each 10 iters, one saving only contains 10 rewards
			rewards_name = 'rewards_' + args.model_name + '_from' + str(idx - 10) + 'to' + str(idx) + '.txt'
			with open(rewards_name, "wb") as fp:  # Pickling
				pickle.dump(saved_rewards, fp)
			print('rewards in the last several iters:', saved_rewards)

			saved_rewards = []

		# plot the rewards each 5 iterations
		if i % 5 == 0:
			pass
			iter = list(range(len(avg_rewards)))
			plt.plot(iter, avg_rewards)
			plt.show()



if __name__ == '__main__':
	print('make sure to execute: [export OMP_NUM_THREADS=1] already.')
	main()
