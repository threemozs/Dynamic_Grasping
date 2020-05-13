
from ppo import PPO
import gym
import torch
import matplotlib.pyplot as plt
from policy import Policy
from value import Value
import os
import arguements
import pickle
import time
import numpy as np
import pandas as pd


def plots():

    ReLU_name = 'rewards.txt'

    with open(ReLU_name, "rb") as fp:  # Unpickling
        ReLU= pickle.load(fp)[0:980]

    p1 = pd.Series(ReLU)
    ma1 = p1.rolling(30).mean()


    iter = list(range(len(ReLU)))

    fig, ax1 = plt.subplots(figsize=(5, 3))
    plt.ylabel('Rewards')
    line1 = ax1.plot(iter, p1, color='dodgerblue', alpha=0.2)
    m1 = ax1.plot(ma1, label='Rewards', color='dodgerblue', alpha=1)

    ax1.legend()
    # plt.figure(figsize=(10, 5))
    plt.xlabel('Iterations')
    plt.ylabel('Rewards')
    plt.savefig('REWARD_FINAL.png')
    plt.show()


if __name__ == '__main__':
    plots()

