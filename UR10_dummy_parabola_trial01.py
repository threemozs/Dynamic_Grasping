'''Yihang Xu, 5/8/2020'''

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape  # modify scene
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import numpy as np
import math
import time
"""     1. Function to make get current position/orientation: 
        -end_pos: [x_cons, y, z]   ---> 
        -end effector rotation angle: 
            ee = agent.get_tip()
            ee_pos = ee.get_position()
            ee_orient = ee.get_orientation()
            
        2. Function to actuate the robot:
        - path = agent.get_path(position=next_pos, euler=[3.1415, 3.1415, 0.0])  # linear or nonlinear?
                            joint_angles = solve_ik(pos, orient)   then   agent.set_joint_target_positions(joint_angles)

        
        3.Rewards
        time spent penalty: -1
        flag1: if the obj falls, ends the simulation. -20
        flag2: if the obj reach at the goal stably. +100
        
        4. NN
        input: state((y,z, theta))  or (a,b,c,d1,d2)-->(v, w)  or (x)-->(v, w)
        output: action(v_y, v_z, w) with a gaussian noise -> actual action[, , ,] and log_p
        max_velocity=1.0
        
        5. Siumulation
        next_state_ = [y + v_y * dt, z + v_z * dt, theta + w * dt]   
        feed it into the simulator
        get next_state_ # --?


        6.Reward: -1 -20*flag1 + 100*flag2

        7.Loss: PPO loss: (log-log)*A
        

        Neural network:
        policy:3-20-10-3
        value:3-20-1 
"""


# LOOPS = 10  # training loop num
# batchsz = 2048

SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach.ttt')
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)  # lunch the ttt file
pr.start()
# time.sleep(5)

agent = UR10()

# position_min, position_max = [0.8, -0.2, 1.0], [1.5, 0.7, 1.0]

starting_joint_positions = [-1.5547982454299927, -0.11217942088842392, 2.505795478820801, 0.7483376860618591,
                            1.587110161781311, 4.083085536956787]  # these angles correspond to [1.0, 0.2, 1.0]
# starting_joint_positions = [-1.556139349937439, -0.12191621959209442, 2.501739978790283, 0.7608031034469604, 1.5856964588165283, -2.2]
# starting_joint_positions = [-1.2234,-0.3290,1.6263,1.1871,1.9181,1.5707]
agent.set_joint_positions(starting_joint_positions)
'''
ee-->end effector
start ee pos: [1, 0.2, 1.0]
    corresponding joint pos:   [-1.556139349937439, -0.12191621959209442, 2.501739978790283, 0.7608031034469604, 1.5856964588165283, -2.000717401504516]
ee goal pos: 1, 0.7 1.5
'''

# agent.set_control_loop_enabled(False)
agent.set_motor_locked_at_zero_velocity(True)

# ee_pos = np.array([1.0, 0.2, 1.0])

def move(dy, dz, omaga, ee_pos, ee_orient):

    # ee's x,y,z of the next step --
    ee_pos[1] += dy
    ee_pos[2] += dz
    # ee's orientation of the next step --
    ee_orient[0] += omaga
    # print('ee_pos:', ee_pos)
    # print('ee_orient:', ee_orient)

    new_joint_angles = agent.solve_ik(ee_pos, euler=ee_orient)  # get the joint angles of the robot by doing IK --

    # agent.set_joint_target_velocities([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])   # not sure how to use this --?

    agent.set_joint_target_positions(new_joint_angles)   # set the joint angles as the result of IK above

    pr.step()  # Step the physics simulation

    # get the actual  position and orientation of the ee after pr.step()
    ee = agent.get_tip()
    ee_pos = ee.get_position()
    ee_orient = ee.get_orientation()
    # print('ee_orient:', ee_orient)
    # print('ee_pos:', ee_pos)

    return ee_pos, ee_orient

''' An indirect way to control the speed along path

    One pr.step() cost ~0.07 seconds,
    We assume that every step cost 0.07 second, delta is the step length along a certain axis.
    Then the speed = delta / 0.07, 
    delta = 0.07 * speed.
    
    Fixed parabola: z = y^2 - 0.4y + 1.04, y(0) = 0.2, z(0) = 1.0
    k = 2y-0.4
    
    dummy network: (y-y0) --> (v_y, w)

'''

d = 0.01   # dummy params
dz = 0.03
omega = 0.01

for _ in range(50):
    print('------ new traj ------')
    agent = UR10()
    agent.set_joint_positions(starting_joint_positions)    # starting_joint_positions [1.0 , 0.2, 1.0]
    ee = agent.get_tip()
    ee_pos = ee.get_position()
    ee_orient = ee.get_orientation()
    print('initial_ee_pos:', ee_pos)
    print('initial_ee_orient:', ee_orient)
    # exit()

    target = Shape.create(type=PrimitiveShape.CUBOID,  # the cuboid
                          size=[0.05, 0.05, 0.4],
                          mass=0.5,
                          smooth=False,
                          color=[1.0, 0.1, 0.1],
                          static=False, respondable=True)
    target.set_position(np.array([1.0, 0.2, 1.0]))  # initial position of the target

    time.sleep(0.5)

    # ee_pos = np.array([1.0, 0.2, 1.0])
    # ee_angle = np.array([-2.2, 3.1415, 0.0])

    for i in range(20):   # 20 steps
        print('ee_pos:', ee_pos)  # current pos of ee

        y = ee_pos[1]
        z = ee_pos[2]
        print('y:', y)

        v = 0.2  # velocity along y axis, cont here, can be change to s(t)
        dy = 0.07 * v  # the step length along y axis
        print('dy:', dy)
        y_ = y + dy  # estimated next y pos
        z_ = y_ ** 2 - 0.4 * y_ + 1.04  # estimated next z pos
        # k = (z_ - z) / (y_ - y)  # the slope between current point and next point
        # print('k:', k)
        dz = z_ - z
        print('dz:', dz)
        ee_pos, ee_angle = move(dy, dz, omega, ee_pos, ee_orient)  # move the ee for 20 mini steps

    target.set_position([-10, -10, -10])

# [move(0.02, 0.02) for _ in range(20)]
# new_joint_angles = agent.solve_ik(position=np.array([1.0, 0.2, 1.0]), euler=np.array([3.0, 3.0, 0.0]))
# print('new_joint_angles:', new_joint_angles)

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application