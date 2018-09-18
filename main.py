"""
Implementation of original low dimensional DDPG - Deep Deterministic Policy Gradient

"Continuous control with deep reinforcement learning"

1. low dimensional state representation tested on Open AI Gym
2. change environment in config file

Environment example: Open AI Gym Reacher-v1 task:
state: [cos theta1, cos theta2, sin theta1, sin theta2, goal x, goal y, vel theta1, vel theta2,
dist(goal x, fingertip x), dist(goal y, fingertip y), 0]
action: [torque1, torque2]
reward: - dist(goal, fingertip) - sum(square(action))

Author: Yi Wan
"""
import gym
import numpy as np
import pyprind
import tensorflow as tf
from gym import wrappers
import matplotlib.pyplot as plt

from actor_critic import Actor, Critic
from noise import generate_noise
from replay_buffer import ReplayBuffer
from config_low_dim import \
    NUM_EPOCHS, MAX_EP_STEPS, NUM_TRAININGS_PER_EP, EPOCH_LEN, TESTING_EPISODES, ACTOR_LEARNING_RATE, \
    CRITIC_LEARNING_RATE, CRITIC_WEIGHT_DECAY, GAMMA, TAU, OU_INIT_EXP_EPS, OU_FINAL_EXP_EPS, RENDER_ENV, \
    GYM_MONITOR_EN, ENV_NAME, MONITOR_DIR, SUMMARY_DIR, RANDOM_SEED, BUFFER_SIZE, MINIBATCH_SIZE
from networks_low_dim import create_actor_network, create_critic_network

print "NUM_EPOCHS: ", NUM_EPOCHS
print "MAX_EP_STEPS: ", MAX_EP_STEPS
print "NUM_TRAININGS_PER_EP: ", NUM_TRAININGS_PER_EP
print "EPOCH_LEN: ", EPOCH_LEN
print "TESTING_EPISODES: ", TESTING_EPISODES
print "ACTOR_LEARNING_RATE: ", ACTOR_LEARNING_RATE
print "CRITIC_LEARNING_RATE: ", CRITIC_LEARNING_RATE
print "CRITIC_WEIGHT_DECAY: ", CRITIC_WEIGHT_DECAY
print "GAMMA: ", GAMMA
print "TAU: ", TAU
print "OU_INIT_EXP_EPS: ", OU_INIT_EXP_EPS
print "OU_FINAL_EXP_EPS: ", OU_FINAL_EXP_EPS
print "RENDER_ENV: ", RENDER_ENV
print "GYM_MONITOR_EN: ", GYM_MONITOR_EN
print "ENV_NAME: ", ENV_NAME
print "MONITOR_DIR: ", MONITOR_DIR
print "SUMMARY_DIR: ", SUMMARY_DIR
print "RANDOM_SEED: ", RANDOM_SEED
print "BUFFER_SIZE: ", BUFFER_SIZE
print "MINIBATCH_SIZE: ", MINIBATCH_SIZE

# ===========================
#   Agent Training
# ===========================


def train(sess, env, actor, critic):

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.init_target_network()
    critic.init_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in xrange(NUM_EPOCHS):
        es_path_return = []
        r_list = []
        q_value_list = []
        y_value_list = []
        q_loss_list = []
        q_reg_list = []
        p_surr_list = []
        for _ in pyprind.prog_bar(range(EPOCH_LEN)):
            # Initialize environment
            s1 = env.reset()
            path_return = 0
            # for j in xrange(MAX_EP_STEPS):
            while 1:
                a = actor.predict(np.reshape(s1, (1, actor.s_dim)))[0] + \
                    generate_noise(actor.a_dim, i, "OUNoise", OU_INIT_EXP_EPS, OU_FINAL_EXP_EPS, NUM_EPOCHS)
                a = np.clip(a, env.action_space.low, env.action_space.high)

                s2, r, terminal, info = env.step(a)

                path_return += r

                replay_buffer.add(
                    np.reshape(s1, (actor.s_dim,)),
                    np.reshape(a, (actor.a_dim,)),
                    r,
                    np.reshape(s2, (actor.s_dim,))
                )

                s1 = s2

                if replay_buffer.size() > MINIBATCH_SIZE:
                    s1_batch, a_batch, r_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # Calculate targets
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                    y_value = r_batch + GAMMA * target_q.reshape((MINIBATCH_SIZE,))

                    # Update the critic given the targets
                    q_value, q_loss, q_reg, _ = critic.train(s1_batch, a_batch, np.reshape(y_value, (MINIBATCH_SIZE, 1)))

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s1_batch)
                    critic_action_gradients, p_surr = critic.action_gradients(s1_batch, a_outs)
                    actor.train(s1_batch, critic_action_gradients[0])

                    # Update target actor_critic
                    actor.update_target_network()
                    critic.update_target_network()

                    # Update evaluation info
                    r_list.append(r_batch)
                    q_value_list.append(q_value)
                    y_value_list.append(y_value)
                    q_loss_list.append(q_loss)
                    q_reg_list.append(q_reg)
                    p_surr_list.append(p_surr)

                if terminal:
                    # print path_return
                    # print replay_buffer.size()
                    es_path_return.append(path_return)
                    break
        all_rs = np.concatenate(r_list)
        all_qs = np.concatenate(q_value_list)
        all_ys = np.concatenate(y_value_list)
        # plt.plot(all_rs, all_qs, 'ro')
        # plt.show()
        paths = test(env, actor)
        returns = [sum(path['rewards']) for path in paths]
        actions = np.concatenate([path["actions"] for path in paths])
        print '| Epoch', i
        print '| AverageReturn', np.mean(returns)
        print '| StdReturn', np.std(returns)
        print '| MaxReturn', np.max(returns)
        print '| MinReturn', np.min(returns)
        print '| AverageEsReturn', np.mean(es_path_return)
        print '| EsStdReturn', np.std(es_path_return)
        print '| EsMaxReturn', np.max(es_path_return)
        print '| EsMinReturn', np.min(es_path_return)
        print '| AverageQLoss: ', np.mean(q_loss_list)
        print '| AveragePolicySurr: ', np.mean(p_surr_list)
        print '| AverageQ: ', np.mean(all_qs)
        print '| AverageY: ', np.mean(all_ys)
        print '| AverageAction: ', np.mean(np.square(actions))
        print '| QFunRegParamNorm: ', np.mean(q_reg_list)
        print '\n'


def test(env, actor):
    paths = []
    for i in xrange(TESTING_EPISODES):
        s = env.reset()
        path = {'rewards': [], 'actions': []}
        frm = 0
        # while frm < MAX_EP_STEPS:
        while 1:
            if i == 0:
                env.render()
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))[0]
            a = np.clip(a, env.action_space.low, env.action_space.high)
            s, r, terminal, info = env.step(a)
            path['rewards'].append(r)
            path['actions'].append(a)
            if terminal:
                paths.append(path)
                break
            frm += 1

    return paths


def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert np.all(env.action_space.high == -env.action_space.low)

        actor = Actor(sess, create_actor_network, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)

        critic = Critic(sess, create_critic_network, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), CRITIC_WEIGHT_DECAY)

        # if GYM_MONITOR_EN:
        #     if not RENDER_ENV:
        #         env = wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)
        #     else:
        #         env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
