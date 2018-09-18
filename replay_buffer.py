""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, s2):
        experience = (s, a, r, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferWithTerminal(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferWithGoal(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, s2, target):
        experience = (s, a, r, s2, target)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        g_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferWithGoalWithTerminal(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2, target):
        experience = (s, a, r, t, s2, target)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        g_batch = np.array([_[5] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch, g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferTrueGoalPseudoGoal(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, ddpg_r, vt_r, s2, ddpg_g, vt_g):
        experience = (s, a, ddpg_r, vt_r, s2, ddpg_g, vt_g)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        ddpg_r_batch = np.array([_[2] for _ in batch])
        vt_r_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        ddpg_g_batch = np.array([_[5] for _ in batch])
        vt_g_batch = np.array([_[6] for _ in batch])

        return s_batch, a_batch, ddpg_r_batch, vt_r_batch, s2_batch, ddpg_g_batch, vt_g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferTrueGoalPseudoGoalWithTerminal(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, ddpg_r, vt_r, t, s2, ddpg_g, vt_g):
        experience = (s, a, ddpg_r, vt_r, t, s2, ddpg_g, vt_g)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        ddpg_r_batch = np.array([_[2] for _ in batch])
        vt_r_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        s2_batch = np.array([_[5] for _ in batch])
        ddpg_g_batch = np.array([_[6] for _ in batch])
        vt_g_batch = np.array([_[7] for _ in batch])

        return s_batch, a_batch, ddpg_r_batch, vt_r_batch, t_batch, s2_batch, ddpg_g_batch, vt_g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferTrueGoalPseudoGoalBalanced(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.rewarding_count = 0
        self.non_rewarding_count = 0
        self.rewarding_buffer = deque()
        self.non_rewarding_buffer = deque()
        random.seed(random_seed)

    def add_rewarding(self, s, a, tr, pr, s2, tg, pg):
        experience = (s, a, tr, pr, s2, tg, pg)

        if self.rewarding_count < self.buffer_size:
            self.rewarding_buffer.append(experience)
            self.rewarding_count += 1
        else:
            self.rewarding_buffer.popleft()
            self.rewarding_buffer.append(experience)

    def add_non_rewarding(self, s, a, tr, pr, s2, tg, pg):
        experience = (s, a, tr, pr, s2, tg, pg)

        if self.non_rewarding_count < self.buffer_size:
            self.non_rewarding_buffer.append(experience)
            self.non_rewarding_count += 1
        else:
            self.non_rewarding_buffer.popleft()
            self.non_rewarding_buffer.append(experience)

    def rewarding_buffer_size(self):
        return self.rewarding_count

    def non_rewarding_buffer_size(self):
        return self.non_rewarding_count

    def sample_batch(self, batch_size):
        if batch_size / 2 > self.rewarding_count and batch_size / 2 > self.non_rewarding_count:
            print "buffer not enough samples"
            exit(1)
        elif batch_size / 2 < self.rewarding_count and batch_size / 2 < self.non_rewarding_count:
            rewarding_batch = random.sample(self.rewarding_buffer, batch_size / 2)
            non_rewarding_batch = random.sample(self.non_rewarding_buffer, batch_size / 2)
            batch = rewarding_batch + non_rewarding_batch

        elif batch_size / 2 > self.rewarding_count:
            batch = random.sample(self.non_rewarding_buffer, batch_size / 2)

        else:
            batch = random.sample(self.rewarding_buffer, batch_size / 2)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        tr_batch = np.array([_[2] for _ in batch])
        pr_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        tg_batch = np.array([_[5] for _ in batch])
        pg_batch = np.array([_[6] for _ in batch])

        return s_batch, a_batch, tr_batch, pr_batch, s2_batch, tg_batch, pg_batch


    def clear(self):
        self.rewarding_buffer.clear()
        self.non_rewarding_buffer.clear()
        self.rewarding_count = 0
        self.non_rewarding_count = 0


class ReplayBufferTwoStates(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, gym_s, a, r, t, s2, gym_s2):
        experience = (s, gym_s, a, r, t, s2, gym_s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        gym_s_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        r_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        s2_batch = np.array([_[5] for _ in batch])
        gym_s2_batch = np.array([_[6] for _ in batch])

        return s_batch, gym_s_batch, a_batch, r_batch, t_batch, s2_batch, gym_s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferBalanced(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.rewarding_count = 0
        self.non_rewarding_count = 0
        self.mean_reward = 0
        self.rewarding_buffer = deque()
        self.non_rewarding_buffer = deque()
        random.seed(random_seed)

    def add_rewarding(self, s, a, r, s2):
        experience = (s, a, r, s2)

        if self.rewarding_count < self.buffer_size:
            self.rewarding_buffer.append(experience)
            self.rewarding_count += 1
        else:
            self.rewarding_buffer.popleft()
            self.rewarding_buffer.append(experience)

    def add_non_rewarding(self, s, a, r, s2):
        experience = (s, a, r, s2)

        if self.non_rewarding_count < self.buffer_size:
            self.non_rewarding_buffer.append(experience)
            self.non_rewarding_count += 1
        else:
            self.non_rewarding_buffer.popleft()
            self.non_rewarding_buffer.append(experience)

    def rewarding_buffer_size(self):
        return self.rewarding_count

    def non_rewarding_buffer_size(self):
        return self.non_rewarding_count

    def sample_batch(self, batch_size):
        if batch_size / 2 > self.rewarding_count and batch_size / 2 > self.non_rewarding_count:
            print "buffer not enough samples"
            exit(1)
        elif batch_size / 2 < self.rewarding_count and batch_size / 2 < self.non_rewarding_count:
            rewarding_batch = random.sample(self.rewarding_buffer, batch_size / 2)
            non_rewarding_batch = random.sample(self.non_rewarding_buffer, batch_size / 2)
            batch = rewarding_batch + non_rewarding_batch

        elif batch_size / 2 > self.rewarding_count:
            batch = random.sample(self.non_rewarding_buffer, batch_size / 2)

        else:
            batch = random.sample(self.rewarding_buffer, batch_size / 2)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.rewarding_buffer.clear()
        self.non_rewarding_buffer.clear()
        self.rewarding_count = 0
        self.non_rewarding_count = 0


class ReplayBufferBiasedWithGoal(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.positive_count = 0
        self.negative_count = 0
        self.mean_reward = 0
        self.positive_buffer = deque()
        self.negative_buffer = deque()
        self.rewards = deque()
        self.rewards_count = 0
        random.seed(random_seed)

    def add(self, s, a, r, r2, t, s2, target):
        experience = (s, a, r, r2, t, s2, target)
        if self.rewards_count < self.buffer_size:
            self.rewards.append(r)
            self.rewards_count += 1
        else:
            self.rewards.popleft()
            self.rewards.append(r)
        self.mean_reward = np.mean(self.rewards)
        margin = (np.max(self.rewards) - self.mean_reward) * 0.3
        if r > self.mean_reward + margin:
            if self.positive_count < self.buffer_size:
                self.positive_buffer.append(experience)
                self.positive_count += 1
            else:
                self.positive_buffer.popleft()
                self.positive_buffer.append(experience)
        else:
            if self.negative_count < self.buffer_size:
                self.negative_buffer.append(experience)
                self.negative_count += 1
            else:
                self.negative_buffer.popleft()
                self.negative_buffer.append(experience)

    def positive_size(self):
        return self.positive_count

    def negative_size(self):
        return self.negative_count

    def sample_batch(self, batch_size):

        if self.positive_count < batch_size / 2:
            positive_batch = random.sample(self.positive_buffer, self.positive_count)
        else:
            positive_batch = random.sample(self.positive_buffer, batch_size / 2)

        if self.negative_count < batch_size / 2:
            negative_batch = random.sample(self.negative_buffer, self.negative_count)
        else:
            negative_batch = random.sample(self.negative_buffer, batch_size / 2)

        batch = positive_batch + negative_batch

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        r2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        s2_batch = np.array([_[5] for _ in batch])
        g_batch = np.array([_[6] for _ in batch])

        return s_batch, a_batch, r_batch, r2_batch, t_batch, s2_batch, g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferWithGoal2R(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r1, r2, t, s2, target):
        experience = (s, a, r1, r2, t, s2, target)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r1_batch = np.array([_[2] for _ in batch])
        r2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        s2_batch = np.array([_[5] for _ in batch])
        g_batch = np.array([_[6] for _ in batch])

        return s_batch, a_batch, r1_batch, r2_batch, t_batch, s2_batch, g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


class ReplayBufferBiasedDdpgVt(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.positive_count = 0
        self.negative_count = 0
        self.mean_reward = 0
        self.positive_buffer = deque()
        self.negative_buffer = deque()
        self.rewards = deque()
        self.rewards_count = 0
        random.seed(random_seed)

    def add(self, s, a, ddpg_r, vt_r, t, s2, ddpg_g, vt_g):
        experience = (s, a, ddpg_r, vt_r, t, s2, ddpg_g, vt_g)

        if self.rewards_count < self.buffer_size:
            self.rewards.append(ddpg_r)
            self.rewards_count += 1
        else:
            self.rewards.popleft()
            self.rewards.append(ddpg_r)
        self.mean_reward = np.mean(self.rewards)
        margin = (np.max(self.rewards) - self.mean_reward) * 0.3
        if ddpg_r > self.mean_reward + margin:
            if self.positive_count < self.buffer_size:
                self.positive_buffer.append(experience)
                self.positive_count += 1
            else:
                self.positive_buffer.popleft()
                self.positive_buffer.append(experience)
        else:
            if self.negative_count < self.buffer_size:
                self.negative_buffer.append(experience)
                self.negative_count += 1
            else:
                self.negative_buffer.popleft()
                self.negative_buffer.append(experience)

    def positive_size(self):
        return self.positive_count

    def negative_size(self):
        return self.negative_count

    def sample_batch(self, batch_size):
        if self.positive_count < batch_size / 2:
            positive_batch = random.sample(self.positive_buffer, self.positive_count)
        else:
            positive_batch = random.sample(self.positive_buffer, batch_size / 2)

        if self.negative_count < batch_size / 2:
            negative_batch = random.sample(self.negative_buffer, self.negative_count)
        else:
            negative_batch = random.sample(self.negative_buffer, batch_size / 2)

        batch = positive_batch + negative_batch

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        ddpg_r_batch = np.array([_[2] for _ in batch])
        vt_r_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        s2_batch = np.array([_[5] for _ in batch])
        ddpg_g_batch = np.array([_[6] for _ in batch])
        vt_g_batch = np.array([_[7] for _ in batch])

        return s_batch, a_batch, ddpg_r_batch, vt_r_batch, t_batch, s2_batch, ddpg_g_batch, vt_g_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
