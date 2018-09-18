# ==========================
#   Training Parameters
# ==========================
# Number of training epochs, NUM_TRAINING_EPS = NUM_EPOCHS * EPOCH_LEN
NUM_EPOCHS = 500
# Max episode length
MAX_EP_STEPS = 50
# Number of training iterations per episode, it's generally equal to maximum steps per episode.
NUM_TRAININGS_PER_EP = 50
# Number of training episodes every training epoch
EPOCH_LEN = 100
# Number of testing episodes every TESTING_FREQ training episodes
TESTING_EPISODES = 100
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Weight decay for Critic Network
CRITIC_WEIGHT_DECAY = 0
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# OU process initial exploration epsilon
OU_INIT_EXP_EPS = 0.2
# OU process initial exploration epsilon
OU_FINAL_EXP_EPS = 0.2
# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Reacher-v1'
# Directory for storing gym results
MONITOR_DIR = './results/Reacher-v1'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 64