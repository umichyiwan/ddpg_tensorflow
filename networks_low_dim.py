import tensorflow as tf
from initialization import fan_in_init, uniform_init
from layers import fc_layer, fusion_layer


def create_actor_network(s_dim, a_dim, a_bound):
    s = tf.placeholder(tf.float32, shape=[None, s_dim])

    # First fully-connected layer.
    with tf.variable_scope('fc1'):
        layer_fc1, _ = fc_layer(s, 400, 'relu', fan_in_init(s_dim))

    # Second fully-connected layer.
    with tf.variable_scope('fc2'):
        layer_fc2, _ = fc_layer(layer_fc1, 300, 'relu', fan_in_init(400))

    # Output layer.
    with tf.variable_scope('output'):
        output, _ = fc_layer(layer_fc2, a_dim, 'tanh', uniform_init(-0.003, 0.003))
        a = tf.multiply(output, a_bound)  # Scale output to -action_bound to action_bound

    return s, a


def create_critic_network(s_dim, a_dim):
    s = tf.placeholder(tf.float32, shape=[None, s_dim])
    a = tf.placeholder(tf.float32, shape=[None, a_dim])

    # First fully-connected layer.
    with tf.variable_scope('fc1'):
        layer_fc1, fc1_weights = fc_layer(s, 400, 'relu', fan_in_init(s_dim))

    # Action Fusion layer.
    with tf.variable_scope('action_fusion'):
        action_fusion, af_weights1, af_weights2 = fusion_layer(layer_fc1, a, 300, 'relu', fan_in_init(400), fan_in_init(a_dim))

    # Output layer.
    with tf.variable_scope('output'):
        q, fc2_weights = fc_layer(action_fusion, 1, 'linear', uniform_init(-0.003, 0.003))

    regularizers = tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(af_weights1) + tf.nn.l2_loss(
        af_weights2)
    return s, a, q, regularizers


def create_model_network(s_dim, a_dim, o_dim):
    s = tf.placeholder(tf.float32, shape=[None, s_dim])
    a = tf.placeholder(tf.float32, shape=[None, a_dim])

    # Action Fusion layer.
    with tf.variable_scope('action_fusion'):
        action_fusion, _, _ = fusion_layer(s, a, 200, 'relu', fan_in_init(400), fan_in_init(400))

    with tf.variable_scope('fc1'):
        fc1, _ = fc_layer(action_fusion, 6, 'relu', fan_in_init(200))

    with tf.variable_scope('fc2'):
        fc2, _ = fc_layer(fc1, 200, 'relu', fan_in_init(200))

    with tf.variable_scope('output'):
        output, _ = fc_layer(action_fusion, o_dim, 'linear', uniform_init(-0.003, 0.003))

    return s, a, output

# def create_model_network(s_dim, a_dim, o_dim):
#     s = tf.placeholder(tf.float32, shape=[None, s_dim])
#     a = tf.placeholder(tf.float32, shape=[None, a_dim])
#
#     # First fully-connected layer.
#     with tf.variable_scope('fc1'):
#         layer_fc1, _ = fc_layer(s, 400, 'relu', fan_in_init(s_dim))
#
#     # Action Fusion layer.
#     with tf.variable_scope('action_fusion'):
#         action_fusion, _, _ = fusion_layer(layer_fc1, a, 300, 'relu', fan_in_init(400), fan_in_init(a_dim))
#
#     # Output layer.
#     # with tf.variable_scope('r_output'):
#     #     r_output, _ = fc_layer(action_fusion, 1, 'linear', uniform_init(-0.003, 0.003))
#
#     with tf.variable_scope('output'):
#         output, _ = fc_layer(action_fusion, o_dim, 'linear', uniform_init(-0.003, 0.003))
#
#     return s, a, output


def create_predictor_network(s_dim):
    s = tf.placeholder(tf.float32, shape=[None, s_dim])

    # First fully-connected layer.
    with tf.variable_scope('fc1'):
        layer_fc1, _ = fc_layer(s, 400, 'relu', fan_in_init(s_dim))

    # Second fully-connected layer.
    with tf.variable_scope('fc2'):
        layer_fc2, _ = fc_layer(layer_fc1, 300, 'relu', fan_in_init(400))

    # Output layer.
    with tf.variable_scope('output'):
        output, _ = fc_layer(layer_fc2, 1, 'linear', uniform_init(-0.003, 0.003))

    return s, output
