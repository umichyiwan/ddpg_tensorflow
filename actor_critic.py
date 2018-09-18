import tensorflow as tf


class Actor(object):
    """
    Input to the networks is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, create_networks, s_dim, a_dim, a_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim

        # Actor Network
        with tf.variable_scope('actor'):
            self.s, self.a = create_networks(s_dim, a_dim, a_bound)

        network_params = tf.trainable_variables()

        # Target Network
        with tf.variable_scope('actor_target'):
            self.target_s, self.target_a = create_networks(s_dim, a_dim, a_bound)

        target_network_params = tf.trainable_variables()[len(network_params):]

        # Op for initializing target network with same weights as in network
        self.init_target_network_params = [
            target_network_params[i].assign(network_params[i])
            for i in range(len(target_network_params))
            ]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = [
            target_network_params[i].assign(tf.multiply(network_params[i], tau) + tf.multiply(target_network_params[i], 1. - tau))
            for i in range(len(target_network_params))
            ]

        # This gradient will be provided by the critic network
        self.critic_action_gradients = tf.placeholder(tf.float32, [None, a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.a, network_params, -self.critic_action_gradients)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.actor_gradients, network_params))

        self.num_trainable_vars = len(network_params) + len(target_network_params)

    def train(self, s, critic_action_gradients):
        self.sess.run(self.optimize, feed_dict={
            self.s: s,
            self.critic_action_gradients: critic_action_gradients
        })

    def predict(self, s):
        return self.sess.run(self.a, feed_dict={
            self.s: s
        })

    def predict_target(self, s):
        return self.sess.run(self.target_a, feed_dict={
            self.target_s: s
        })

    def init_target_network(self):
        self.sess.run(self.init_target_network_params)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    """
    Input to the networks is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, create_networks, s_dim, a_dim, learning_rate, tau, num_actor_vars, weight_decay):
        self.sess = sess

        # Critic network
        with tf.variable_scope('critic'):
            self.s, self.a, self.q, self.regularizer = create_networks(s_dim, a_dim)

        network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.variable_scope('critic_target'):
            self.target_s, self.target_a, self.target_q, _ = create_networks(s_dim, a_dim)

        target_network_params = tf.trainable_variables()[(len(network_params) + num_actor_vars):]

        # Op for initializing target network with same weights as in network
        self.init_target_network_params = [
            target_network_params[i].assign(network_params[i])
            for i in range(len(target_network_params))
            ]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = [
            target_network_params[i].assign(tf.multiply(network_params[i], tau) + tf.multiply(target_network_params[i], 1. - tau))
            for i in range(len(target_network_params))
            ]

        # Network target y
        self.y = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.p_surr = - tf.reduce_mean(self.q)  # policy surrogate loss (action is selected by policy)
        self.q_loss = tf.reduce_mean(tf.squared_difference(self.y, self.q))
        self.loss = self.q_loss + tf.reduce_mean(0.5 * weight_decay * self.regularizer)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.critic_action_gradients = tf.gradients(self.q, self.a)

    def train(self, s, a, y):
        return self.sess.run([self.q, self.q_loss, self.regularizer, self.optimize], feed_dict={
            self.s: s,
            self.a: a,
            self.y: y
        })

    def predict(self, s, a):
        return self.sess.run(self.q, feed_dict={
            self.s: s,
            self.a: a
        })

    def predict_target(self, s, a):
        return self.sess.run(self.target_q, feed_dict={
            self.target_s: s,
            self.target_a: a
        })

    def action_gradients(self, s, a):
        return self.sess.run([self.critic_action_gradients, self.p_surr], feed_dict={
            self.s: s,
            self.a: a
        })

    def init_target_network(self):
        self.sess.run(self.init_target_network_params)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)