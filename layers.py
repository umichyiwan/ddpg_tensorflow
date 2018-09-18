import tensorflow as tf

# ==========================
# Neural Network Layers
# ==========================


def conv_layer(
        input,              # The previous layer.
        filter_size,        # Width and height of filters.
        num_filters,        # Number of filters.
        stride,
        initializer=tf.truncated_normal_initializer(stddev=0.05)
):
    num_input_channels = input.get_shape().as_list()[3]
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = tf.get_variable("weights", shape, initializer=initializer)
    biases = tf.get_variable("biases", num_filters, initializer=tf.constant_initializer(0))
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    layer += biases
    layer = tf.nn.relu(layer)

    return layer, weights


def deconv_layer(
        input,              # The previous layer.
        filter_size,        # Width and height of filters.
        num_filters,        # Number of filters.
        stride,
        activation='relu',
        initializer=tf.truncated_normal_initializer(stddev=0.05)
):
    _, input_height, input_width, num_input_channels = input.get_shape().as_list()
    batch_size = tf.shape(input)[0]
    shape = [filter_size, filter_size, num_filters, num_input_channels]
    weights = tf.get_variable("weights", shape, initializer=initializer)
    biases = tf.get_variable("biases", [num_filters], initializer=initializer)
    out_height = (input_height - 1) * stride + filter_size
    out_width = (input_width - 1) * stride + filter_size
    output_shape = tf.stack([batch_size, out_height, out_width, num_filters])
    layer = tf.nn.conv2d_transpose(input, weights, output_shape, strides=[1, stride, stride, 1], padding='VALID')
    layer += biases
    if activation == 'relu':
        layer = tf.nn.relu(layer)
    return layer, weights


def fc_layer(
        input,          # The previous layer.
        num_outputs,    # Num. outputs.
        activation,     # Non-linear function
        initializer=tf.truncated_normal_initializer(stddev=0.02)
):
    num_inputs = input.get_shape().as_list()[1]
    # Create new weights and biases.
    weights = tf.get_variable("weights", shape=[num_inputs, num_outputs], initializer=initializer)
    biases = tf.get_variable("biases", shape=num_outputs, initializer=tf.constant_initializer(0))
    layer = tf.matmul(input, weights) + biases

    if activation == 'relu':
        layer = tf.nn.relu(layer)
    elif activation == 'tanh':
        layer = tf.nn.tanh(layer)
    elif activation == 'linear':
        pass
    else:
        print "error: no this activation layer " + activation
        exit(0)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def fusion_layer(
        inputs_1,
        inputs_2,
        num_outputs,
        activation='linear',
        initializer_1=tf.truncated_normal_initializer(stddev=0.02),
        initializer_2=tf.truncated_normal_initializer(stddev=0.02)
):
    num_inputs_1 = inputs_1.get_shape().as_list()[1]
    num_inputs_2 = inputs_2.get_shape().as_list()[1]
    weights1 = tf.get_variable("weights1", shape=[num_inputs_1, num_outputs], initializer=initializer_1)
    weights2 = tf.get_variable("weights2", shape=[num_inputs_2, num_outputs], initializer=initializer_2)
    biases = tf.get_variable("biases", shape=num_outputs, initializer=tf.constant_initializer(0))
    action_fusion = tf.matmul(inputs_1, weights1) + tf.matmul(inputs_2, weights2) + biases

    if activation == 'relu':
        action_fusion = tf.nn.relu(action_fusion)
    elif activation == 'tanh':
        action_fusion = tf.nn.tanh(action_fusion)
    elif activation == 'linear':
        pass
    else:
        print "error: no this activation layer " + activation
        exit(0)
    return action_fusion, weights1, weights2