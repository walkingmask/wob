# 各レイヤーにアクセスしやすいように改変


from .common import *


class FFPolicy3(Policy):
    def __init__(self, ob_space, ac_space):
        super().__init__(ob_space, ac_space)

        # conv
        conv = tf.nn.elu(conv2d(self.x, 16, "l1", [5, 5], [2, 2]))
        conv = tf.nn.elu(conv2d(conv,   24, "l2", [5, 5], [2, 2]))
        conv = tf.nn.elu(conv2d(conv,   32, "l3", [5, 5], [2, 2]))
        conv = tf.nn.elu(conv2d(conv,   48, "l4", [5, 5], [2, 2]))
        conv = tf.nn.elu(conv2d(conv,   32, "l5", [5, 5], [2, 2]))
        self.conv = conv

        self.gap = tf.reduce_mean(self.conv, [1,2])

        self.fc = linear(self.gap, 384, "fc", normalized_columns_initializer())

        self.logits = linear(self.fc, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(self.fc, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
