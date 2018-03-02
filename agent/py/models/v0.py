# v1のpoolingを2x2のaverage poolingにしてfcを無くしたもの
# 本当の最初の実装


from .common import *


class FFPolicy0(Policy):
    def __init__(self, ob_space, ac_space):
        super().__init__(ob_space, ac_space)

        x = tf.nn.elu(conv2d(x, 16, "l1", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 24, "l2", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 32, "l3", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 48, "l4", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 32, "l5", [5, 5], [2, 2]))

        x = tf.nn.pool(x, [2, 2], "AVG", "SAME")
        x = tf.contrib.layers.flatten(x)

        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
