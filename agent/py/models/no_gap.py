# v1のpooling(GAP)を2x2のaverage poolingにしたもの


from .layer import *


class FFPolicy2(object):
    def __init__(self, ob_space, ac_space):
        x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = x

        x = tf.nn.elu(conv2d(x, 16, "l1", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 24, "l2", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 32, "l3", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 48, "l4", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 32, "l5", [5, 5], [2, 2]))

        x = tf.nn.pool(x, [2, 2], "AVG", "SAME")
        x = tf.contrib.layers.flatten(x)

        x = linear(x, 384, "fc", normalized_columns_initializer())

        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf], {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]
