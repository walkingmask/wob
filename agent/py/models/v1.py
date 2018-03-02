# GAP(Global Average Pooling)を使った実装
# それ以外はwobに従った(つもり)


from .common import *


class FFPolicy(Policy):
    def __init__(self, ob_space, ac_space):
        super().__init__(ob_space, ac_space)

        # conv
        x = tf.nn.elu(conv2d(x, 16, "l1", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 24, "l2", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 32, "l3", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 48, "l4", [5, 5], [2, 2]))
        x = tf.nn.elu(conv2d(x, 32, "l5", [5, 5], [2, 2]))

        # GAP
        x = tf.reduce_mean(x, [1,2])

        # どう初期化するかわからなかったので、normalized_columns_initializerをデフォルトで使った
        x = linear(x, 384, "fc", normalized_columns_initializer())

        # logitは、確率(0,1)をR(-inf,inf)に変換する関数(p=0.5でlogit=0)
        # つまり、logitsはlinearで出力されたR^action_spaceをそれぞれの行動の確率をlogitしたもの
        # softmax(sigmoid)はこの逆変換で、つまりlogitsをsoftmaxすると確率になる
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
