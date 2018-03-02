import numpy as np

import tensorflow as tf


# 方策ベースクラス
class Policy(object):
    def __init__(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32, [None] + list(ob_space))

    # 行動on_hot, 価値スカラ
    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf], {self.x: [ob]})

    # 探索なしのact
    def act_max(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.logits, {self.x: [ob]})[0]

    # 価値スカラ
    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


# 行正規化したランダムな値のテンソルを返すinitializerを返す
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# tf.nn.conv2dのwrapper
# 重みの初期化や共有変数についての設定など
def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        # strideはこの形である必要があるらしい
        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        stride_shape = [1, stride[0], stride[1], 1]
        # filterもstride同様に（x.get_shape()[3]はin_channels）
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # weightの初期化のためにw_boundを計算しているけど、これの意味は？
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * num_filters
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        # get_variableで共有変数の取得または作成
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b


# tf.matmulの共有変数wrapper
def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


# tf.reduce_max(tensor, [axis])でtensorの中のaxisで最大値のtensorを返す
# tf.multinomial(logits, x)で多項ロジットモデルを使ってlogitsに応じたshapeがxのランダム要素を生成する
    # 例 multinomial([10,10,5], 5) => ([0,1,0,2,1])
    # logitsの各要素の大きさを確率的に扱って、x個のlogitsのインデックスを出力する
    # multinomialは多項分布の意味
# tf.squeeze(t, [axis])でtのaxisで次元が1のテンソルを削減
# まとめると、logitsからその最大値を引いて、multinomialでカテゴリインデックスをランダムに計算してone_hotにしたものを返す
def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

