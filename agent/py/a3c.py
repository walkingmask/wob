# 勾配計算・更新
# スレッド


from collections import namedtuple
import threading

import scipy.signal
import numpy as np

import six.moves.queue as queue

import tensorflow as tf

import models


# 報酬リストから割引報酬リストを生成
# 例: discount([0,0,0,0,1], 0.99) => [0.9605,0.9702,0.9801,0.99,1.]
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# 経験から割引報酬とアドバンテージを計算してバッチとしてリターン
def process_rollout(rollout, gamma, lambda_=1.0):
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])

    # 割引報酬
    batch_r = discount(rewards_plus_v, gamma)[:-1]

    # アドバンテージ
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # Generalized Advantage Estimation
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal"])


class PartialRollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False

    def add(self, state, action, reward, value, terminal):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal


# スレッド部
class RunnerThread(threading.Thread):
    def __init__(self, env, policy, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    # threading.Threadのstartとrun
    def run(self):
        with self.sess.as_default():
            self._run()

    # env_runnerを走らせてqueueに経験を貯める
    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise)
        while True:
            self.queue.put(next(rollout_provider), timeout=600.0)


# thread上でenvを走らせる本体
# nextで呼び出されるたびに経験をyieldする
def env_runner(env, policy, num_local_steps, summary_writer, render):
    # 初期状態
    last_state = env.reset()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        # num_local_steps分、または1エピソード分経験を貯める
        for _ in range(num_local_steps):
            # 行動、価値の取得
            fetched = policy.act(last_state)
            action, value_ = fetched[0], fetched[1]

            # argmaxしてるけど、actで返ってくるのは多項分布を利用した確率的行動選択によるactionの1hot
            # 次の状態、報酬、終端フラグなどをenvから取得
            action_ = action.argmax()
            print("Action: ", str(action_))
            state, reward, terminal, info = env.step(action_)

            if render:
                env.render()

            # rewardを(-1,1)でclip
            if reward != 0:
                reward = np.clip(reward, -1, 1)

            # 経験を蓄積
            # rolloutは経験らしい
            # 経験もだし、集合の一部に絞って絞って何がしかする事でもあるっぽい
            # 前の状態、行動、報酬、価値、終端状態
            rollout.add(last_state, action, reward, value_, terminal)
            length += 1
            rewards += reward
            last_state = state

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

            # 終端・初期化
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        # num_local_stepsでterminalにならなかった場合
        # 価値関数の推定値をrとして保存？
        if not terminal_end:
            rollout.r = policy.value(last_state)

        yield rollout


# ネットワーク、勾配の定義、threadのスタート、勾配の計算、ネットワークの更新
class A3C(object):
    def __init__(self, env, task, visualise, model_name, n_steps, gamma=0.9, lambda_=0.95):
        self.env = env
        self.task = task
        self.gamma=gamma
        self.lambda_=lambda_

        # ワーカーの識別子
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        # ネットワークのクラスをmodel_nameに応じて動的に切り替える
        model_cls = getattr(models, model_name)

        # グローバルネットワーク
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = model_cls(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        # ローカルネットワーク
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                pi = model_cls(env.observation_space.shape, env.action_space.n)
                self.local_network = pi
                pi.global_step = self.global_step


            # 勾配の計算

            # ac: action、アドバンテージ、報酬のplaceholder
            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            # log_softmax = logits - log(reduce_sum(exp(logits), dim))
            # https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # 方策勾配、advはprocess_rolloutで計算されたadvantage
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # 価値関数のloss
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))

            # ?
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            # batch size
            bs = tf.to_float(tf.shape(pi.x)[0])

            # GAE
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            self.runner = RunnerThread(env, pi, n_steps, visualise)

            # tf.gradients(x, y): xが勾配を計算するグラフ、yが計算に使う変数
            # 勾配
            grads = tf.gradients(self.loss, pi.var_list)

            # ログ
            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.scalar("model/entropy", entropy / bs)
            tf.summary.image("model/state", pi.x)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            self.summary_op = tf.summary.merge_all()

            # tf.assignによるパラメータのコピー（グローバルからローカル）
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            # [(grads[0], var_list[0]), (...), ...]の形にする
            grads_and_vars = list(zip(grads, self.network.var_list))

            # バッチ数分グローバルステップを加算
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # wobに従ってAdam
            opt = tf.train.AdamOptimizer(1e-4)

            # apply_gradients(x, y): xが勾配、yが勾配を使ってupdateする変数
            # tf.groupで複数のオペレーションを一度に実行できるopを定義できる
            # 以下の場合、勾配の計算とglobal_stepの加算
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            self.summary_writer = None
            self.local_steps = 0 # summary用

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
    # queueが空になるか終端状態を持つrolloutが来るまでrolloutを取り出す

        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        # 重みをグローバルからコピー
        sess.run(self.sync)
        # 経験の取り出し
        rollout = self.pull_batch_from_queue()
        # 経験を学習可能な形にする
        batch = process_rollout(rollout, self.gamma, self.lambda_)

        # task番号が0かlocal_stepが11の倍数ならsummaryを出力
        should_compute_summary = (self.task == 0 and self.local_steps % 11 == 0)

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            # summary_op, global_step
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1
