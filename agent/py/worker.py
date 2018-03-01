#!/usr/bin/env python
# coding:utf-8


# train.pyでnum_workersの数に応じて、これが複数プロセス生成される
# 主に走るのはここ
# クラスター、SuperVisor、sessやpsなどの設定


import argparse
import logging
import os
import sys
import signal
import time

import cv2

import go_vncdriver # tfの前にimportしないと怒られる
import tensorflow as tf

from a3c import A3C
from envs import create_env


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# tf.train.Saverは学習パラメータ保存用class
# 諸々の理由によりwrapper書いてるらしい
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


# server: tf.train.Server
def run(args, server):
    # envとエージェントの作成
    env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)
    trainer = A3C(env, args.task, args.visualise, args.model, args.n_steps, args.gamma, args.lambda_)

    # save対象の変数
    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]

    # 初期化に関するop
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save, max_to_keep=1)
    def init_all(sess):
        logger.info("Initializing all parameters.")
        sess.run(init_all_op)

    # tf.ConfigProto: tfのsessionのパラメータの設定
    # https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/protobuf/config.proto#L206
    # device_filters: わからん、ConfigProtoのドキュメント空だし
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    # 学習用ログディレクトリ
    logdir = os.path.join(args.log_dir, 'train')

    # tfの学習ログハンドラ
    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)

    # 保存場所のログ
    logger.info("Events directory: %s_%s", logdir, args.task)

    # ckptをrestoreするopとか
    if args.ckpt is not None and not tf.train.get_checkpoint_state(logdir):
        excluded_variables = [
            'global/value/w/Adam:0',
            'global/value/w/Adam_1:0',
            'global/value/b/Adam:0',
            'global/value/b/Adam_1:0',
            'global/global_step:0',
        ]
        variables_to_restore = [v for v in variables_to_save if not v.name in excluded_variables]
        saver_to_restore = FastSaver(variables_to_restore)
        def restore_after_init_all(sess):
            init_all(sess)
            logger.info("Restoring parameters.")
            saver_to_restore.restore(sess, args.ckpt)
        init_fn = restore_after_init_all
    else:
        init_fn = init_all

    # tf.train.Supervisor: モデルやサマリーの保存なんかをよしなにやってくれるやつ
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = args.num_global_steps

    # 初期情報ログ
    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")

    # managed_session: svの制御下でのsession
    # sess.as_default(): sessをメインのsessとして使う
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        # グローバル変数をローカルにコピー
        sess.run(trainer.sync)
        # スレッドの開始
        trainer.start(sess, summary_writer)
        # global_stepの取得と記録
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        # should_stop: スレッドが停止したらTrueになる
        # num_global_stepsまで
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            # 勾配の更新など
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # 全てのworkerが止まるように要請
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)


# パラメータサーバとワーカのhost ipとportの設定
def cluster_spec(num_workers, num_ps):

    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):

    # プログラムの引数の設定
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('--env-id', default="wob.mini.ClickDialog2-v0", help='Environment id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")
    # ckptの指定を追加
    parser.add_argument('-c', '--ckpt', default=None, help='Path to checkpoint (like "model.ckpt-xxxxxxx")')
    # modelの指定を追加
    parser.add_argument('--model', default='FFPolicy', help='Specify model')
    # num_global_stepsの指定を追加
    parser.add_argument('--num-global-steps', type=int, default=1000000,
                        help='Number of global steps')
    # n_stepsの指定を追加
    parser.add_argument('--n-steps', type=int, default=200, help='n_steps')
    # gamma,lambda_の指定を追加
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--lambda_', type=float, default=0.95, help='lambda')
    args = parser.parse_args()

    # クラスターの設定
    spec = cluster_spec(args.num_workers, 1)
    # tf.train.ClusterSpec: Represents a cluster as a set of "tasks", organized into "jobs".
    # https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
    # as_cluster_def(): Returns a tf.train.ClusterDef protocol buffer based on this cluster.
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    # 終了シグナルの定義
    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # tf.train.Server: An in-process TensorFlow server, for use in distributed training.
    # https://www.tensorflow.org/api_docs/python/tf/train/Server

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    # tf.app.run(): コマンドラインオプション定義できるらしい
    # https://qiita.com/kikusumk3/items/47a47bfc1931fd572cd8
    tf.app.run()
