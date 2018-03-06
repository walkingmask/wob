#!/usr/bin/env python
# coding:utf-8


# Play

# [Tips]
# For make a play video
# $ python play.py env_id -c /pathto/ckpt -s 100 -i /pathto/imgs
# $ convert -delay 5 -loop 0 /pathto/imgs/play*.png ./play.gif


import argparse
import bz2
import logging
import os
import pickle
import random
import sys
import signal

import numpy as np

import cv2

import go_vncdriver
import tensorflow as tf

from envs import create_wob_env
import models
from worker import FastSaver


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Player(object):
    def __init__(self, env, model_name, img_dir, render=False):
        self.env = env
        self.render = render

        model_cls = getattr(models, model_name)
        with tf.variable_scope("global"):
            self.network = model_cls(env.observation_space.shape, env.action_space.n)

        logger.info("Used model is %s.", model_cls.__name__)

        self.img_flag = False
        if img_dir is not None:
            os.makedirs(img_dir)
            self.img_dir = img_dir
            self.img_flag = True

    def start(self, sess, max_steps=100000):
        self.sess = sess
        self.max_steps = max_steps
        self.step = 0

    def run(self, sv):
        with self.sess.as_default():
            self._run(sv)

    def _run(self, sv):
        env = self.env

        state = env.reset()

        if self.img_flag:
            states = [state]

        eps = 1
        length = 0
        num_rewards = 0
        num_positive_rewards = 0

        while True:
            if sv.should_stop():
                break

            if self.step > self.max_steps:
                break

            action = self.network.act(state)[0].argmax()
#            action = self.network.act_max(state)[0]
            print(action)

            state, reward, terminal, info = env.step(action)

            if self.img_flag:
                states.append(state)

            if self.render:
                env.render()

            if reward != 0:
                logger.info("[wob player] [reward] %f", reward)
                num_rewards += 1
                if reward > 0:
                    num_positive_rewards += 1

            if terminal:
                state = env.reset()

                if self.img_flag:
                    states.append(state)

                logger.info("[wob player] [episode] %d finished. length: %d. at step %d."
                                % (eps, length, self.step))

                eps += 1
                length = 0

            self.step += 1
            length += 1

        logger.info("[wob player] [SR] Finished playing. eps: %d. num rewards: %d. num positive rewards: %d. SR: %f"
                    % (eps, num_rewards, num_positive_rewards, num_positive_rewards/num_rewards))

        if self.img_flag:
            self.store(states)

    def store(self, frames):
        for i, frame in enumerate(frames):
            frame *= 255
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_file = self.img_dir + "/play%05d.png" % i
            cv2.imwrite(img_file, frame)
        logger.info("Store states to %s." % img_file)


def run(args):
    logger.info("Play started.")
    logger.info("[wob player] [env] %s", args.env_id)

    if args.ckpt is not None:
        logger.info("[wob player] [ckpt] %s", args.ckpt)

    env = create_wob_env(args.env_id, None, None)

    with tf.Graph().as_default():
        player = Player(env, args.model, args.img_dir, args.visualise)

        variables_to_play = [v for v in tf.global_variables()]
        init_op = tf.variables_initializer(variables_to_play)
        init_all_op = tf.global_variables_initializer()

        saver = FastSaver(variables_to_play)

        def init_all(sess):
            logger.info("Initializing all parameters.")
            sess.run(init_all_op)

        def load_pretrain(sess):
            init_all(sess)
            logger.info("Restoring parameters.")
            saver.restore(sess, args.ckpt)

        if args.ckpt is None:
            init_fn = init_all
        else:
            init_fn = load_pretrain

        sv = tf.train.Supervisor(summary_op=None,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 ready_op=tf.report_uninitialized_variables(variables_to_play))

        with sv.managed_session() as sess:
            player.start(sess, args.max_steps)
            player.run(sv)

        sv.stop()
        logger.info('Reached %s steps. worker stopped.', player.step)


def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', help='environment id to play.')
    parser.add_argument('-c', '--ckpt', default=None, help='path to checkpoint (like "model.ckpt-xxxxxxx").')
    parser.add_argument('-s', '--max-steps', type=int, default=100000, help='max steps.')
    parser.add_argument('--visualise', action='store_true', help="visualise the environment.")
    parser.add_argument('--model', default='FFPolicy', help='specify model.')
    parser.add_argument('-i', '--img-dir', default=None, help='path to directory to store images.')
    args = parser.parse_args()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    run(args)


if __name__ == "__main__":
    tf.app.run()
