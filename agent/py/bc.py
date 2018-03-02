#!/usr/bin/env python
# coding:utf-8


# Behavioral Cloning


import argparse
import bz2
import logging
import os
import pickle
import random
import sys
import signal

import go_vncdriver
import tensorflow as tf

import models
from worker import FastSaver


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataSet(object):
    def __init__(self, path, batch_size=32):
        self.batch_size = batch_size
        self.observation_space = (210, 160, 3)
        self.action_space = 1200
        dataset = self._load(path)
        self._load_dataset(dataset)
        self.len_dataset = len(self.dataset)
        self._train_test_split()
        logger.info("dataset loaded.")
        logger.info("   dataset size: {}, batch size: {},".format(self.len_dataset, self.batch_size))
        logger.info("   train size: {}, test size: {}.".format(self.len_train, self.len_test))

    def _train_test_split(self):
        len_dataset = self.len_dataset

        len_train = int(0.8*len_dataset)
        len_test  = len_dataset - len_train
        random.shuffle(self.dataset)
        self.train = self.dataset[:len_train]
        self.test = self.dataset[len_train:]
        self.len_train = len_train
        self.len_test = len_test

    def get_train_batch(self):
        batch = random.sample(self.train, self.batch_size)
        return list(zip(*batch))

    def get_test_batch(self):
        batch = random.sample(self.test, self.batch_size)
        return list(zip(*batch))

    def _load_dataset(self, dataset):
        X, Y = [], []
        for data in dataset:
            obs, act = data
            X.append(obs * (1.0 / 255.0))
            Y.append(act[0])
        self.dataset = list(zip(X, Y))

    def _load(self, fname):
        with bz2.open(fname, 'rb') as fh:
            return pickle.load(fh)


class BC(object):
    def __init__(self, dataset, model_name, x10=True):
        self.dataset = dataset
        batch_size = dataset.batch_size

        model_cls = getattr(models, model_name)

        with tf.variable_scope("global"):
            pi = model_cls(dataset.observation_space, dataset.action_space)
            self.network = pi

        self.step = tf.get_variable("step", [], tf.int32,
                            initializer=tf.constant_initializer(0, dtype=tf.int32),
                            trainable=False)

        # 教師行動のインデックス
        self.act = tf.placeholder(tf.int32, [None], name="y_act")

        if x10:
            # wobに従い、特定のアクションのlossをx10する
            act_1hot = tf.one_hot(self.act, dataset.action_space)
            x10 = [10 if e == 1 else 1 for i in range(20) for j in range(20) for e in range(3)] # click only
    #        x10 = [1 if e == 0 else 10 for i in range(20) for j in range(20) for e in range(3)] # click, drag
    #        x10 += [01 for i in range(1200, len(dataset.action_space))] # KeyEvnet
            x10 = tf.constant(x10, dtype=tf.float32)
            cross_entoropy_x10 = tf.reduce_sum(tf.nn.log_softmax(pi.logits) * act_1hot * x10, axis=1)
            self.loss = - tf.reduce_mean(cross_entoropy_x10, name='loss')
        else:
            # 普通のloss
            self.loss = - tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                labels=self.act, logits=pi.logits), name='loss')

        bs = tf.to_float(batch_size)

        # パラメータはwobに従う
        opt = tf.train.AdamOptimizer(1e-3)
        grads = opt.compute_gradients(self.loss)

        tf.summary.scalar("model/loss", self.loss / bs)
        self.summary_op = tf.summary.merge_all()

        inc_step = self.step.assign_add(batch_size)

        self.train_op = tf.group(opt.apply_gradients(grads), inc_step)

        self.summary_writer = None
        self.step_const = 0

    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer

    def process(self, sess):
        batch = self.dataset.get_train_batch()

        should_compute_summary = (self.step_const % 11 == 0)

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.step]
        else:
            fetches = [self.train_op, self.step]

        feed_dict = {
            self.network.x: batch[0],
            self.act: batch[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        self.step_const = fetched[-1]

        return fetched[-1]

    def validate(self, sess):
        step = 0
        losses = []

        while step < self.dataset.len_test:
            batch = self.dataset.get_test_batch()
            feed_dict = { self.network.x: batch[0], self.act: batch[1] }
            losses.append(sess.run(self.loss, feed_dict))
            step += self.dataset.batch_size

        loss = sum(losses)/step

        summary = tf.Summary()
        summary.value.add(tag='validation loss', simple_value=float(loss))
        self.summary_writer.add_summary(summary, self.step_const)
        self.summary_writer.flush()

        return loss


def run(args):
    dataset = DataSet(args.dataset)

    len_epoch = dataset.len_train
    epoch_size = args.epoch_size

    if   epoch_size == 0:
        num_ckpt = 1
        check_freq = 1
        max_epochs = 1
    elif epoch_size == 1:
        num_ckpt = 10
        check_freq = 1
        max_epochs = 10
    elif epoch_size == 2:
        num_ckpt = 10
        check_freq = 10
        max_epochs = 100
    else:
        raise ValueError('%d is invalid epoch size.' % epoch_size)

    ckpt_path = os.path.join(args.log_dir, 'ckpt')
    summary_path = os.path.join(args.log_dir, 'summary')

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    with tf.Graph().as_default():
        trainer = BC(dataset, args.model, args.no_x10)

        variables_to_save = [v for v in tf.global_variables()]

        logger.info('Trainable vars:')
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name):
            logger.info('  %s %s', v.name, v.get_shape())

        saver = FastSaver(variables_to_save, max_to_keep=num_ckpt)

        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()

        def init_fn(ses):
            logger.info("Initializing all parameters.")
            ses.run(init_all_op)

        summary_writer = tf.summary.FileWriter(summary_path)
        logger.info("Events directory: %s.", summary_path)

        sv = tf.train.Supervisor(logdir=None,
                                 saver=saver,
                                 summary_op=None,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 summary_writer=summary_writer,
                                 ready_op=tf.report_uninitialized_variables(variables_to_save))

        with sv.managed_session() as sess, sess.as_default():
            trainer.start(sess, summary_writer)
            step = sess.run(trainer.step)
            logger.info("Starting training at step=%d", step)

            epoch = 0

            while not sv.should_stop():
                step = trainer.process(sess)

                # update epoch and save ckeckpoint
                if epoch < int(step/len_epoch):
                    epoch += 1

                    if epoch%check_freq == 0:
                        save_path = ckpt_path + '/epoch-' + str(epoch)
                        sv.saver.save(sess, save_path)

                if epoch >= max_epochs:
                    break

                # progress and validation
                if step % (32*10) == 0:
                    logger.info("epoch: {}, total step: {}, validation loss: {}.".format(
                                    epoch, step, trainer.validate(sess)))

        sv.stop()
        logger.info('reached %s steps. worker stopped.', step)


def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('dataset', help='path to dataset file (.bz2).')
    parser.add_argument('log_dir', help='path to log directory.')
    parser.add_argument('-e', '--epoch-size', type=int, default=1,
                        help='epoch size. choices are 0(1 epoch), 1(10 epochs) ,2(100 epochs).')
    parser.add_argument('--model', default='FFPolicy', help='specify model.')
    parser.add_argument('--no-x10', action='store_false')
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
