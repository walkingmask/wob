# envs (universe wrappers)


import logging
import time

import numpy as np

import cv2

import gym
from gym import spaces
from gym.spaces.box import Box

import universe
from universe import vectorized
from universe.spaces.vnc_event import KeyEvent, PointerEvent

# https://github.com/openai/universe/tree/master/universe/wrappers
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()


def create_env(env_id, client_id, remotes, **kwargs):
    """
    gym.spec(env_id).tagsの値に応じてenvを作成する

    client_id(int): ワーカーの識別番号みたいなん
    """
    tags = gym.spec(env_id).tags

    if tags.get('wob', False):
        return create_wob_env(env_id, client_id, remotes, **kwargs)
    elif tags.get('atari', False) and tags.get('vnc', False):
        return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    else:
        # Atariと断定
        return create_atari_env(env_id)


def create_wob_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = WoBAction(env)
#    env = WoBAction2(env)
    env = WoBObservation(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)

    env.configure(remotes=remotes, start_timeout=15 * 60, fps=12.0, client_id=client_id)

    return env


class WoBAction(vectorized.ActionWrapper):
    def __init__(self, env):
        super(WoBAction, self).__init__(env)

        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        ys = np.arange(75+50, 75+210, 8) + 4
        xs = np.arange(10, 10+160, 8) + 4
        for y in ys:
            for x in xs:
                # no-op (move only)
                self._actions.append([PointerEvent(x, y, 0)])
                # click
                self._actions.append([PointerEvent(x, y, 0),
                                PointerEvent(x, y, 1),
                                PointerEvent(x, y, 0)])
                # drag
                # x, y でボタンダウンするだけの実装。後に no-op がきて初めて意味をなす。
                # TODO: もっといい実装はないか？(内部で前の x, y を保存する？)
                self._actions.append([PointerEvent(x, y, 1)])

    def _action(self, action_n):
        return [self._actions[int(action)] for action in action_n]


class WoBAction2(vectorized.ActionWrapper):
    """
    ScrollとKeyEventも含めたAction wrapper
    """
    def __init__(self, env):
        super(WoBAction, self).__init__(env)

        self._generate_actions(env)
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self, env):
        self._actions = []

        # +4は画面の端を含めないようにするため
        ys = np.arange(75+50, 75+210, 8) + 4
        xs = np.arange(10, 10+160, 8) + 4

        for y in ys:
            for x in xs:
                # no-op (move only)
                self._actions.append([PointerEvent(x, y, 0)])
                # click
                self._actions.append([PointerEvent(x, y, 0),
                                      PointerEvent(x, y, 1),
                                      PointerEvent(x, y, 0)])
                # drag
                # x, y でボタンダウンするだけの実装。後に no-op がきて初めて意味をなす。
                # TODO: もっといい実装はないか？(内部で前の x, y を保存する？)
                self._actions.append([PointerEvent(x, y, 1)])
                # scroll-up
                self._actions.append([PointerEvent(x, y, 0),
                                      PointerEvent(x, y, 8),
                                      PointerEvent(x, y, 0)])
                # scroll-down
                self._actions.append([PointerEvent(x, y, 0),
                                      PointerEvent(x, y, 16),
                                      PointerEvent(x, y, 0)])

        for key_event in env.action_space._key_set:
            if key_event.down:
                if key_event.key < 255:
                    self._actions.append([KeyEvent(key_event.key), KeyEvent(key_event.key, down=False)])

        # often use cntrol keys
        self._actions.append(KeyEvent.build('ArrowLeft'))
        self._actions.append(KeyEvent.build('ArrowUp'))
        self._actions.append(KeyEvent.build('return'))
        self._actions.append(KeyEvent.build('right'))

    #    self._actions.appedn(build('tab'))
    #    self._actions.appedn(build('pgdn'))
    #    self._actions.appedn(build('pgup'))

        # Key combinations
        self._actions.append(KeyEvent.build('ctrl-a'))
        self._actions.append(KeyEvent.build('ctrl-c'))
        self._actions.append(KeyEvent.build('ctrl-x'))
        self._actions.append(KeyEvent.build('ctrl-v'))

    def _action(self, action_n):
        return [self._actions[int(action)] for action in action_n]


def _process_frame_wob(frame):
    """
    wob画面のみを切り取り、正規化する
    """
    frame = frame[75:75+210, 10:10+160, :].astype(np.float32)
    frame *= (1.0 / 255.0)
    return frame


class WoBObservation(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(WoBObservation, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [210, 160, 3])

    def _observation(self, observation_n):
        return [_process_frame_wob(observation) for observation in observation_n]


def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env


def create_atari_env(env_id):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env


# envのinfoのラッパー

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log


# AtariのObservationラッパー

def _process_frame42(frame):
    """
    Atariの画面160x160を42x42にリサイズ
    """
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]
