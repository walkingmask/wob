#!/usr/bin/env python


"""
This code is based on what was in the universe wob's docker image.
image: quay.io/openai/universe.world-of-bits:0.20.0
path: /app/gym-demonstration/gym_demonstration/event_readers.py
"""


import argparse
import heapq
import json
import logging

from universe import spaces
from universe import utils
from universe.vncdriver import fbs_reader, vnc_proxy_server, vnc_client
from universe import spaces as vnc_spaces


logger = logging.getLogger(__name__)


class Error(Exception):
    pass


class InvalidEventLogfileError(Error):
    pass


class FakeTransport(object):
    def loseConnection(self):
        pass

    def write(self, data):
        pass


class Factory(object):
    def __init__(self, error_buffer):
        self.error_buffer = error_buffer
        self.deferred = None


class MergedEventReader(object):
    def __init__(self, *timestamped_iterables):
        """
        Args:
            timestamped_iterables: A set of iterables that return dictionaries. Each dictionary must contain a 'timestamp' key that is monotonically increasing
        """

        self.merged_iterables = heapq.merge(*timestamped_iterables, key=self._extract_timestamp)
        self.timestamp = None

    @staticmethod
    def _extract_timestamp(line):
        if not isinstance(line, dict):
            raise InvalidEventLogfileError("MultiIterator received a line that's not a dictionary: {}".format(line))
        try:
            return float(line['timestamp'])
        except KeyError:
            raise InvalidEventLogfileError("MultiIterator received a line without a timestamp: {}".format(line))

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next item from our iterable lists, sorted by timestamp
        """
        data = next(self.merged_iterables)
        timestamp = data['timestamp']
        if self.timestamp and (timestamp < self.timestamp):
            raise InvalidEventLogfileError("MultiIterator received a line with an out of order timestamp: {}".format(data))

        self.timestamp = timestamp
        return data


class ActionQueue(object):
    def __init__(self):
        self.actions = []
        self.pixel_format = []

    def key_event(self, key, down):
        event = spaces.KeyEvent(key, down)
        self.actions.append(event)

    def pointer_event(self, x, y, buttonmask):
        event = spaces.PointerEvent(x, y, buttonmask)
        self.actions.append(event)

    def set_pixel_format(self, server_pixel_format):
        self.pixel_format.append(server_pixel_format)

    def pop_all(self):
        output = self.actions, self.pixel_format
        self.actions = []
        self.pixel_format = []
        return output


class FBSActionReader(object):
    def __init__(self, action_path):
        self.fbs_reader = iter(fbs_reader.FBSReader(action_path))
        self.action_queue = ActionQueue()
        self.error_buffer = utils.ErrorBuffer()
        self.action_processor = vnc_proxy_server.VNCProxyServer(
            action_queue=self.action_queue,
            error_buffer=self.error_buffer,
            enable_logging=False,
        )
        self.action_processor.transport = FakeTransport()

        self._advance()

    def __iter__(self):
        return self

    def _advance(self):
        self._data, self._timestamp = self.fbs_reader.__next__()

    def next(self):
        return self.__next__()

    def __next__(self):
        if self._timestamp is None:
            raise StopIteration

        timestamp = None
        action = []
        pixel_format = []

        eof = False
        while not (eof or action or pixel_format):
            # Keep cycling until we find something interesting to report

            timestamp = self._timestamp
            while self._timestamp == timestamp:
                assert self._data is not None
                # Advance to the time where this action happened
                self.action_processor.dataReceived(self._data)
                self.error_buffer.check()
                new_action, new_pixel_format = self.action_queue.pop_all()
                action += new_action
                pixel_format += new_pixel_format

                # Advance to the next action
                self._advance()
                if self._timestamp is None:
                    # End of the line. Might have some
                    # action/pixel_formats to return, so don't raise
                    # StopIteration here.
                    eof = True
                    break

        if not (action or pixel_format):
            raise StopIteration
        else:
            return {
                'timestamp': timestamp,
                'action':action,
                'pixel_format': pixel_format
            }

class RawFBSObservationReader(object):
    """
    Thin wrapper around FBSReader
    """
    def __init__(self, observation_path):
        self._reader = fbs_reader.FBSReader(observation_path)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        observation, timestamp = next(self._reader)
        return {
            'timestamp': timestamp,
            'observation': observation,
        }


class FBSEventReader(object):
    """
    Outputs both actions and observations from FBS files
    """
    def __init__(self, observation_path, action_path, paint_cursor=False):
        self.paint_cursor = paint_cursor

        observation_reader = RawFBSObservationReader(observation_path)

        self.error_buffer = utils.ErrorBuffer()
        action_reader = FBSActionReader(action_path)

        self.merged_reader = MergedEventReader(observation_reader, action_reader)

        # Set up state for our observation reader
        # We need to keep this state in Playback because reading in from server.fbs depends on
        # pixel_format messages in client.fbs
        self.pixel_format = []
        self.observation_processor = vnc_client.VNCClient()
        self.observation_processor.factory = Factory(self.error_buffer)
        self.observation_processor.transport = FakeTransport()

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        step_data = next(self.merged_reader)

        action = step_data.get('action', [])
        pixel_format = step_data.get('pixel_format', None)
        raw_observation = step_data.get('observation', None)

        # Apply stateful data
        self._apply_options()
        self._apply_pixel_format(pixel_format)
        observation, obs_info = self._apply_observation(raw_observation)

        if self.observation_processor.numpy_screen is not None:
            for a in action:
                self.observation_processor.numpy_screen.apply_action(a)

        return {
            'timestamp': self.merged_reader.timestamp,
            'observation': observation,
            'action': action,
            'info': obs_info,
        }

    def _apply_options(self):
        """
        allows us to set various options for the playback/rendering that
        might also potentially change over time.
        """
        if self.observation_processor.numpy_screen is not None:
            self.observation_processor.numpy_screen.set_paint_cursor(self.paint_cursor)

    def _apply_observation(self, raw_observation):
        if not raw_observation:
            return None, {}

        # Any SetPixelFormat messages have now come into effect.
        for fmt in self.pixel_format:
            assert self.observation_processor.framebuffer
            assert len(fmt) == 16, "Bad length {} for pixel format: {}".format(len(fmt), fmt)
            self.observation_processor.framebuffer.apply_format(fmt)

        self.observation_processor.dataReceived(raw_observation)
        self.error_buffer.check()

        if self.observation_processor.numpy_screen is None:
            return None, {}

        # We have an observation
        observation, raw_obs_info = self.observation_processor.numpy_screen.flip()
        return observation, {'stats.playback.vnc.updates.n': len(raw_obs_info['vnc_session.framebuffer_updates'])}


    def _apply_pixel_format(self, pixel_format):
        if pixel_format:
            self.pixel_format += pixel_format


class ReaderWrapper(object):
    def __init__(self, observation_path, action_path, paint_cursor=False):
        self.reader = FBSEventReader(observation_path, action_path, paint_cursor)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        n = self.reader.next()
        return n['observation'], n['action']


if __name__ == '__main__':
    from utils import find_fbs

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('log_dir', help='path to the directory containing *.fbs.')
    parser.add_argument('-p', '--paint_cursor', action='store_true', help='render cursor.')
    args = parser.parse_args()

    reader = ReaderWrapper(*find_fbs(args.log_dir), args.paint_cursor)
