import argparse
import bz2
import errno
import os
from pathlib import PosixPath
import pickle
from random import randint

import numpy as np

import cv2

from event_readers import FBSEventReader


def find_fbs(path):
    observation_path = path+'/server.fbs'
    if not PosixPath(observation_path).expanduser().exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), observation_path)

    action_path = path+'/client.fbs'
    if not PosixPath(action_path).expanduser().exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), action_path)

    return observation_path, action_path


def load(fname):
    with bz2.open(fname, 'rb') as fh:
        return pickle.load(fh)


def dump(obj, fname):
    with bz2.open(fname, 'wb') as fh:
        pickle.dump(obj, fh, pickle.HIGHEST_PROTOCOL)


def count(reader):
    oi, ai = 0, 0
    for n in reader:
        obs = n['observation']
        act = n['action']
        if obs is not None:
            oi += 1
        if act != []:
            ai += 1

    print("the total number of frames is", oi)
    print("the total number of actions is", ai)


def play_demo(reader):
    cv2.namedWindow("DemoPlayer", cv2.WINDOW_NORMAL)

    for n in reader:
        timestamp = n['timestamp']
        obs = n['observation']
        act = n['action']

        if act != []:
            print(timestamp, act)
        elif obs is not None:
            print(timestamp, "obs")
            frame = obs[75:75+210, 10:10+160]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("DemoPlayer", frame)
        else:
            print(timestamp)

        keycode = cv2.waitKey()

        if keycode == ord('q'):
            break

    cv2.destroyAllWindows()


def demo2mov(reader, outfile='./demo.mov', fps=60):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mov = cv2.VideoWriter(outfile, fourcc, fps, (160, 210))

    for n in reader:
        obs = n['observation']

        if obs is None:
            pass
        else:
            frame = obs[75:75+210, 10:10+160]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mov.write(frame)


def play_dataset(dataset_path):
    dataset = load(PosixPath(dataset_path).expanduser())

    print("dataset index, act index, act, x-10, y-75")
    cv2.namedWindow("DSPlayer", cv2.WINDOW_NORMAL)

    len_dataset = len(dataset)

    while True:
        i = randint(0, len_dataset)
        obs, act = dataset[i]

        print(i, act[0], act[1], act[1][0].x-10, act[1][0].y-75)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("DSPlayer", obs)

        keycode = cv2.waitKey()
        if keycode == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('log_dir', help='path to the directory containing *.fbs.')
    args = parser.parse_args()

    log_dir = args.log_dir
    reader = FBSEventReader(*find_fbs(log_dir), paint_cursor=True)

    def reload():
        global reader
        reader = FBSEventReader(*find_fbs(log_dir), paint_cursor=True)

#    count(reader)
#    play_demo(reader)
#    demo2mov(reader, '~/Desktop/demo.mov')
