import argparse

import numpy as np

from universe.spaces.vnc_event import PointerEvent

from event_readers import ReaderWrapper
from utils import find_fbs, dump


def _validate(x, y):
    if x < 10 or x > 170:
        return False
    if y < 125 or y > 285:
        return False
    return True


def _get_act_index(x, y, event):
    return int((y-125-4)/8)*60+int((x-10-4)/8)*3+event


def _generate_action(x, y, event):
    # no-op (move only)
    if event == 0:
        return [PointerEvent(x, y, 0)]
    # click
    elif event == 1:
        return [PointerEvent(x, y, 0),
                PointerEvent(x, y, 1),
                PointerEvent(x, y, 0)]
    # drag
    elif event == 2:
        return [PointerEvent(x, y, 1)]


def generate_action(act, event):
    return [_get_act_index(act.x, act.y, event), _generate_action(act.x, act.y, event)]


def convert(reader):
    dataset = []
    obs = None
    abuf = []
    last_act = [-1]

    for _obs, _acts in reader:

        if _obs is not None:
            # copy()しないと参照が渡される
            obs = _obs[75:75+210, 10:10+160].copy()

        if _acts != []:
            acts = []
            for act in _acts:
                # actのx,yが範囲外の場合は捨てる
                if not _validate(act.x, act.y):
                    continue

                # buffering
                if len(abuf) < 2:
                    abuf.append(act)
                else:
                    # abuf[0], abuf[1], actの並びで判定する
                    # 0,    1,    0,   cl
                    # 1,    1,    0,   no
                    # -,    0,    0,   no
                    # -,    1,    1,   dr
                    # -,    0,    1,    ?
                    if act.buttonmask == 0 and abuf[1].buttonmask == 1:
                        # click
                        if abuf[0].buttonmask == 0:
                            acts.append(generate_action(act, 1))
                        # no-op
                        if abuf[0].buttonmask == 1:
                            acts.append(generate_action(act, 0))
                    # no-op
                    elif act.buttonmask == 0 and abuf[1].buttonmask == 0:
                        acts.append(generate_action(act, 0))
                    # drag
                    elif act.buttonmask == 1 and abuf[1].buttonmask == 1:
                        acts.append(generate_action(act, 2))
                    # undefined
                    else:
                        pass

                    abuf[0] = abuf[1]
                    abuf[1] = act

            if obs is not None and len(acts) > 0:
                # datasetとしてはyは1つにしなければならない
                for act in acts:
                    # ここで、連続した同一actを排除
                    if act[0] != last_act[0]:
                        dataset.append([obs, act])
                        last_act = act

    return dataset


def _test_dataset(dataset):
    for i in range(len(dataset)):
        obs, act = dataset[i]
        assert obs is not None, "obs is None."
        assert act != [], "act is empty."
        assert obs.shape == (210, 160, 3), "obs is incorrect."
        assert 0 <= act[0] and act[0] <= 1200, "act index is incorrect."
        assert act[1] != [], "act[1] is incorrect."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('log_dir', help='path to the directory containing *.fbs.')
    parser.add_argument('-o', '--outfile', default='./dataset.bz2', help='path to output.')
    parser.add_argument('-p', '--paint_cursor', action='store_true', help='render cursor.')
    args = parser.parse_args()

    reader = ReaderWrapper(*find_fbs(args.log_dir), args.paint_cursor)
    dataset = convert(reader)
    _test_dataset(dataset)
    dump(dataset, args.outfile)

    exit()
