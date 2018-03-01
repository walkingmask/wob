#!/usr/bin/env python
# coding: utf-8


# worker.py起動用のコマンドを生成して走らせる


import argparse
import os
import sys

from six.moves import shlex_quote


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="wob.mini.ClickDialog2-v0",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                    help="Log directory path")
parser.add_argument('-n', '--dry-run', action='store_true',
                    help="Print out commands rather than executing them")
parser.add_argument('-m', '--mode', type=str, default='tmux',
                    help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")
# ckptの指定を追加
parser.add_argument('-c', '--ckpt', type=str, default=None,
                    help='Path to checkpoint (like "model.ckpt-xxxxxxx")')
# modelの指定を追加
parser.add_argument('--model', type=str, default=None, help='Specify model')
# num_global_stepsの指定を追加
parser.add_argument('--num-global-steps', type=int, default=1000000,
                    help='Specify number of global steps')
# n_stepsの指定を追加
parser.add_argument('--n-steps', type=int, default=200, help='n_steps')
# gamma,lambda_の指定を追加
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--lambda_', type=float, default=0.95, help='lambda')


# modeに応じてコマンド生成
def new_cmd(session, name, cmd, mode, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)

    if mode == 'tmux':
        return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))
    elif mode == 'child':
        return name, "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, logdir, session, name, logdir)
    elif mode == 'nohup':
        return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd), logdir, session, name, logdir)


# commands生成
# dry-runで生成されるコマンドを確認できる
def create_commands(session,
                    num_workers,
                    remotes,
                    env_id,
                    logdir,
                    shell='bash',
                    mode='tmux',
                    visualise=False,
                    ckpt=None,
                    model=None,
                    num_global_steps=1000000,
                    n_steps=200,
                    gamma=0.9,
                    lambda_=0.95):

    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--log-dir', logdir,
        '--env-id', env_id,
        '--num-workers', str(num_workers),
        '--num-global-steps', str(num_global_steps),
        '--n-steps', str(n_steps),
        '--gamma', str(gamma),
        '--lambda_', str(lambda_)]

    if visualise:
        base_cmd += ['--visualise']

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    if ckpt is not None:
        base_cmd += ['--ckpt', ckpt]

    if model is not None:
        base_cmd += ['--model', model]

    # ps
    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job-name", "ps"], mode, logdir, shell)]
    # worker
    for i in range(num_workers):
        cmds_map += [new_cmd(session,
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]], mode, logdir, shell)]
    # tensorboard
    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", logdir, "--port", "12345"], mode, logdir, shell)]
    # htop
    if mode == 'tmux':
        cmds_map += [new_cmd(session, "htop", ["htop"], mode, logdir, shell)]

    # tmux window用
    windows = [v[0] for v in cmds_map]

    # 実行時の説明文
    notes = []
    # コマンド本体
    cmds = [
        # ログ用ディレクトリの作成
        "mkdir -p {}".format(logdir),
        # プログラム実行文を保存
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), logdir),
    ]

    if mode == 'nohup' or mode == 'child':
        cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(logdir)]
        notes += ["Run `source {}/kill.sh` to kill the job".format(logdir)]
    if mode == 'tmux':
        notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
        notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    else:
        notes += ["Use `tail -f {}/*.out` to watch process output".format(logdir)]
    notes += ["Point your browser to http://localhost:12345 to see Tensorboard"]

    if mode == 'tmux':
        cmds += [
        "kill $( lsof -i:12345 -t ) > /dev/null 2>&1",  # kill any process using tensorboard's port
        "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(num_workers+12222), # kill any processes using ps / worker ports
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
        ]
        for w in windows[1:]:
            cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
        cmds += ["sleep 1"]

    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds, notes


# 実行ポイント
def run():
    # 引数処理
    args = parser.parse_args()

    # コマンド群生成
    cmds, notes = create_commands("a3c",
                                  args.num_workers,
                                  args.remotes,
                                  args.env_id,
                                  args.log_dir,
                                  mode=args.mode,
                                  visualise=args.visualise,
                                  ckpt=args.ckpt,
                                  model=args.model,
                                  num_global_steps=args.num_global_steps,
                                  n_steps=args.n_steps,
                                  gamma=args.gamma,
                                  lambda_=args.lambda_)

    if args.dry_run:
        print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")

    # 実行するコマンド群をプリント
    print("\n".join(cmds))
    print("")

    # dry-runじゃなければ実行
    if not args.dry_run:
        if args.mode == "tmux":
            os.environ["TMUX"] = ""
        os.system("\n".join(cmds))

    # noteの出力
    print('\n'.join(notes))


if __name__ == "__main__":
    run()
