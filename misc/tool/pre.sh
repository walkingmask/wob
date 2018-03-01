#!/bin/bash
set -eu


# shutdown-script


# Terminate tmux
[ `ps aux | grep tmux | wc -l` -gt 1 ] && tmux kill-server

# Message for slack
MESSAGE="[`TZ=JST+15 date '+%Y-%m-%d %H:%M:%S'`] Preempted"

/usr/local/bin/report $MESSAGE
