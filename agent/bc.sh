#!/usr/bin/env bash

function usage() {
    echo "usage: $0 DATASET_DIR LOG_DIR [-e EPOCH_SIZE] [-m MODEL] [-h]" 1>&2
    exit 1
}

[ $# -lt 2 ] && usage

DATASET_DIR=$1
[ ! -d $DATASET_DIR ] && usage || shift

LOG_DIR=$1
[ "${LOG_DIR:0:1}" = "-" ] && usage || shift 

EPOCH_SIZE=0
MODEL="FFPolicy"

# args
while getopts e:m:h OPT
do
  case $OPT in
    "e" ) EPOCH_SIZE=$((OPTARG*1)) ;;
    "m" ) MODEL="$OPTARG" ;;
    "h" ) usage ;;
  esac
done

cd ./py

mkdir -p $LOG_DIR

# run with tmux
tmux new-session -s bc -n htop -d bash

for task in `ls $DATASET_DIR`; do
    task=${task%.*}
    task_=${task##*.}
    mkdir -p $LOG_DIR/$task
    tmux new-window -t bc -n $task_ bash
    sleep 1
    tmux send-keys -t bc:$task_ \
        "python bc.py $DATASET_DIR/$task.bz2 $LOG_DIR/$task -e $EPOCH_SIZE --model $MODEL" \
        Enter
done

tmux send-keys -t bc:htop 'htop' Enter

# waiting
while true; do
    if [ "`ps aux | grep python | grep bc.py | wc -l`" -lt 1 ]; then
        break
    fi
    sleep 60
done

# clean up
tmux kill-session
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
mkdir $LOG_DIR/log
mv /tmp/universe-* $LOG_DIR/log/

# if run on GCE, then shutodown right now
[ "`hostname`" = "wob" ] && sudo shutdown -h now
