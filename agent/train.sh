#!/usr/bin/env bash

function usage() {
    echo "usage: $0 CKPT_DIR LOG_DIR [-a NUM_TRAINS]\n" \
         "          [-w NUM_WORKERS] [-m MODEL]\n" \
         "          [-s GLOBAL_STEPS] [-n N_STEPS]\n" \
         "          [-g GAMMA] [-l LAMBDA] [-h]" 1>&2
    exit 1
}

function ckpt_not_found() {
    echo "$CKPT_DIR is not found." 1>&2
    exit 1
}

[ $# -lt 2 ] && usage

CKPT_DIR=$1
[ ! -d $CKPT_DIR ] && ckpt_not_found || shift

LOG_DIR=$1
[ "${LOG_DIR:0:1}" = "-" ] && usage || shift 

NUM_TRAINS=3
NUM_WORKERS=12
MODEL="FFPolicy"
GLOBAL_STEPS=1000000
N_STEPS=200
GAMMA=0.9
LAMBDA=0.95

# args
while getopts a:w:m:s:n:g:l:h OPT
do
  case $OPT in
    "a" ) NUM_TRAINS=$OPTARG ;;
    "w" ) NUM_WORKERS=$OPTARG ;;
    "m" ) MODEL=$OPTARG ;;
    "s" ) GLOBAL_STEPS=$OPTARG ;;
    "n" ) N_STEPS=$OPTARG ;;
    "g" ) GAMMA=$OPTARG ;;
    "l" ) LAMBDA=$OPTARG ;;
    "h" ) usage ;;
  esac
done

cd ./py

mkdir -p $LOG_DIR

for task in `ls $CKPT_DIR`; do
    for i in `seq $NUM_TRAINS`; do
        logdir="$LOG_DIR/$task/$i"
        mkdir -p $logdir
        ckpt="`ls $CKPT_DIR/$task | grep index`"
        ckpt=${ckpt%%.*}
        python train.py -w $NUM_WORKERS -e $task -l $logdir -c $CKPT_DIR/$task/$ckpt --model $MODEL \
            --num-global-steps $GLOBAL_STEPS --n-steps $N_STEPS --gamma $GAMMA --lambda $LAMBDA
        while true; do
            if [ "`ps aux | grep python | grep "job-name worker" | wc -l`" -lt 1 ]; then
                break
            fi
            sleep 60
        done
        tmux kill-session
        pkill python
        docker stop $(docker ps -aq)
        docker rm $(docker ps -aq)
        mkdir $logdir/log
        mv /tmp/universe-* $logdir/log/
    done
done

[ "`hostname`" = "wob" ] && sudo shutdown -h now
