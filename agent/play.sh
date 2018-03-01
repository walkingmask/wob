#!/usr/bin/env bash

function cleanup() {
    tmux kill-session
    pkill python
    docker stop $(docker ps -aq)
    docker rm $(docker ps -aq)
}

function usage() {
    echo "usage: $0 MODE CKPT_DIR LOG_DIR [-s NUM_STEPS] [-m MODEL] [-h]" 1>&2
    exit 1
}

function ckpt_not_found() {
    echo "$CKPT_DIR is not found." 1>&2
    exit 1
}

[ $# -lt 3 ] && usage

MODE=$1
shift

CKPT_DIR=$1
[ ! -d $CKPT_DIR ] && ckpt_not_found || shift

LOG_DIR=$1
[ "${LOG_DIR:0:1}" = "-" ] && usage || shift 

NUM_STEPS=100000
MODEL="FFPolicy"

# args
while getopts s:m:h OPT
do
  case $OPT in
    "s" ) NUM_STEPS=$((OPTARG*1)) ;;
    "m" ) MODEL="$OPTARG" ;;
    "h" ) usage ;;
  esac
done

cd ./py

tmux new-session -s play -n htop -d bash
sleep 1
tmux send-keys -t play:htop 'htop' Enter

if [ "$MODE" = "bc" ]; then

    for task in `ls $CKPT_DIR`; do
        task_=${task##*.} # remove "wob.mini."
        ckpt="`ls $CKPT_DIR/$task | grep index`"
        ckpt=${ckpt%.*} # remove ".index"
        tmux new-window -t play -n $task_ bash
        sleep 1
        tmux send-keys -t play:$task_ \
            "python play.py $task -s $NUM_STEPS -c $CKPT_DIR/$task/$ckpt --model $MODEL" \
            Enter
    done

elif [ "$MODE" = "a3c" ]; then

    for task in `ls $CKPT_DIR`; do
        task_=${task##*.} # remove "wob.mini."
        for i in `ls $CKPT_DIR/$task`; do
            ckpt="`ls $CKPT_DIR/$task/$i | grep index`"
            ckpt=${ckpt%.*} # remove ".index"
            tmux new-window -t play -n ${task_}-$i bash
            sleep 1
            tmux send-keys -t play:${task_}-$i \
                "python play.py $task -s $NUM_STEPS -c $CKPT_DIR/$task/$i/$ckpt --model $MODEL" \
                Enter
        done
    done

elif [ "$MODE" = "rand" ]; then

    for task in `ls $CKPT_DIR`; do
        task_=${task##*.} # remove "wob.mini."
        tmux new-window -t play -n $task_ bash
        sleep 1
        tmux send-keys -t play:$task_ "python play.py $task -s $NUM_STEPS --model $MODEL" Enter
    done

else

    echo "$MODE is invalid mode."
    cleanup
    exit 1

fi

# waiting
sleep 5
while true; do
    if [ "`ps aux | grep python | grep "play.py" | wc -l`" -lt 1 ]; then
        break
    fi
    sleep 60
done

cleanup

mkdir -p $LOG_DIR
mv /tmp/universe-* $LOG_DIR/

# if run on GCE, then shutodown right now
[ "`hostname`" = "wob" ] && sudo shutdown -h now
