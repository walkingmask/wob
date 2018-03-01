#!/usr/bin/env bash

function d2d() {
    DEMOS="$HOME/Desktop/demos"
    DATASETS="$HOME/Desktop/datasets"

    while getopts i:o: OPT
    do
      case $OPT in
        "i" ) DEMOS="$OPTARG" ;;
        "o" ) DATASETS="$OPTARG" ;;
      esac
    done

    if [ ! -d "$DEMOS" ]; then
        echo "Usage: d2d [-i PATH_TO_DEMOS] [-o PATH_TO_OUT]" 1>&2
        return 1
    fi

    mkdir -p $DATASETS

    for TASK in `ls $DEMOS`; do
        for obj in `ls $DEMOS/$TASK`; do
            if [ -d $DEMOS/$TASK/$obj ]; then
                python ./py/demo2dataset.py $DEMOS/$TASK/$obj -o $DATASETS/$TASK.bz2
            fi
        done
    done
}
