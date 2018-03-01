#!/usr/bin/env bash

# util
function rmuniv () {
    /bin/rm /tmp/universe-*
    /bin/rm -rf /tmp/demos/
}
function rmcont () {
    docker stop $(docker ps -aq)
    docker rm $(docker ps -aq)
}
function pykill () {
    pkill python
}
function clean () {
    rmuniv
    rmcont
    pykill
}
function check () {
    ls /tmp | grep demos
    ls /tmp | grep universe
    docker ps -a
    ps aux | grep python | grep -v grep
}

# main
function record () {
    # default variable
    SLEEPING=$((60*11))
    TASK=""

    # args
    while getopts s:t: OPT
    do
      case $OPT in
        "s" ) SLEEPING=$((OPTARG*1)) ;;
        "t" ) TASK="$OPTARG" ;;
      esac
    done

    if [ "$TASK" = "" ]; then
        echo "Usage: record -t TASK [-s SLEEPTIME]" 1>&2
        return 1
    fi

    # make sure docker is running
    if [ `ps aux | grep docker | wc -l | tr -d ' '` -gt 1 ]; then
        :
    else
        echo "docker is not running." 1>&2
        return 1
    fi

    # make sure universe is installed
    if [ `pip freeze | grep universe | wc -l | tr -d ' '` -gt 0 ]; then
        :
    else
        echo "universe is not installed." 1>&2
        return 1
    fi

    # make sure demos directories are exist
    mkdir -p /tmp/demos
    mkdir -p ~/Desktop/demos

    echo "$TASK" >/tmp/demos/env_id.txt

    # start vnc_recorder 
    python ./py/vnc_recorder.py -d /tmp/demos >/dev/null 2>&1 &

    # start wob docker container via python
    python -c "import time;import gym;import universe;env=gym.make('$TASK');env.configure(remotes=1, fps=12);time.sleep(100000)" >/dev/null 2>&1 &

    # wait for starting
    for i in `seq 20 1`; do printf "\rwaiting... %02d" $i && sleep 1; done

    # open vnc
    open vnc://localhost:5899

    # timer
    sleep $SLEEPING
    /usr/bin/osascript <<EOF >>/dev/null
tell application "System Events"
    activate
    display dialog "Good job!" buttons {"OK"} with title "('e')"
end tell
EOF

    # terminate
    kill %3 %2

    # move logs
    mv /tmp/demos ~/Desktop/demos/$TASK
    mv /tmp/universe-* ~/Desktop/demos/$TASK/

    clean
    check
}