#!/bin/bash
set -eu


# Scheduled report every 10 minutes


# Informations
PY_PROCESSES="`ps aux | grep python | grep -v grep | grep -v google | wc -l`"
DISK_USAGE="`df -h / --output=pcent | tail -1 | tr -d ' '`"
UP_TIME="`uptime -p`"

# Message for slack
MESSAGE="[`TZ=JST+15 date '+%Y-%m-%d %H:%M:%S'`] Running with $PY_PROCESSES python processes, $DISK_USAGE disk used, $UP_TIME."

/usr/local/bin/report $MESSAGE
