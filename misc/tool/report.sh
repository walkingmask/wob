#!/bin/bash
set -eu


# Report to slack


# Parameters for slack
WEBHOOKURL=""
CHANNEL=""
BOTNAME=""
FACEICON=""

MESSAGE="$@"

# slackに投げる
curl -s -S -X POST --data-urlencode "payload={\"channel\": \"${CHANNEL}\", \"username\": \"${BOTNAME}\", \"icon_emoji\": \"${FACEICON}\", \"text\": \"${MESSAGE}\" }" ${WEBHOOKURL} >/dev/null
