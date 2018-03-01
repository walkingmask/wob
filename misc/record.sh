#!/usr/bin/env bash
set -eu


#
# Recording MiniWeb demonstrations
#
# # Required
#   - AppleScript
#   - QuickTime Player's Screen Recorder
#   - cliclick
#   - Chrome, javascript
#   - peco
#


# Select a task
if [ ! -f ./todo.txt ]; then
    echo "Error! There is no task list." >&2
    exit 1
fi
if [ ! -f ./done.txt ]; then
    touch ./done.txt
fi
task="`diff --old-line-format='%L' --new-line-format='%L' --unchanged-line-format='' ./todo.txt ./done.txt | peco`"
echo $task

# Increase the volume
/usr/bin/osascript -e "set volume 50/100*3"

# Open QuickTime Player's Screen Recorder
/usr/bin/osascript <<EOF
tell application "QuickTime Player"
   activate
   close every window saving no
   set newScreenRecording to new screen recording
end tell
EOF

# Wait until start button is pressed
echo "Click recording button." && sleep 3

# Drag to specify the recording range
startX=0
startY=0
endX=160
endY=210
buttonX=$((startX+(endX-startX)/2))
buttonY=$((startY+(endY-startY)/2))
/usr/local/bin/cliclick dd:$startX,$startY du:$endX,$endY c:$buttonX,$buttonY

# JavaScript for track mouse and keyboard operations
jscode="
document.addEventListener('keydown',   function(e) { console.log('action: KeyPress,' + e.keyCode) })
document.addEventListener('mousemove', function(e) { console.log('action: MouseMove,' + e.pageX + ',' + e.pageY) })
document.addEventListener('mousedown', function(e) { console.log('action: MouseDown,' + e.pageX + ',' + e.pageY)})
document.addEventListener('mouseup',   function(e) { console.log('action: MouseUp,' + e.pageX + ',' + e.pageY)})
"

# Open Chrome in full screen
/usr/bin/osascript <<EOF
tell application "Google Chrome"
    activate
    tell application "System Events"
        delay 1
        keystroke "f" using {control down, command down}
    end
end tell
EOF

# First execute button-click to get timestamp
/usr/bin/osascript <<EOF >>/dev/null
tell application "Google Chrome"
    activate
    open location "http://alpha.openai.com/miniwob/preview/miniwob/click-button.html"
    tell application "System Events"
        delay 1
        keystroke "j" using {option down, command down}
    end
    execute front window's active tab javascript "$jscode"
end tell
EOF

sleep 10

/usr/bin/osascript <<EOF >>/dev/null
tell application "Google Chrome"
    activate
    open location "http://alpha.openai.com/miniwob/preview/miniwob/${task}.html"
    tell application "System Events"
        delay 1
        keystroke "j" using {option down, command down}
    end
    execute front window's active tab javascript "$jscode"
end tell
EOF

# Notify after 10 minutes
sleep 610 && /usr/bin/osascript <<EOF >>/dev/null
tell application "System Events"
    activate
    display dialog "Good job!" buttons {"OK"} with title "('e')"
end tell
EOF

echo "Here you need to save videos and console logs."
mkdir -p ./demos/$task

# Done task
echo $task >>./done.txt
sed -i'' "/$task/d" ./todo.txt

exit 0
