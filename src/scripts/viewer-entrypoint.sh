echo Controllable environment launched in tmux session
echo Attach with: tmux attach -t tmux-session 
echo Detach with: Ctrl-b, d 
tmux new-session -d -s tmux-session xvfb-run -p XSMP python /app/src/python/ai/ai_viewer.py --interactive-mode
sleep 10
fluxbox &
sleep 10
x11vnc -display :99 -bg -nopw -xkb -viewonly 
echo "viewer started, sleeping..."
python3 -u /app/src/python/debug.py --silent 
