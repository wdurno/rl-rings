xvfb-run -p XSMP python /app/src/python/ai/ai_viewer.py &
fluxbox &
x11vnc -display :99 -bg -nopw -xkb -viewonly 
echo "viewer started, sleeping..."
python3 -u /app/src/python/debug/debug.py 
