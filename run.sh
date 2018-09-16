#!/usr/bin/env sh
./tools/playgame.py  --nolaunch --player_seed 42 --end_wait=0.25 --log_dir game_logs --turns 100 --map_file tools/maps/example/tutorial1.map "$@" "python MyBot.py" "python tools/sample_bots/python/LeftyBot.py" #"python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/GreedyBot.py"
echo "\n\nContent of debug log:"
cat ./debug.txt
