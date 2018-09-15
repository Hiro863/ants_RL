#!/usr/bin/env sh
./tools/playgame.py --player_seed 42 --end_wait=0.25 --verbose --log_dir game_logs --turns 100 --map_file tools/maps/example/tutorial1.map "$@" "python3 MyBot.py" "python3 tools/sample_bots/python/LeftyBot.py" #"python3 tools/sample_bots/python/HunterBot.py" "python3 tools/sample_bots/python/GreedyBot.py"
echo "\n\nContent of debug log:"
cat ./debug.txt
