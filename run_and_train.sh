#!/usr/bin/env bash#!/usr/bin/env sh
END=100
for i in $(seq 1 $END); do
    echo $i
    ./tools/playgame.py  --nolaunch --player_seed 42 --end_wait=0.25 --log_dir game_logs --turns 1000 --map_file tools/maps/example/tutorial1.map "$@" "python3 MyBot.py" "python tools/sample_bots/python/LeftyBot.py" #"python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/GreedyBot.py"
    echo "\n\nContent of debug log:"
    cat ./debug.txt
    python3 TrainSession.py
done