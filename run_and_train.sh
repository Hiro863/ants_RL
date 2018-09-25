#!/usr/bin/env bash#!/usr/bin/env sh
#echo Observing...
#python3 TrainSession.py observing

iteration=5000
gameplay=10
for i in $(seq 1 $iteration); do

    for j in $(seq 1 $gameplay); do
    echo Iteration: $i, Gameplay: $j
    ./tools/playgame.py  --nolaunch --player_seed 42 --end_wait=0.25 --log_dir game_logs --turns 1000 --map_file tools/maps/example/tutorial1.map "$@" "python3 MyBot.py" "python tools/sample_bots/python/LeftyBot.py" #"python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/GreedyBot.py"
    #echo "\n\nContent of debug log:"
    #cat ./debug.txt
    done

    python3 TrainSession.py
    python3 save_results.py
    python3 visualise.py
    rm training.p
    if (( $i % 10 == 0 ));
    then
    cp -fR weights target_weights
    fi
done