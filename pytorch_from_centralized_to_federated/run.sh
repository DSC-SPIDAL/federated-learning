#!/bin/bash

echo "Starting server"
python server2.py &
sleep 3

for i in `seq 0 3`; do
    echo "Starting client $i"
    python client.py &
done


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

wait
