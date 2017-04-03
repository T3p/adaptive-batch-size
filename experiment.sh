#!/bin/sh
rm ./reinforce_out.txt
echo "REINFORCE N = 10000" > reinforce_out.txt
for i in `seq 1 100`;
do
    python lqg_agent.py 10000 reinforce_out.txt;
done   
