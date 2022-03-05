#!/bin/sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#PREPARE ES/
rm ES/*/x*

for dir in ES/*;
do
	cd $dir && split --lines 501 *.csv && cd $SCRIPT_DIR
done

find ES/*/* -size -10c -delete
#awk -i inplace -F "," '{print $7}' */x* #get average
awk -i inplace -F "," '{print $3}' ES/*/x* #get max
sed -i -e "1d" ES/*/x*

#rm ES/*/*.yaml #ES/*/*.csv



#prepare CCEA/ [WIP]
#rm -r CCEA/*/Replay CCEA/*/neural_nets.csv
#awk -i inplace -F "," '{print $2}' CCEA/*/*.* #get max g
##rename -n 's/CCEA/*/*\.*//' *
#i=0
#for file in $(find CCEA/*/* -type f); do
        #mv $file $(dirname -- $file)/e$i
        #i=$(($i+1))
#done
