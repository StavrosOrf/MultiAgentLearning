#!/bin/sh
rm */x*

for dir in *;
do
	cd $dir && split --lines 501 *.csv && cd ../
done

find */* -size -10c -delete
#awk -i inplace -F "," '{print $7}' */x* #get average
awk -i inplace -F "," '{print $3}' */x* #get max
sed -i -e "1d" */x*

rm */*.yaml #*/*.csv
