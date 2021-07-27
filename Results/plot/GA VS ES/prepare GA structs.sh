#!/bin/sh
#find */* -size -10c -delete
awk -i inplace -F "," '{print $2}' GA*/* #get average
