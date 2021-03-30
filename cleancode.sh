#!/bin/sh
#run it to clean up the code
sed -i 's/[ \t]*$//' "$1" *.cpp */*/*.cpp */*/*.h
sed -i 's/ ;/;/g' *.cpp */*/*.cpp */*/*.h
