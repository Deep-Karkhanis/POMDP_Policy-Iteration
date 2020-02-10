#!/bin/bash

if [ $2 -eq 1 ]
then	
	cd /home/deep-k98/PolicyIter/AI-Toolbox-master/build
	make
	
	mkdir -p /home/deep-k98/PolicyIter/src/$1
	cd /home/deep-k98/PolicyIter/src/$1
	rm -f ./AIToolbox.so
	ln -s /home/deep-k98/PolicyIter/AI-Toolbox-master/build/AIToolbox.so ./AIToolbox.so 

	g++ -std=c++17 $1.cpp /usr/local/include/AIToolbox_dep/* -o $1
fi

cd /home/deep-k98/PolicyIter/src/$1

if [ $3 -eq 1 ]
then	
	g++ -std=c++17 $1.cpp /usr/local/include/AIToolbox_dep/* -o $1
fi

if [ $4 -eq 1 ]
then	
	for i in {1..10}
	do	
		./$1 < inp_$1
	done
else
	./$1 < inp_$1	
fi




