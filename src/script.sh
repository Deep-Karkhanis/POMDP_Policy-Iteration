#!/bin/bash

if [ $2 -eq 1 ]
then
	cd /home/taufeeque/Desktop/SEM6/POMDP/AI-Toolbox/build
	make -j

	mkdir -p /home/taufeeque/Desktop/SEM6/POMDP/POMDP_Policy-Iteration/src/$1
	cd /home/taufeeque/Desktop/SEM6/POMDP/POMDP_Policy-Iteration/src/$1
	rm -f ./AIToolbox.so
	ln -s /home/taufeeque/Desktop/SEM6/POMDP/AI-Toolbox/build/AIToolbox.so ./AIToolbox.so

	g++ -std=c++17 $1.cpp /usr/local/include/AIToolbox_dep/libAIToolboxPOMDP.a /usr/local/include/AIToolbox_dep/libAIToolboxMDP.a  -o $1
fi

cd /home/taufeeque/Desktop/SEM6/POMDP/POMDP_Policy-Iteration/src/$1

if [ $3 -eq 1 ]
then
	g++ -std=c++17 $1.cpp /usr/local/lib/liblpsolve55.so /usr/local/include/AIToolbox_dep/libAIToolboxPOMDP.a /usr/local/include/AIToolbox_dep/libAIToolboxFMDP.a /usr/local/include/AIToolbox_dep/libAIToolboxMDP.a  -o $1
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
