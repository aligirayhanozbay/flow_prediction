#!/usr/bin/bash
function set_boundary {
	fname=$1
	xmin=$2
	xmax=$3
	ymin=$4
	ymax=$5
	sed -e "s/xmin =.*/xmin = ${xmin};/" -e "s/xmax =.*/xmax = ${xmax};/" -e "s/ymin =.*/ymin = ${ymin};/" -e "s/ymax =.*/ymax = ${ymax};/" -i $1
}

args=("$@")
counter=0
for arg in "$@"
do
	echo ${counter}
	if [[ $counter -lt 4 ]]
	then
		echo 'skipped'
	else
		set_boundary ${arg} ${args[0]} ${args[1]} ${args[2]} ${args[3]}
	fi
	let counter++
done
	
