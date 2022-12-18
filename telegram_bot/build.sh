#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`


cd "$(dirname "$0")"
root_dir=$PWD 
cd $root_dir

docker build . \
    -f $root_dir/Dockerfile \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t tg_bot:latest