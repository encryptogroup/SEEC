#!/bin/bash

id=$1

dir=$(pwd)
cd build-release
make aescbc -j 30
cd $dir

for in_blocks in {1,2,8,16,32,64,128,256,512,1024};do
  if [[ $id == 0 ]];then
    heaptrack="$HOME/heaptrack/build/bin/heaptrack -o heaptrack_MOTION_$in_blocks"
  fi
   $heaptrack ./build-release/bin/aescbc --my-id $id --parties 0,130.83.125.166,7745 1,130.83.125.167,7744 --data-bytes $(($in_blocks * 16)) > MOTION_block_$in_blocks.log
done