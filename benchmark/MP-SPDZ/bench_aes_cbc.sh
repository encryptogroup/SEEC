#!/bin/bash

id=$1

./Scripts/setup-online.sh

for in_blocks in {1,2,8,16,32,64,128,256,512,1024};do
  if [[ $id == 0 ]];then
    heaptrack="$HOME/heaptrack/build/bin/heaptrack -o heaptrack_MP_SPDZ_$in_blocks"
  fi
  ./compile.py -B 64 aescbc_circuit $in_blocks
   $heaptrack ./semi-bin-party.x $id aescbc_circuit-$in_blocks -F -pn 12353 -h 130.83.125.166 2> MP_SPDZ_block_$in_blocks.log
done