#!/bin/bash

id=$1

curr_dir=$(pwd)
cd ../../
cargo build --release --example aes_cbc
cd $curr_dir

export RUST_LOG=info

for in_blocks in {1,2,8,16,32,64,128,256,512,1024};do
  if [[ $id == 0 ]];then
    heaptrack="$HOME/heaptrack/build/bin/heaptrack -o heaptrack_SEEC_$in_blocks"
    $heaptrack ../../target/release/examples/aes_cbc --id $id --server 130.83.125.166:7744
  else
    ../../target/release/examples/aes_cbc --id $id --server 130.83.125.166:7744 --input-bytes $(($in_blocks*16)) > SEEC_blocks_$in_blocks.log
    sleep 1
  fi
done

for in_blocks in {1,2,8,16,32,64,128,256,512,1024};do
  if [[ $id == 0 ]];then
    heaptrack="$HOME/heaptrack/build/bin/heaptrack -o heaptrack_SEEC_SC_$in_blocks"
    $heaptrack ../../target/release/examples/aes_cbc --id $id --server 130.83.125.166:7744 --use-sc > SEEC_SC_blocks_$in_blocks.log
  else
    ../../target/release/examples/aes_cbc --id $id --server 130.83.125.166:7744 --use-sc --input-bytes $(($in_blocks*16))
    sleep 1
  fi
done