#!/bin/bash

PATH_TO_BIN="../../../../target/release/examples/privmail_sc"

# Run different tests
TEST_SETS=("mail-shares" "mail-shares-19-" "mail-shares-29-" "mail-shares-1000-")
for TEST_SET in ${TEST_SETS[@]}; do

# Run party 0 in the background and ignore any output
$PATH_TO_BIN \
    --my-id 0 \
    --server 127.0.0.1:23000 \
    --query-file-path "query-share0/220120-041612_te4M915F.yaml" \
    --mail-dir-path "${TEST_SET}0/" \
     &

sleep 1

# Run party 1 and get the statistics on terminal
$PATH_TO_BIN \
    --my-id 1 \
    --server 127.0.0.1:23000 \
    --query-file-path "query-share1/220120-041612_T2VOmDPg.yaml" \
    --mail-dir-path "${TEST_SET}1/" \

sleep 1
done
