#! /bin/bash

CLICK_DIR=../../click
ROUTER_DIR=../../cuda
RUNTIME=30

$CLICK_DIR/userlevel/click $CLICK_DIR/conf/cuda_sample.click &

sleep 1

echo -e "\n\nGPU BATCH 128"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=128
echo -e "\n\nGPU BATCH 256"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=256
echo -e "\n\nGPU BATCH 512"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=512
echo -e "\n\nGPU BATCH 1024"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=1024
echo -e "\n\nGPU BATCH 2048"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=2048
echo -e "\n\nGPU BATCH 4096"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=4096


echo -e "\n\nCPU BATCH 128"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=128
echo -e "\n\nCPU BATCH 256"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=256
echo -e "\n\nCPU BATCH 512"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=512
echo -e "\n\nCPU BATCH 1024"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=1024
echo -e "\n\nCPU BATCH 2048"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=2048
echo -e "\n\nCPU BATCH 4096"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=4096


killall click
