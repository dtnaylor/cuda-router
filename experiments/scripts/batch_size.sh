#! /bin/bash

CLICK_DIR=../../click
ROUTER_DIR=../../cuda
RUNTIME=30

$CLICK_DIR/userlevel/click $CLICK_DIR/conf/cuda_sample.click &

sleep 1

echo -e "\n\nGPU BATCH 32"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=32
sleep 2
echo -e "\n\nGPU BATCH 64"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=64
sleep 2
echo -e "\n\nGPU BATCH 128"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=128
sleep 2
echo -e "\n\nGPU BATCH 256"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=256
sleep 2
echo -e "\n\nGPU BATCH 512"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=512
sleep 2
echo -e "\n\nGPU BATCH 1024"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=1024
sleep 2
echo -e "\n\nGPU BATCH 2048"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=2048
sleep 2
echo -e "\n\nGPU BATCH 4096"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=4096
sleep 2
echo -e "\n\nGPU BATCH 8192"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=8192
sleep 2
echo -e "\n\nGPU BATCH 16384"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=16384
sleep 2
echo -e "\n\nGPU BATCH 65536"
$ROUTER_DIR/router -runtime=$RUNTIME -batch=65536
sleep 2


echo -e "\n\nCPU BATCH 32"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=32
sleep 2
echo -e "\n\nCPU BATCH 64"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=64
sleep 2
echo -e "\n\nCPU BATCH 128"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=128
sleep 2
echo -e "\n\nCPU BATCH 256"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=256
sleep 2
echo -e "\n\nCPU BATCH 512"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=512
sleep 2
echo -e "\n\nCPU BATCH 1024"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=1024
sleep 2
echo -e "\n\nCPU BATCH 2048"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=2048
sleep 2
echo -e "\n\nCPU BATCH 4096"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=4096
sleep 2
echo -e "\n\nCPU BATCH 8192"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=8192
sleep 2
echo -e "\n\nCPU BATCH 16384"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=16384
sleep 2
echo -e "\n\nCPU BATCH 65536"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -batch=65536
sleep 2

killall click
