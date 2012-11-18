#! /bin/bash

CLICK_DIR=../../click
ROUTER_DIR=../../cuda
RUNTIME=30

$CLICK_DIR/userlevel/click $CLICK_DIR/conf/cuda_sample.click &

sleep 1

echo -e "\n\nGPU WAIT 25"
$ROUTER_DIR/router -runtime=$RUNTIME -wait=25
echo -e "\n\nGPU WAIT 50"
$ROUTER_DIR/router -runtime=$RUNTIME -wait=50
echo -e "\n\nGPU WAIT 100"
$ROUTER_DIR/router -runtime=$RUNTIME -wait=100
echo -e "\n\nGPU WAIT 200"
$ROUTER_DIR/router -runtime=$RUNTIME -wait=200
echo -e "\n\nGPU WAIT 400"
$ROUTER_DIR/router -runtime=$RUNTIME -wait=400
echo -e "\n\nGPU WAIT 800"
$ROUTER_DIR/router -runtime=$RUNTIME -wait=800

echo -e "\n\nCPU WAIT 25"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -wait=25
echo -e "\n\nCPU WAIT 50"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -wait=50
echo -e "\n\nCPU WAIT 100"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -wait=100
echo -e "\n\nCPU WAIT 200"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -wait=200
echo -e "\n\nCPU WAIT 400"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -wait=400
echo -e "\n\nCPU WAIT 800"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -wait=800


killall click
