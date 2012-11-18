#! /bin/bash

CLICK_DIR=../../click
ROUTER_DIR=../../cuda
RUNTIME=30

$CLICK_DIR/userlevel/click $CLICK_DIR/conf/cuda_sample.click &

sleep 1

echo -e "\n\nGPU NUMRULES 25"
$ROUTER_DIR/router -runtime=$RUNTIME -numrules=25
echo -e "\n\nGPU NUMRULES 50"
$ROUTER_DIR/router -runtime=$RUNTIME -numrules=50
echo -e "\n\nGPU NUMRULES 100"
$ROUTER_DIR/router -runtime=$RUNTIME -numrules=100
echo -e "\n\nGPU NUMRULES 200"
$ROUTER_DIR/router -runtime=$RUNTIME -numrules=200
echo -e "\n\nGPU NUMRULES 400"
$ROUTER_DIR/router -runtime=$RUNTIME -numrules=400
echo -e "\n\nGPU NUMRULES 800"
$ROUTER_DIR/router -runtime=$RUNTIME -numrules=800

echo -e "\n\nCPU NUMRULES 25"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -numrules=25
echo -e "\n\nCPU NUMRULES 50"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -numrules=50
echo -e "\n\nCPU NUMRULES 100"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -numrules=100
echo -e "\n\nCPU NUMRULES 200"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -numrules=200
echo -e "\n\nCPU NUMRULES 400"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -numrules=400
echo -e "\n\nCPU NUMRULES 800"
$ROUTER_DIR/router -sequential -runtime=$RUNTIME -numrules=800


killall click
