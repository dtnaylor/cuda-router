#! /bin/bash
#set xlabel "x" 0.000000,0.000000  ""
#set ylabel "y=exp(-x)" 0.000000,0.000000  ""
#set title "Pade approximation" 0.000000,0.000000  ""
#set xrange [ 0 : 2 ] noreverse nowriteback
#set yrange [ 0 : 1 ] noreverse nowriteback
#set mxtics 5.000000
#set mytics 5.000000
#set xtics border mirror norotate 1
#set ytics border mirror norotate 0.5
#    EOF
#set yrange [-150:150]
#set size ratio 0.25


# Plot speedup vs nprocs
gnuplot <<EOC 
set terminal postscript
set output "$1.ps"
set xlabel "$1 ($2)"
set ylabel "Bandwidth (Gbps)"
set title "Bandwidth vs. $1"
plot "$3" using 1:2 title "GPU" with linespoints, "$3" using 1:6 title "CPU" with linespoints

set xlabel "$1 ($2)"
set ylabel "Latency (ms)"
set title "Latency vs. $1"
plot "$3" using 1:3 title "GPU Max Latency" with linespoints, "$3" using 1:7 title "CPU Max Latency" with linespoints, "$3" using 1:4 title "GPU Min Latency" with linespoints, "$3" using 1:8 title "CPU Min Latency" with linespoints

set xlabel "$1 ($2)"
set ylabel "Processing Time (us)"
set title "Processing Time vs. $1"
plot "$3" using 1:5 title "GPU" with linespoints, "$3" using 1:9 title "CPU" with linespoints
EOC
