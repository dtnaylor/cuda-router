#! /bin/bash

# Plot speedup vs nprocs
gnuplot <<EOC 
set terminal postscript 
set key out vert
set key top right
set size 0.52,0.5
set xtics rotate by -45
set output "$1.ps"
set xlabel "$1 ($2)"
set ylabel "Bandwidth (Gbps)"
set title "Bandwidth vs. $1"
plot "$3" using 1:2 title "GPU" with linespoints, "$3" using 1:6 title "CPU" with linespoints

set xlabel "$1 ($2)"
set ylabel "Latency (ms)"
set size 0.69,0.5
set title "Latency vs. $1"
plot "$3" using 1:3 title "GPU Max Latency" with linespoints, "$3" using 1:7 title "CPU Max Latency" with linespoints, "$3" using 1:4 title "GPU Min Latency" with linespoints, "$3" using 1:8 title "CPU Min Latency" with linespoints

set size 0.55,0.5
set xlabel "$1 ($2)"
set ylabel "Processing Time (ms)"
set title "Processing Time vs. $1"
plot "$3" using (\$1/1000):5 title "GPU" with linespoints, "$3" using 1:9 title "CPU" with linespoints

set size 0.7,0.5
set style data histograms
set xtics rotate by 270
set style histogram rowstacked
set yrange [0:25]
set boxwidth 1 relative
set style fill pattern 1.0 border -1
set ylabel "Time (ms)"
set title "GPU Time Breakdown vs. $1"
plot "$3" using 10 t "Gather Packets"\
	, '' using 5 t "Process"\
	, '' using 11:xticlabels(1) t "Send Packets"\
	, '' using 12 t "Copy to Device"\
	, '' using 13 t "Copy from Device"

set ylabel "Time (ms)"
set title "CPU Time Breakdown vs. $1"
plot "$3" using 14 t "Gather Packets", '' using 9 t "Process", '' using 15:xticlabels(1) t "Send Packets"

set ylabel "% Time"
set yrange [0:100]
set title "GPU % Time Breakdown vs. $1"
plot "$3" using (100*\$10/(\$10+\$12+\$5+\$13+\$11)) t "Gather Packets"\
	, '' using (100*\$5/(\$10+\$12+\$5+\$13+\$11)) t "Process"\
	, '' using (100*\$11/(\$10+\$12+\$5+\$13+\$11)):xticlabels(1) t "Send Packets"\
	, '' using (100*\$12/(\$10+\$12+\$5+\$13+\$11)) t "Copy to Device"\
	, '' using (100*\$13/(\$10+\$12+\$5+\$13+\$11)) t "Copy from Device"

set ylabel "% Time"
set title "CPU % Time Breakdown vs. $1"
plot "$3" using (100*\$14/(\$14+\$9+\$15)) t "Gather Packets"\
	, '' using (100*\$9/(\$14+\$9+\$15)) t "Process"\
	, '' using (100*\$15/(\$14+\$9+\$15)):xticlabels(1) t "Send Packets"

EOC
