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
set terminal postscript color
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

set style data histograms
set style histogram rowstacked
set boxwidth 1 relative
set style fill pattern 1.0 border -1
set key top left
set ylabel "Time (us)"
set title "GPU Time Breakdown vs. $1"
plot "$3" using 10 t "Gather Packets", '' using 12 t "Copy to Device", '' using 5 t "Process", '' using 13 t "Copy from Device", '' using 11:xticlabels(1) t "Send Packets"

set style data histograms
set style histogram rowstacked
set boxwidth 1 relative
set style fill pattern 1.0 border -1
set key top left
set ylabel "% Time"
set title "GPU % Time Breakdown vs. $1"
plot "$3" using (100*\$10/(\$10+\$12+\$5+\$13+\$11)) t "Gather Packets"\
	, '' using (100*\$12/(\$10+\$12+\$5+\$13+\$11)) t "Copy to Device"\
	, '' using (100*\$5/(\$10+\$12+\$5+\$13+\$11)) t "Process"\
	, '' using (100*\$13/(\$10+\$12+\$5+\$13+\$11)) t "Copy from Device"\
	, '' using (100*\$11/(\$10+\$12+\$5+\$13+\$11)):xticlabels(1) t "Send Packets"

set style data histograms
set style histogram rowstacked
set boxwidth 1 relative
set style fill pattern 1.0 border -1
set key top left
set ylabel "Time (us)"
set title "CPU Time Breakdown vs. $1"
plot "$3" using 14 t "Gather Packets", '' using 9 t "Process", '' using 15:xticlabels(1) t "Send Packets"

set style data histograms
set style histogram rowstacked
set boxwidth 1 relative
set style fill pattern 1.0 border -1
set key top left
set ylabel "% Time"
set title "CPU % Time Breakdown vs. $1"
plot "$3" using (100*\$14/(\$14+\$9+\$15)) t "Gather Packets"\
	, '' using (100*\$9/(\$14+\$9+\$15)) t "Process"\
	, '' using (100*\$15/(\$14+\$9+\$15)):xticlabels(1) t "Send Packets"

EOC
