#!/usr/bin/python

import sys
from numpy import *

def main(argv):
    ind_var = argv[0]
    infile = argv[1]
    outfile = infile[0:-7] + '.data'

    with open(infile, 'r') as f:
        lines = f.readlines()
    f.closed
    
    # We're building nested hash tables:
    # stats
    #   GPU
    #     128
    #       bw = ...
    #       max_lat = ...
    #       ...
    #     256
    #       bw = ...
    #       ...
    #     ...
    #   CPU
    #     ...

    stats = { 'GPU': {}, 'CPU':{} }
    stat = None
    cur_mode = ''
    cur_ind_var_val = -1
    ind_var_vals = []

    for line in lines:
        if ind_var in line:
            arr = line.split(' ')
            cur_mode = arr[0]
            cur_ind_var_val = int(arr[2])

            stats[cur_mode][cur_ind_var_val] = {}

            if cur_mode == 'GPU':
                ind_var_vals.append(cur_ind_var_val)


        if 'Bandwidth' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['bw'] = float(arr[9])
        elif 'processing time' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['proc_time'] = float(arr[3])/1000
        elif 'get time' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['get_time'] = float(arr[4])/1000
        elif 'send time' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['send_time'] = float(arr[4])/1000
        elif 'copy to device time' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['copy_to_device_time'] = float(arr[6])/1000
        elif 'copy from device time' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['copy_from_device_time'] = float(arr[6])/1000
        elif 'latency' in line:
            arr = line.split(' ')
            stats[cur_mode][cur_ind_var_val]['max_latency'] = float(arr[3])/1000
            stats[cur_mode][cur_ind_var_val]['min_latency'] = float(arr[6])/1000

    with open(outfile, 'w') as f:
        headers = 'val\tgpu_bw\tgpu_max_lat\tgpu_min_lat\tgpu_proc_time\tcpu_bw\tcpu_max_lat\tcpu_min_lat\tcpu_proc_time\tgpu_get_time\tgpu_send_time\tgpu_copy_to_time\tgpu_copy_from_time\tcpu_get_time\tcpu_send_time\n'
        f.write(headers)
        for val in ind_var_vals:
            gpu = stats['GPU'][val]
            cpu = stats['CPU'][val]
            line = '%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (val, gpu['bw'], gpu['max_latency'], gpu['min_latency'], gpu['proc_time'], cpu['bw'], cpu['max_latency'], cpu['min_latency'], cpu['proc_time'], gpu['get_time'], gpu['send_time'], gpu['copy_to_device_time'], gpu['copy_from_device_time'], cpu['get_time'], cpu['send_time'])
            f.write(line)
    f.closed




if __name__ == "__main__":
   main(sys.argv[1:])
