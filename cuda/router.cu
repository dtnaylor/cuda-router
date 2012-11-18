#include <router.h>

#define MEASURE_LATENCY
#define MEASURE_BANDWIDTH
#define MEASURE_PROCESSING_MICROBENCHMARK

#define DEFAULT_BLOCK_SIZE 32


int runtime = 0;
struct timeval start_time, cur_time;
bool _do_run = true;
inline bool do_run() {
	if (_do_run && runtime > 0) {
		gettimeofday(&cur_time, NULL);

		if (cur_time.tv_sec - start_time.tv_sec > runtime)
			return false;
		else
			return true;
	}
	return _do_run;
}



/**
 * Start a loop:
 *	1) Gather packets
 *  2) Copy packets to GPU and process
 *	3) Copy results back and print performance stats
 *
 * We do this with pipelining: while the GPU is processing one buffer of packets,
 * we're copying over the next batch so that it can begin processing them as soon
 * as it finishes processing the first batch.
 */
int run(int argc, char **argv, int block_size, int server_sockfd, udpc client)
{
	PRINT(V_INFO, "Running CPU/GPU code\n\n");

	gettimeofday(&start_time, NULL);

	unsigned int buf_size = sizeof(packet)*get_batch_size();
	unsigned int results_size = sizeof(int)*get_batch_size();

	// Allocate host memory for two batches of up to batch_size packets
	// We will alternate between filling and processing these two buffers
	// (at any given time one of the buffers will either be being filled
	// or being processed)
	packet* h_p1 = (packet *)malloc(buf_size);
	check_malloc(h_p1, "h_p1", __LINE__);
	packet* h_p2 = (packet *)malloc(buf_size);
	check_malloc(h_p2, "h_p2", __LINE__);

	// Allocate host memory for 2 arrays of results
	int *h_results1 = (int*)malloc(results_size);
	check_malloc(h_results1, "h_results1", __LINE__);
	int *h_results2 = (int*)malloc(results_size);
	check_malloc(h_results2, "h_results2", __LINE__);

	// Allocate device memory for up to batch_size packets
	// TODO: wait and allocate only the amount needed after we receive?
	packet *d_p1;
	check_error(cudaMalloc((void **) &d_p1, buf_size), "cudaMalloc d_p1", __LINE__);
	packet *d_p2;
	check_error(cudaMalloc((void **) &d_p2, buf_size), "cudaMalloc d_p2", __LINE__);
	// Allocate device memory for results
	int *d_results1;
	check_error(cudaMalloc((void **) &d_results1, results_size), "cudaMalloc d_results1", __LINE__);
	int *d_results2;
	check_error(cudaMalloc((void **) &d_results2, results_size), "cudaMalloc d_results2", __LINE__);


	// Setup execution parameters
	int threads_in_block = block_size;
	int blocks_in_grid = get_batch_size() / threads_in_block;  // FIXME: optimize if we don't have a full batch
	if (get_batch_size() % threads_in_block != 0) {
		blocks_in_grid++;  // need an extra block for the extra threads
	}


	// Run any processing-specific setup code needed
	// (e.g., this might copy the FIB to GPU for LPM)
	setup();
   

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	check_error(cudaEventCreate(&start), "Create start event", __LINE__);

	cudaEvent_t stop;
	check_error(cudaEventCreate(&stop), "Create stop event", __LINE__);

	double avg_micro = 0;
	int micro_nIters = 0;
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */

#ifdef MEASURE_BANDWIDTH
	long packets_processed = 0;
	struct timeval bw_start, bw_stop;
	gettimeofday(&bw_start, NULL);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_LATENCY
	double max_latency, min_latency, avg_max_latency = 0, avg_min_latency = 0;
	int lat_nIters = 0;
	struct timeval lat_start_oldest1, lat_start_oldest2, lat_start_newest1, lat_start_newest2, lat_stop;
	struct timeval *lat_start_oldest_current = &lat_start_oldest1;
	struct timeval *lat_start_oldest_next = &lat_start_oldest2;
	struct timeval *lat_start_newest_current = &lat_start_newest1;
	struct timeval *lat_start_newest_next = &lat_start_newest2;
#endif /* MEASURE_LATENCY */

	bool data_ready = false;
	bool results_ready = false;
	int num_packets_current;
	int num_packets_next;
	packet *h_p_current = h_p1;
	packet *h_p_next = h_p2;
	packet *d_p_current = d_p1;
	packet *d_p_next = d_p2;
	int *h_results_current = h_results1;
	int *h_results_previous = h_results2;
	int *d_results_current = d_results1;
	int *d_results_previous = d_results2;
	
	/* The main loop:
		1) Execute the CUDA kernel
		2) While it's executing, copy the results from the last execution back to the host
		3) While it's executing, copy the packets for the next execution to the GPU */
	
	while(do_run()) {
		
		/*************************************************************
		 *                1) EXECUTE THE CUDA KERNEL                 *
		 *************************************************************/
		if (data_ready) { // First execution of loop: data_ready = false

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
			// Record the start event
			check_error(cudaEventRecord(start, NULL), "Record start event", __LINE__);
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */

			// Execute the kernel
			PRINT(V_DEBUG, "vvvvv   Begin processing %d packets   vvvvv\n\n", num_packets_current);
			process_packets<<< blocks_in_grid, threads_in_block >>>(d_p_current, d_results_current, num_packets_current, block_size);

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
			// Record the stop event
			check_error(cudaEventRecord(stop, NULL), "Record stop event", __LINE__);
#endif /*MEASURE_PROCESSING_MICROBENCHMARK*/


#ifdef MEASURE_BANDWIDTH
			packets_processed += num_packets_current;
#endif /* MEASURE_BANDWIDTH */

		}


		
		/*************************************************************
		 *          2) COPY BACK RESULTS FROM LAST BATCH             *
		 *************************************************************/
		if (results_ready) { // First and second executions of loop: results_ready = false

			// TODO: double-check that stuff is really executing when I think it is.
			// I think that calling cudaEventRecord(stop) right before this will record
			// when the kernel stops executing, but won't block until this actually happens.
			// The cudaEventSynchronize call below does block until the kernel stops.
			// So, I think anything we do here will execute on the CPU while the GPU executes
			// the kernel call we made above.

			// Copy the last set of results back from the GPU
			check_error(cudaMemcpy(h_results_previous, d_results_previous, results_size, cudaMemcpyDeviceToHost), "cudaMemcpy (h_results, d_results)", __LINE__);

			// Send packets (right now, h_p_next still holds the *previous* batch)
			send_packets(client, h_p_next, num_packets_next);

#ifdef MEASURE_LATENCY
			gettimeofday(&lat_stop, NULL);
			max_latency = (lat_stop.tv_sec - lat_start_oldest_current->tv_sec) * 1000000.0 + (lat_stop.tv_usec - lat_start_oldest_current->tv_usec);
			min_latency = (lat_stop.tv_sec - lat_start_newest_current->tv_sec) * 1000000.0 + (lat_stop.tv_usec - lat_start_newest_current->tv_usec);
			PRINT(V_DEBUG_TIMING, "Latencies from last batch: Max: %f msec   Min: %f msec\n", max_latency, min_latency);

			lat_nIters++;
			avg_min_latency += min_latency; // keep a cummulative total and divide later
			avg_max_latency += max_latency;
#endif /* MEASURE_LATENCY */
		
			// Print results
			PRINT(V_DEBUG, "Results from last batch:\n");
			int i;
			for (i = 0; i < get_batch_size(); i++) {
				PRINT(V_DEBUG, "%d, ", h_results_previous[i]);
			}
			PRINT(V_DEBUG, "\n\n");

		}

		
		
		
		/*************************************************************
		 *                 3) COPY NEXT BATCH TO GPU                 *
		 *************************************************************/
		// Get next batch of packets and copy them to the GPU
		// FIXME: We're forcing the results from the current execution to wait
		// until we get the next batch of packets. Is this OK?
#ifdef MEASURE_LATENCY
		// Approx time we received the first packet of the batch
		// (not perfect if the first packet doesn't arrive immediately)
		gettimeofday(lat_start_oldest_next, NULL);
#endif /* MEASURE_LATENCY */
		num_packets_next = 0;
		while (num_packets_next == 0 && do_run()) {
			num_packets_next = get_packets(server_sockfd, h_p_next);
		}
#ifdef MEASURE_LATENCY
		gettimeofday(lat_start_newest_next, NULL);
#endif /* MEASURE_LATENCY */
		check_error(cudaMemcpy(d_p_next, h_p_next, buf_size, cudaMemcpyHostToDevice), "cudaMemcpy (d_p_next, h_p_next)", __LINE__);



		if (data_ready) {

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
			// Wait for the stop event to complete (which waits for the kernel to finish)
			check_error(cudaEventSynchronize(stop), "Failed to synchronize stop event", __LINE__);
			
			float msecTotal = 0.0f;
			check_error(cudaEventElapsedTime(&msecTotal, start, stop), "Getting time elapsed b/w events", __LINE__);

			micro_nIters++;
			avg_micro += msecTotal;

			// Compute and print the performance
			PRINT(V_DEBUG_TIMING,
				"Performance= Time= %.3f msec, WorkgroupSize= %u threads/block\n",
				msecTotal,
				threads_in_block);
#else
			// Wait for kernel execution to complete
			check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __LINE__);
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */


			PRINT(V_DEBUG, "^^^^^   Done processing batch   ^^^^^\n\n\n");

			results_ready = true;
		}
		data_ready = true;




		// Get ready for the next loop iteration
		SWAP(num_packets_current, num_packets_next, int);
		SWAP(h_p_current, h_p_next, packet*);
		SWAP(d_p_current, d_p_next, packet*);
		SWAP(h_results_current, h_results_previous, int*);
		SWAP(d_results_current, d_results_previous, int*);
#ifdef MEASURE_LATENCY
		SWAP(lat_start_oldest_current, lat_start_oldest_next, struct timeval*);
		SWAP(lat_start_newest_current, lat_start_newest_next, struct timeval*);
#endif /* MEASURE_LATENCY */

	}


#ifdef MEASURE_BANDWIDTH
	// Calculate how many packets we processed per second
	gettimeofday(&bw_stop, NULL);
	double total_time = (bw_stop.tv_sec - bw_start.tv_sec) + (bw_stop.tv_usec - bw_start.tv_usec) / 1000000.0;
	double pkts_per_sec = double(packets_processed) / total_time;	

	PRINT(V_INFO, "Bandwidth: %f packets per second  (64B pkts ==> %f Gbps)\n", pkts_per_sec, pkts_per_sec * 64.0 / 1000000.0);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
	avg_micro /= micro_nIters;
	PRINT(V_INFO, "Average processing time: %f msec\n", avg_micro);
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */

#ifdef MEASURE_LATENCY
	avg_min_latency /= lat_nIters;
	avg_max_latency /= lat_nIters;
	PRINT(V_INFO, "Average latency:  Max: %f msec,   Min: %f msec\n", avg_max_latency, avg_min_latency);
#endif /* MEASURE_LATENCY */




	// Clean up memory
	free(h_p1);
	free(h_p2);
	free(h_results1);
	free(h_results2);
	cudaFree(d_p1);
	cudaFree(d_p2);
	cudaFree(d_results1);
	cudaFree(d_results2);

	cudaDeviceReset();

	return EXIT_SUCCESS;
}

void test(int server_sockfd) 
{
	PRINT(V_INFO, "Batch Size: %d\n", get_batch_size());
	
	// Initialize a buffer for storing up to batch_size packets
	packet* p = (packet *)malloc(sizeof(packet)*get_batch_size());
	
	while(1) {
	  int num_packets = get_packets(server_sockfd, p);
	  PRINT(V_DEBUG, "i = %d\n", num_packets);

	  if (num_packets > 0) {
	  	struct ip *ip_hdr = (struct ip*)p->buf;
		struct udphdr *udp_hdr = (struct udphdr*)&(p->buf[sizeof(struct ip)]);
		PRINT(V_DEBUG, "Dest: %s (%u)\n", inet_ntoa(ip_hdr->ip_dst), ntohs(udp_hdr->uh_dport));
		PRINT(V_DEBUG, "Source: %s (%u)\n", inet_ntoa(ip_hdr->ip_src), ntohs(udp_hdr->uh_sport));
		PRINT(V_DEBUG, "Next proto: %u\n", ip_hdr->ip_p);
	  }
	}
}


int run_sequential(int argc, char **argv, int server_sockfd, udpc client)
{
	PRINT(V_INFO, "Running sequential router code on CPU only\n\n");

	gettimeofday(&start_time, NULL);
	
	unsigned int buf_size = sizeof(packet)*get_batch_size();
	unsigned int results_size = sizeof(int)*get_batch_size();

	// Allocate buffer for packets
	packet* p = (packet *)malloc(buf_size);
	check_malloc(p, "p", __LINE__);

	// Allocate array for results
	int *results = (int*)malloc(results_size);
	check_malloc(results, "results", __LINE__);


	// Run any processing-specific setup code needed
	// (e.g., this might prepare a data structure for LPM)
	setup_sequential();


#ifdef MEASURE_PROCESSING_MICROBENCHMARK
	struct timeval micro_start, micro_stop;
	double avg_micro = 0;
	int micro_nIters = 0;
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */

#ifdef MEASURE_BANDWIDTH
	long packets_processed = 0;
	struct timeval bw_start, bw_stop;
	gettimeofday(&bw_start, NULL);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_LATENCY
	struct timeval lat_start_oldest, lat_start_newest, lat_stop;
	double max_latency, min_latency, avg_max_latency = 0, avg_min_latency = 0;
	int lat_nIters = 0;
#endif /* MEASURE_LATENCY */


	/* The main loop:
		1) Get a batch of packets
		2) Process them */
	int num_packets;
	while(do_run()) {
		
		// Get next batch of packets

#ifdef MEASURE_LATENCY
		gettimeofday(&lat_start_oldest, NULL);
#endif /* MEASURE_LATENCY */
		num_packets = 0;
		while (num_packets == 0 && do_run()) {
			num_packets = get_packets(server_sockfd, p);
		}
#ifdef MEASURE_LATENCY
		gettimeofday(&lat_start_newest, NULL);
#endif /* MEASURE_LATENCY */
		


		// Process the batch

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
		gettimeofday(&micro_start, NULL);
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */

		PRINT(V_DEBUG, "Processing %d packets\n\n", num_packets);
		process_packets_sequential(p, results, num_packets);

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
		gettimeofday(&micro_stop, NULL);
		double total_time = (micro_stop.tv_sec - micro_start.tv_sec) * 1000000.0 + (micro_stop.tv_usec - micro_start.tv_usec);

		micro_nIters++;
		avg_micro += total_time;

		PRINT(V_DEBUG_TIMING, "Performance: %f msec\n", total_time);
#endif /*MEASURE_PROCESSING_MICROBENCHMARK*/

		// Return the batch of packets to click for forwarding
		send_packets(client, p, num_packets);

#ifdef MEASURE_LATENCY
		gettimeofday(&lat_stop, NULL);
		max_latency = (lat_stop.tv_sec - lat_start_oldest.tv_sec) * 1000000.0 + (lat_stop.tv_usec - lat_start_oldest.tv_usec);
		min_latency = (lat_stop.tv_sec - lat_start_newest.tv_sec) * 1000000.0 + (lat_stop.tv_usec - lat_start_newest.tv_usec);
		PRINT(V_DEBUG_TIMING, "Latencies: Max: %f msec   Min: %f msec\n", max_latency, min_latency);

		lat_nIters++;
		avg_max_latency += max_latency; // store cummulative latency; divide later
		avg_min_latency += min_latency;
#endif /* MEASURE_LATENCY */

#ifdef MEASURE_BANDWIDTH
			packets_processed += num_packets;
#endif /* MEASURE_BANDWIDTH */
			
			
		// Print results
		PRINT(V_DEBUG, "Results:\n");
		int i;
		for (i = 0; i < get_batch_size(); i++) {
			PRINT(V_DEBUG, "%d, ", results[i]);
		}
		PRINT(V_DEBUG, "\n\n\n");
	}


#ifdef MEASURE_BANDWIDTH
	// Calculate how many packets we processed per second
	gettimeofday(&bw_stop, NULL);
	double total_time = (bw_stop.tv_sec - bw_start.tv_sec) + (bw_stop.tv_usec - bw_start.tv_usec) / 1000000.0;
	double pkts_per_sec = double(packets_processed) / total_time;	

	PRINT(V_INFO, "Bandwidth: %f packets per second   (64B pkts ==> %f Gbps)\n", pkts_per_sec, pkts_per_sec * 64.0 / 1000000.0);
#endif /* MEASURE_BANDWIDTH */

#ifdef MEASURE_PROCESSING_MICROBENCHMARK
	avg_micro /= micro_nIters;
	PRINT(V_INFO, "Average processing time: %f msec\n", avg_micro);
#endif /* MEASURE_PROCESSING_MICROBENCHMARK */

#ifdef MEASURE_LATENCY
	avg_max_latency /= lat_nIters;
	avg_min_latency /= lat_nIters;
	PRINT(V_INFO, "Average latency: Max: %f msec,    Min: %f\n", avg_max_latency, avg_min_latency);
#endif /* MEASURE_LATENCY */

	return EXIT_SUCCESS;
}


// Catch Ctrl-C
void sig_handler (int sig)
{
	_do_run = false; 
}


/**
 * Program main
 */
int main(int argc, char **argv)
{

	if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
		checkCmdLineFlag(argc, (const char **)argv, "?"))
	{
		printf("Usage -device=n (n >= 0 for deviceID)\n");
		printf("	  -batch=n  (Sets the number of packets in a batch; n > 0)\n");
		printf("	  -wait=n   (Sets how long we wait (milliseconds) for a complete batch of packets; n > 0)\n");
		printf("	  -block=n  (Sets the number of threads in a block; n > 0)\n");
		printf("	  -sequential  (runs router in CPU-only mode w/ sequential code)\n");
		printf("      -runtime=n  (Sets a runtime in runtime; n > 0)\n");

		exit(EXIT_SUCCESS);
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "batch"))
	{
		int size = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
		set_batch_size(size);
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "wait"))
	{
		int wait = getCmdLineArgumentInt(argc, (const char **)argv, "wait");
		set_batch_wait(wait);
	}
	
	int block_size = DEFAULT_BLOCK_SIZE;
	if (checkCmdLineFlag(argc, (const char **)argv, "block"))
	{
		int n = getCmdLineArgumentInt(argc, (const char **)argv, "block");
		if (n > 0) {
			block_size = n;
		}
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "runtime"))
	{
		int n = getCmdLineArgumentInt(argc, (const char **)argv, "runtime");
		if (n > 0) {
			runtime = n;
		}
	}

	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		cudaSetDevice(devID);
	}

	cudaDeviceProp deviceProp;
	check_error(cudaGetDevice(&devID), "cudaGetDevice", __LINE__);
	check_error(cudaGetDeviceProperties(&deviceProp, devID), "cudaGetDeviceProperties", __LINE__);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	PRINT(V_INFO, "GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	
	
	// Set up the socket for receiving packets from click
	int server_sockfd = init_server_socket();
	if(server_sockfd == -1) {
	  return -1;
	}

	// Set up the socket for returning packets to click
	udpc client = init_client_socket();
	if(client.fd == -1) {
	  return -1;
	}

	//test(server_sockfd);
	
	// Catch Ctrl-C
	signal (SIGQUIT, sig_handler);
	signal (SIGINT, sig_handler);

	// Start the router!
	if (checkCmdLineFlag(argc, (const char **)argv, "sequential"))
	{
		return run_sequential(argc, argv, server_sockfd, client);
	}
	else
	{
		return run(argc, argv, block_size, server_sockfd, client);
	}
}
