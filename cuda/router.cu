#include <router.h>

#define DEFAULT_BLOCK_SIZE 32


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
int run(int argc, char **argv, int block_size, int sockfd)
{
	unsigned int buf_size = sizeof(packet)*get_batch_size();
	unsigned int results_size = sizeof(int)*get_batch_size();

    // Allocate host memory for two batches of up to batch_size packets
	// We will alternate between filling and processing these two buffers
	// (at any given time one of the buffers will either be being filled
	// or being processed)
    packet* h_p1 = (packet *)malloc(buf_size);
	if (h_p1 == NULL) {
		fprintf(stderr, "Failed to allocate packet buffer\n");
		exit(EXIT_FAILURE);
	}
    packet* h_p2 = (packet *)malloc(buf_size);
	if (h_p2 == NULL) {
		fprintf(stderr, "Failed to allocate packet buffer\n");
		exit(EXIT_FAILURE);
	}

	// Allocate host memory for 2 arrays of results
	int *h_results1 = (int*)malloc(results_size);
	if (h_results1 == NULL) {
		fprintf(stderr, "Failed to allocate results array\n");
		exit(EXIT_FAILURE);
	}
	int *h_results2 = (int*)malloc(results_size);
	if (h_results2 == NULL) {
		fprintf(stderr, "Failed to allocate results array\n");
		exit(EXIT_FAILURE);
	}

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


	// Receive initial batch of packets to kick off the pipeline
	int num_packets = get_packets(sockfd, h_p1);
	while (num_packets == 0) {
		num_packets = get_packets(sockfd, h_p1);
	}


    // Copy host memory to device
    check_error(cudaMemcpy(d_p1, h_p1, buf_size, cudaMemcpyHostToDevice), "cudaMemcpy (d_p1, h_p1)", __LINE__);


    // Setup execution parameters
	int threads_in_block = block_size;
	int blocks_in_grid = get_batch_size() / threads_in_block;  // FIXME: optimize if we don't have a full batch
	if (get_batch_size() % threads_in_block != 0) {
		blocks_in_grid++;  // need an extra block for the extra threads
	}
   

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	check_error(cudaEventCreate(&start), "Create start event", __LINE__);

	cudaEvent_t stop;
	check_error(cudaEventCreate(&stop), "Create stop event", __LINE__);


    // Perform warmup operation so subsequent CUDA kernel executions have accurate timing
    process_packets_firewall<<< blocks_in_grid, threads_in_block >>>(d_p1, d_results1, num_packets, block_size);
    cudaDeviceSynchronize(); // wait for warmup GPU execution to finish



	/* The main loop:
		1) Execute the CUDA kernel
		2) While it's executing, copy the results from the last execution back to the host
		3) While it's executing, copy the packets for the next execution to the GPU
		4) When it finishes executing, print out some timing information   */

	bool first_execution = true;
	packet *h_p_current = h_p1;
	packet *h_p_next = h_p2;
	packet *d_p_current = d_p1;
	packet *d_p_next = d_p2;
	int *h_results_current = h_results1;
	int *h_results_previous = h_results2;
	int *d_results_current = d_results1;
	int *d_results_previous = d_results2;
	
	while(1) {

    	// Record the start event
    	check_error(cudaEventRecord(start, NULL), "Record start event", __LINE__);

    	// Execute the kernel
		printf("vvvvv   Begin processing %d packets   vvvvv\n\n", num_packets);
    	process_packets_firewall<<< blocks_in_grid, threads_in_block >>>(d_p_current, d_results_current, num_packets, block_size);

    	// Record the stop event
    	check_error(cudaEventRecord(stop, NULL), "Record stop event", __LINE__);


		// TODO: double-check that stuff is really executing when I think it is.
		// I think that calling cudaEventRecord(stop) right before this will record
		// when the kernel stops executing, but won't block until this actually happens.
		// The cudaEventSynchronize call below does block until the kernel stops.
		// So, I think anything we do here will execute on the CPU while the GPU executes
		// the kernel call we made above.

		if (!first_execution) {
			// Copy the last set of results back from the GPU
    		check_error(cudaMemcpy(h_results_previous, d_results_previous, results_size, cudaMemcpyDeviceToHost), "cudaMemcpy (h_results, d_results)", __LINE__);
		
			// Print results
			printf("Results from last batch:\n");
			int i;
			for (i = 0; i < get_batch_size(); i++) {
				printf("%d, ", h_results_previous[i]);
			}
			printf("\n\n");
		}

		// Get next batch of packets and copy them to the GPU
		// FIXME: We're forcing the results from the current execution to wait
		// until we get the next batch of packets. Is this OK?
		num_packets = get_packets(sockfd, h_p_next);
		while (num_packets == 0) {
			num_packets = get_packets(sockfd, h_p_next);
		}
    	check_error(cudaMemcpy(d_p_next, h_p_next, buf_size, cudaMemcpyHostToDevice), "cudaMemcpy (d_p_next, h_p_next)", __LINE__);




    	// Wait for the stop event to complete (which waits for the kernel to finish)
    	check_error(cudaEventSynchronize(stop), "Failed to synchronize stop event", __LINE__);

    	float msecTotal = 0.0f;
    	check_error(cudaEventElapsedTime(&msecTotal, start, stop), "Getting time elapsed b/w events", __LINE__);

    	// Compute and print the performance
    	printf(
    	    "Performance= Time= %.3f msec, WorkgroupSize= %u threads/block\n",
    	    msecTotal,
    	    threads_in_block);
		printf("^^^^^   Done processing batch   ^^^^^\n\n\n");




		// Get ready for the next loop iteration
		first_execution = false;
		SWAP(h_p_current, h_p_next, packet*);
		SWAP(d_p_current, d_p_next, packet*);
		SWAP(h_results_current, h_results_previous, int*);
		SWAP(d_results_current, d_results_previous, int*);

	}




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

void test(int sockfd) 
{
	printf("Batch Size: %d\n", get_batch_size());
	
	// Initialize a buffer for storing up to batch_size packets
    packet* p = (packet *)malloc(sizeof(packet)*get_batch_size());
    
    while(1) {
      int num_packets = get_packets(sockfd, p);
      printf("i = %d\n", num_packets);

	  if (num_packets > 0) {
	  	struct ip *ip_hdr = (struct ip*)p->buf;
		struct udphdr *udp_hdr = (struct udphdr*)&(p->buf[sizeof(struct ip)]);
		printf("Dest: %s (%u)\n", inet_ntoa(ip_hdr->ip_dst), ntohs(udp_hdr->uh_dport));
		printf("Source: %s (%u)\n", inet_ntoa(ip_hdr->ip_src), ntohs(udp_hdr->uh_sport));
		printf("Next proto: %u\n", ip_hdr->ip_p);
	  }
    }
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
        printf("      -batch=n  (Sets the number of packets in a batch; n > 0)\n");
        printf("      -block=n  (Sets the number of threads in a block; n > 0)\n");

        exit(EXIT_SUCCESS);
    }
    
	if (checkCmdLineFlag(argc, (const char **)argv, "batch"))
    {
        int size = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
        set_batch_size(size);
    }
	
	int block_size = DEFAULT_BLOCK_SIZE;
	if (checkCmdLineFlag(argc, (const char **)argv, "block"))
    {
        int n = getCmdLineArgumentInt(argc, (const char **)argv, "block");
		if (n > 0) {
        	block_size = n;
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

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	
	
	// Set up the socket for receiving packets from click
    int sockfd = init_socket();
    if(sockfd == -1) {
      return -1;
    }

	//test(sockfd);

	// Start the router!
    int result = run(argc, argv, block_size, sockfd);
    exit(result);
}
