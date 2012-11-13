#include <router.h>

#define DEFAULT_BLOCK_SIZE 32


/**
 * Start a loop:
 *	1) Gather packets
 *  2) Copy packets to GPU and process
 *	3) Copy results back and print performance stats
 */
int run(int argc, char **argv, int block_size, int sockfd)
{
	unsigned int buf_size = sizeof(packet)*get_batch_size();
	unsigned int results_size = sizeof(int)*get_batch_size();

    // Allocate host memory for up to batch_size packets
    packet* h_p = (packet *)malloc(buf_size);
	if (h_p == NULL) {
		fprintf(stderr, "Failed to allocate packet buffer\n");
		exit(EXIT_FAILURE);
	}
	// Allocate host memory for results
	int *h_results = (int*)malloc(results_size);
	if (h_results == NULL) {
		fprintf(stderr, "Failed to allocate results array\n");
		exit(EXIT_FAILURE);
	}

    // Allocate device memory for up to batch_size packets
	// TODO: wait and allocate only the amount needed after we receive?
    packet *d_p;
    check_error(cudaMalloc((void **) &d_p, buf_size), "cudaMalloc d_p", __LINE__);
	// Allocate device memory for results
	int *d_results;
    check_error(cudaMalloc((void **) &d_results, results_size), "cudaMalloc d_results", __LINE__);


	// Receive a batch of packets
	int num_packets = get_packets(sockfd, h_p);
	while (num_packets ==0) {
		num_packets = get_packets(sockfd, h_p);
		printf("Received no packets\n");
	}
	/*if (num_packets >= 0) {
		printf("Received no packets\n");
		return 0;
	}*/


    // Copy host memory to device
    check_error(cudaMemcpy(d_p, h_p, buf_size, cudaMemcpyHostToDevice), "cudaMemcpy (d_p, h_p)", __LINE__);


    // Setup execution parameters
	int threads_in_block = block_size;
	int blocks_in_grid = get_batch_size() / threads_in_block;  // FIXME: optimize if we don't have a full batch
	if (get_batch_size() % threads_in_block != 0) {
		blocks_in_grid++;  // need an extra block for the extra threads
	}


    // Performs warmup operation so subsequent executions have accurate timing
    process_packets_firewall<<< blocks_in_grid, threads_in_block >>>(d_p, d_results, num_packets, block_size);
    cudaDeviceSynchronize();


    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
	check_error(cudaEventCreate(&start), "Create start event", __LINE__);

    cudaEvent_t stop;
    check_error(cudaEventCreate(&stop), "Create stop event", __LINE__);

    // Record the start event
    check_error(cudaEventRecord(start, NULL), "Record start event", __LINE__);

    // Execute the kernel
    process_packets_firewall<<< blocks_in_grid, threads_in_block >>>(d_p, d_results, num_packets, block_size);

    // Record the stop event
    check_error(cudaEventRecord(stop, NULL), "Record stop event", __LINE__);

    // Wait for the stop event to complete
    check_error(cudaEventSynchronize(stop), "Failed to synchronize stop event", __LINE__);

    float msecTotal = 0.0f;
    check_error(cudaEventElapsedTime(&msecTotal, start, stop), "Getting time elapsed b/w events", __LINE__);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal; //  / nIter;
    printf(
        "Performance= Time= %.3f msec, WorkgroupSize= %u threads/block\n",
        msecPerMatrixMul,
        threads_in_block);

    // Copy result from device to host
    check_error(cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost), "cudaMemcpy (h_results, d_results)", __LINE__);


	// Check for correctness
	bool correct = true;

	int i;
	for (i = 0; i < get_batch_size(); i++) {
		printf("%d, ", h_results[i]);
	}
	printf("\n\n");


    // Clean up memory
    free(h_p);
    free(h_results);
    cudaFree(d_p);
    cudaFree(d_results);

    cudaDeviceReset();


    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
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

	sleep(5);

	// Start the router!
    int result = run(argc, argv, block_size, sockfd);
    exit(result);
}
