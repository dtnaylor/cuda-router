#include <router.h>

#ifdef LPM_TRIE

/**
 * A CUDA kernel to be executed on the GPU.
 */
__global__ void
process_packets(packet *p, int *results, int num_packets, int block_size)
{

	int packet_index = blockIdx.x * block_size + threadIdx.x;

}

/**
 * LPM-specific setup. This will be called a single time by router.cu 
 * before the kernel function runs for the first time
 */
void setup()
{

}



/**
 * A CPU-only sequential version of the firewall packet processing function
 */
void process_packets_sequential(packet *p, int *resultsl, int num_packets)
{

}


/**
 * LPM-specific setup. This will be called a single time by router.cu
 * before the sequential CPU function runs for the first time
 */
void setup_sequential()
{

}

#endif /* LPM_TRIE */
