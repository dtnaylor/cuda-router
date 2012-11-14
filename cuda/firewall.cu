#include <router.h>

#ifdef FIREWALL

/**
 * A CUDA kernel to be executed on the GPU.
 * Checks each packet in the array p against a set of firewall rules.
 * Fill the array results with RESULT_FORWARD, RESULT_DROP, or RESULT_UNSET
 */
__global__ void
process_packets(packet *p, int *results, int num_packets, int block_size)
{

	// TODO: Actually implement firewall

	int packet_index = blockIdx.x * block_size + threadIdx.x;
	struct ip *ip_hdr = (struct ip*)p[packet_index].buf;
	if (packet_index < num_packets) {
		results[packet_index] = ip_hdr->ip_p;
	} else {
		results[packet_index] = RESULT_UNSET;
	}

}

/**
 * Firewall-specific setup. This will be called a single time by router.cu 
 * before the kernel function runs for the first time
 */
void setup()
{
	// TODO: Copy firewall rules to GPU so the kernel function can use them
}



/**
 * A CPU-only sequential version of the firewall packet processing function
 */
void process_packets_sequential(packet *p, int *results, int num_packets)
{
	int i;
	for (i = 0; i < get_batch_size(); i++) {
		struct ip *ip_hdr = (struct ip*)p[i].buf;
		if (i < num_packets) {
			results[i] = ip_hdr->ip_p;
		} else {
			results[i] = RESULT_UNSET;
		}
	}
}


/**
 * Firewall-specific setup. This will be called a single time by router.cu
 * before the sequential CPU function runs for the first time
 */
void setup_sequential()
{

}

#endif /* FIREWALL */
