#include <router.h>


/**
 * A CUDA kernel to be executed on the GPU.
 * Checks each packet in the array p against a set of firewall rules.
 * Fill the array results with RESULT_FORWARD, RESULT_DROP, or RESULT_UNSET
 */
__global__ void
process_packets_firewall(packet *p, int *results, int num_packets, int block_size)
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
