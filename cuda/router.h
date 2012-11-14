#ifndef ROUTER_H
#define ROUTER_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Packet collector
#include <packet-collector.hh>

#define SWAP(a,b,type) { type tmp=(a); (a)=(b); (b)=tmp; }


__global__ void process_packets_firewall(packet *p, int *results, int num_packets, int block_size);


/**
 * Checks the supplied cuda error for failure
 */
inline cudaError_t check_error(cudaError_t error, char* error_str, int line)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s returned error (line %d): %s\n", error_str, line, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	return error;
}

/**
 * Checks that the supplied pointer is not NULL
 */
inline void check_malloc(void *p, char* error_str, int line)
{
	if (p == NULL) {
		fprintf(stderr, "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
		exit(EXIT_FAILURE);
	}
}

#endif /* ROUTER_H */
