#ifndef ROUTER_H
#define ROUTER_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/tcp.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Packet collector
#include <packet-collector.hh>

#define SWAP(a,b,type) { type tmp=(a); (a)=(b); (b)=tmp; }


#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) fprintf(stdout, __VA_ARGS__); } while (0)


// Change this define to determine which processing function is used
// (e.g., firewall, longest prefix match, etc.)
//#define FIREWALL
//#define LPM_TRIE
//#define BOTH_SEQ
#define BOTH_PAR

//#define PINNED_PACKET_MEMORY  // packet arrays are pinned and mapped
#define PINNED_MEMORY  // results arrays are pinned and mapped



__global__ void process_packets(packet *p, int *results, int num_packets, int block_size);
void setup();
void teardown();
void process_packets_sequential(packet *p, int *results, int num_packets);
void setup_sequential();

#if defined FIREWALL || defined BOTH_SEQ || defined BOTH_PAR
int set_num_rules(int s);
int get_num_rules();
#endif /* FIREWALL */


/**
 * Checks the supplied cuda error for failure
 */
inline cudaError_t check_error(cudaError_t error, char* error_str, int line)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s returned error (line %d): %s\n", error_str, line, cudaGetErrorString(error));
        cudaDeviceReset();
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
        cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#endif /* ROUTER_H */
