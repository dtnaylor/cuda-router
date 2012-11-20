#ifndef CUDA_LOOKUP_H
#define CUDA_LOOKUP_H

#include "tree.h"  

char* go_cuda(char *serializedtree, uint32_t treesize, struct iplookup_node *ilun_array, uint32_t ilunarraysize);

void* _transfer_to_host(char *d_buffer, char *buffer, uint32_t size);

char* _transfer_to_gpu(char *buffer, uint32_t size);

void cuda_lpm_lookup_oncpu(char* d_serializedtree, struct iplookup_node *ilun_array, uint32_t ilunarraysize);


__global__ void cuda_lpm_lookup(char* d_serializedtree, struct iplookup_node *ilun_array, uint32_t ilunarraysize);

#endif
