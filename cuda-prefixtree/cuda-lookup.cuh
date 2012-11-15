#ifndef CUDA_LOOKUP_H
#define CUDA_LOOKUP_H

#include "tree.h"  

int go_cuda(char *serializedtree, uint32_t treesize, struct iplookup_node *ilun_array, uint32_t ilunarraysize);

#endif
