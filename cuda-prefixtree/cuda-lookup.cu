/**
 * For licensing, see ./LICENSE
 *
 * Currently, this code allows insertions and lookups, but not
 * deletions. The data structure is a simple trie, which provides an
 * O(log n) bound on insertions and lookups. Deletions wouldn't be
 * tricky to add.
 *
 * When performing a lookup, bit comparisons decide left/right
 * traversal from the head of the tree, and the prefix length defines
 * a maximum depth when inserting. The lookup function will traverse
 * the tree until it determines that no more specific match than the
 * best already found is possible. The code will replace all valid IP
 * addresses (according to inet_pton()) with the matching prefix, or
 * "NF" if there was no match. It will not attempt to match tokens
 * that are not prefixes, but will print them out in the output.
 *
 * The code reads the lines to convert from standard input; it reads a
 * list of prefixes from a file, specified by the "-f" parameter. The
 * prefix file should contain one prefix per line, with the prefix and
 * the netmask separated by a space. All output is sent to standard
 * output.
 */

#include "cuda-lookup.cuh"

#include <stdio.h>
#include <math.h>
//#include "cuPrintf.cu"

#include <cuda_runtime.h>

int gpudebug = 1;
#define DEBUG(...) do { if (gpudebug) fprintf(stdout, __VA_ARGS__); } while (0)


//void cuda_lookup(struct iplookup_node *ilun, struct internal_node* n);

//int cuda_lpm_lookup(struct lpm_tree* tree, struct iplookup_node *ilun);



__device__ void cuda_lookup(struct iplookup_node *ilun, struct internal_node* n)
{
	uint32_t b = MAX_BITS;
	struct internal_node* next = n;

	do {
		n = next;
		b--;

		//parent = (struct internal_node*)n;
		uint32_t v_bit = ilun->ip & ((uint32_t)pow((double)2, (double)b));

		/* If we've found an internal node, determine which
		   direction to descend. */
		if (v_bit) {
			//next = n->r;
			next = (struct internal_node*)((char*)n + n->r_offset);
		}
		else {
			//next = n->l;
			next = (struct internal_node*)((char*)n + n->l_offset);
		}

		if (n->type == DAT_NODE) {
			struct data_node* node = (struct data_node*)n;


			uint32_t mask = 0xFFFFFFFF;

			mask = mask - ((uint32_t)pow((double)2, (double)(32 - node->netmask)) - 1);

			if ((ilun->ip & mask) == node->prefix) {
                ilun->port = node->port;
			}
			else {
				break;
			}
		}
    } while (next != n); //termination when offset is 0 and they are equal
	//} while (next != NULL);

}


__global__ void cuda_lpm_lookup(char* d_serializedtree, struct iplookup_node *ilun_array, uint32_t ilunarraysize)
{
    //int i = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= ilunarraysize)
        return;

        
    //cuda_lookup(&(ilun_array[i]), (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset));
    struct iplookup_node *ilun = &(ilun_array[i]);
    struct internal_node* n = (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset);
    
    ilun->port = MAX_BITS;
    uint32_t b = MAX_BITS;
	struct internal_node* next = n;
    
	do {
		n = next;
		b--;
        
		//parent = (struct internal_node*)n;
		uint32_t v_bit = ilun->ip & ((uint32_t)pow((double)2, (double)b));
        
		/* If we've found an internal node, determine which
         direction to descend. */
		if (v_bit) {
			//next = n->r;
			next = (struct internal_node*)((char*)n + n->r_offset);
		}
		else {
			//next = n->l;
			next = (struct internal_node*)((char*)n + n->l_offset);
		}
        
		if (n->type == DAT_NODE) {
			struct data_node* node = (struct data_node*)n;
            
            
			uint32_t mask = 0xFFFFFFFF;
            
			mask = mask - ((uint32_t)pow((double)2, (double)(32 - node->netmask)) - 1);
            
			if ((ilun->ip & mask) == node->prefix) {
                ilun->port = node->port;
			}
			else {
				break;
			}
		}
    } while (next != n); //termination when offset is 0 and they are equal
	//} while (next != NULL);

}




/* lpm_lookup:
 * Perform a lookup. Given a string 'ip_string' convert to the
 * best-matching prefix if the string is a valid IPv4 address
 * (according to inet_pton), and store it in 'output' and return 1. If
 * no match is found, store the string "NF" in 'output' and return
 * 1. If 'ip_string' is not a valid IPv4 address, return 0, and
 * 'output' is not modified.
 */
//int lpm_lookup(struct lpm_tree* tree, char* ip_string, char* output)
//int lpm_lookup(struct lpm_tree* tree, struct iplookup_node *ilun, char* ip_string, char* output)

/*
__global__ void cuda_lpm_lookup(char* d_serializedtree, struct iplookup_node *ilun_array, uint32_t ilunarraysize)
{
    //int i = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<ilunarraysize)
        cuda_lookup(&(ilun_array[i]), (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset));
    //cuPrintf("here\n");
}
*/
__global__ void cuda_addonip(struct iplookup_node *ilun_array, uint32_t ilunarraysize)
{
    //int i = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<ilunarraysize)
        (&ilun_array[i])->ip += MAX_BITS;
    
}

char* _transfer_to_gpu(char *buffer, uint32_t size) {
    //allocate space on the device for the tree
    char *d_buffer = NULL;
    cudaError_t err;

    err = cudaMalloc((void **)&d_buffer, size);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory on GPU (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    //copy buffer on the device
    err = cudaMemcpy(d_buffer, buffer, size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy tree on the GPU (error code %s)!\n", cudaGetErrorString(err));
        exit(-2);
    }
    DEBUG("Transfer to device (%d bytes from %p) successful.\n", size, buffer);
    
    return d_buffer;
}

void* _transfer_to_host(char *d_buffer, char *buffer, uint32_t size) {
    cudaError_t err;
    
    //memset(buffer, '\0', size);
    
    //copy device buffer to the host buffer  
    err = cudaMemcpy(buffer, d_buffer, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host (error code %s)!\n", cudaGetErrorString(err));
        exit(-4);
    }
    DEBUG("Transfer to host (%d bytes at %p) successful.\n", size, buffer);
    
    return (void*)buffer;
}

char* go_cuda(char *serializedtree, uint32_t treesize, struct iplookup_node *ilun_array, uint32_t ilunarraysize) {
    char *d_serializedtree = NULL;
    struct iplookup_node* d_ilun_array = NULL;
    cudaError_t err;
    
    //cudaPrintfInit();
    
    DEBUG("go_cuda received: treeser %p size %d ilun %p ilunarraysize %d\n", serializedtree, treesize, &(ilun_array[0]), ilunarraysize);


    if(serializedtree != NULL) // the idea is not to transfer if not needed, in that case take old pointer
        d_serializedtree = _transfer_to_gpu(serializedtree, treesize);
    
    d_ilun_array = (struct iplookup_node *)_transfer_to_gpu((char *)ilun_array, ilunarraysize * sizeof(struct iplookup_node) );
    
    
    int threadsPerBlock = 256;
    int blocksPerGrid =(ilunarraysize + threadsPerBlock - 1) / threadsPerBlock;
    DEBUG("CUDA launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    cuda_lpm_lookup<<<blocksPerGrid, threadsPerBlock>>>(d_serializedtree, d_ilun_array, ilunarraysize);
    
    //cuda_addonip<<<blocksPerGrid, threadsPerBlock>>>(d_ilun_array, ilunarraysize);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch lookup kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-3);
    }
    else
        DEBUG("CUDA launch successful!\n");
    
    //d_ilun_array->port += 1;
    
    _transfer_to_host((char *)d_ilun_array, (char *)ilun_array, ilunarraysize*sizeof(struct iplookup_node));
    
    cudaFree(d_serializedtree);
    cudaFree(d_ilun_array);

    cudaDeviceReset();
    
    DEBUG("go_cuda delivers: treeser %p size %d ilun %p ilunarraysize %d\n", serializedtree, treesize, &(ilun_array[0]), ilunarraysize);

    return serializedtree;
}
