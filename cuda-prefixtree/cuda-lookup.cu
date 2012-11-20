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
#include <cuda_runtime.h>

#ifdef USECUPRINTF    
#include "cuPrintf.cuh"
#if __CUDA_ARCH__ < 200     //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                       //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
blockIdx.y*gridDim.x+blockIdx.x,\
threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
__VA_ARGS__)
#endif

#endif


#ifdef USEDEBUG
int gpudebug = 1;
#else
int gpudebug = 0;
#endif

#define DEBUG(...) do { if (gpudebug) fprintf(stdout, __VA_ARGS__); } while (0)


//void cuda_lookup(struct iplookup_node *ilun, struct internal_node* n);

//int cuda_lpm_lookup(struct lpm_tree* tree, struct iplookup_node *ilun);



__device__ void cuda_lookup(struct iplookup_node *ilun, struct internal_node* n)
{
	uint32_t b = MAX_BITS;
	struct internal_node* next = n;
    ilun->port = (uint16_t) 0;

	do {
		n = next;
		b--;

		//parent = (struct internal_node*)n;
		//uint32_t v_bit = ilun->ip & ((uint32_t)pow((double)2, (double)b));
        uint32_t v_bit = ilun->ip & ((uint32_t)1 << b);

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

			//mask = mask - ((uint32_t)pow((double)2, (double)(32 - node->netmask)) - 1);
            mask = mask - (((uint32_t)1 << (32 - node->netmask)) - 1);

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
    uint16_t iterations = 0;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= ilunarraysize)
        return;

        
    //cuda_lookup(&(ilun_array[i]), (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset));
    struct iplookup_node *ilun = &(ilun_array[i]);
    struct internal_node* n = (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset);
    
    ilun->port = (uint16_t) 0;
    uint32_t b = MAX_BITS;
	struct internal_node* next = n;
    
#ifdef USECUPRINTF
    CUPRINTF("root is:%p, loffset %d roffset %d\n", next, next->l_offset, next->r_offset);
#endif
    
	do {
		n = next;
		b--;
        iterations++;
		//parent = (struct internal_node*)n;
		//uint32_t v_bit = ilun->ip & ((uint32_t)pow((double)2, (double)b));
        uint32_t v_bit = ilun->ip & ((uint32_t)1 << b);

#ifdef USECUPRINTF
        CUPRINTF("v_bit %u \t bits %u \t pow2 %u \t mypow %u\n", v_bit, b, (uint32_t)pow((double)2, (double)b), 1<<b);
#endif
		/* If we've found an internal node, determine which
         direction to descend. */
		if (v_bit) {
			//next = n->r;
			next = (struct internal_node*)((char*)n + n->r_offset);
#ifdef USECUPRINTF            
            CUPRINTF("next right is:%p loffset %d roffset %d\n", next, next->l_offset, next->r_offset);

#endif
		}
		else {
			//next = n->l;
			next = (struct internal_node*)((char*)n + n->l_offset);
#ifdef USECUPRINTF
            CUPRINTF("next left is:%p loffset %d roffset %d\n", next, next->l_offset, next->r_offset);

#endif
		}
        
		if (n->type == DAT_NODE) {
			struct data_node* node = (struct data_node*)n;
            
#ifdef USECUPRINTF
            CUPRINTF("data node found , iter %d\n", iterations);
#endif

            
			uint32_t mask = 0xFFFFFFFF;
            
			//mask = mask - ((uint32_t)pow((double)2, (double)(32 - node->netmask)) - 1);
            mask = mask - (((uint32_t)1 << (32 - node->netmask)) - 1);

			if ((ilun->ip & mask) == node->prefix) {
                ilun->port = node->port;
                iterations *=100;
			}
			else {
                iterations *=10;
				break;
			}
		}
        else {
            //if(next==n) ilun->port = 0;
        }
        
    } while (next != n); //termination when offset is 0 and they are equal
	//} while (next != NULL);
#ifdef USECUPRINTF
    CUPRINTF("abandoning , iter %d\n", iterations);
#endif
    ilun->port2 = iterations;
}

void cuda_lpm_lookup_oncpu(char* d_serializedtree, struct iplookup_node *ilun_array, uint32_t ilunarraysize)
{
    //int i = threadIdx.x;
    uint16_t iterations = 0;
    int i = 0;
    while(i < ilunarraysize) {
    if(i >= ilunarraysize)
        return;
    
    
    //cuda_lookup(&(ilun_array[i]), (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset));
    struct iplookup_node *ilun = &(ilun_array[i]);
    struct internal_node* n = (struct internal_node*)((char*)d_serializedtree + ((struct lpm_tree*)d_serializedtree)->h_offset);
    
    ilun->port = (uint16_t) 0;
    uint32_t b = MAX_BITS;
	struct internal_node* next = n;
    
	do {
		n = next;
		b--;
        iterations++;
		//parent = (struct internal_node*)n;
		//uint32_t v_bit = ilun->ip & ((uint32_t)pow((double)2, (double)b));
        uint32_t v_bit = ilun->ip & ((uint32_t)1 << b);

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
            
			//mask = mask - ((uint32_t)pow((double)2, (double)(32 - node->netmask)) - 1);
            mask = mask - (((uint32_t)1 << (32 - node->netmask)) - 1);

			if ((ilun->ip & mask) == node->prefix) {
                ilun->port = node->port;
                //iterations *=100;
			}
			else {
                //iterations *=10;
				break;
			}
		}
        else {
            //if(next==n) ilun->port = 0;
        }
        
    } while (next != n); //termination when offset is 0 and they are equal
	//} while (next != NULL);
    
    ilun->port2 = iterations;
        i++;
    }
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
    
#ifdef USECUPRINTF    cudaPrintfInit();
#endif
    
    //DEBUG("go_cuda received: treeser %p size %d ilun %p ilunarraysize %d\n", serializedtree, treesize, &(ilun_array[0]), ilunarraysize);


    if(serializedtree != NULL) // the idea is not to transfer if not needed, in that case take old pointer
        d_serializedtree = _transfer_to_gpu(serializedtree, treesize);
    
    d_ilun_array = (struct iplookup_node *)_transfer_to_gpu((char *)ilun_array, ilunarraysize * sizeof(struct iplookup_node) );
    
    
    int threadsPerBlock = 256;
    int blocksPerGrid =(ilunarraysize + threadsPerBlock - 1) / threadsPerBlock;
    //int blocksPerGrid =1;
    printf("CUDA launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    cuda_lpm_lookup<<<blocksPerGrid, threadsPerBlock>>>(d_serializedtree, d_ilun_array, ilunarraysize);
    
    //cuda_addonip<<<blocksPerGrid, threadsPerBlock>>>(d_ilun_array, ilunarraysize);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch lookup kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-3);
    }
    else
        printf("CUDA launch successful!\n");
    
    //d_ilun_array->port += 1;
     
    _transfer_to_host((char *)d_ilun_array, (char *)ilun_array, ilunarraysize*sizeof(struct iplookup_node));
    _transfer_to_host((char *)d_serializedtree, (char *)serializedtree, treesize);

#ifdef USECUPRINTF    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif
    
    cudaFree(d_serializedtree);
    cudaFree(d_ilun_array);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    //cuda_lpm_lookup_oncpu(serializedtree, ilun_array, ilunarraysize);
    
    //DEBUG("go_cuda delivers: treeser %p size %ud ilun %p ilunarraysize %ud (sizeof char %lu, sizeof char* %lu)\n", serializedtree, treesize, &(ilun_array[0]), ilunarraysize, sizeof(char), sizeof(char*));

    //DEBUG("strustrure sizes: structtree=%d pointertree=%d structintnode=%d structdatanode=%d structilun=%d uint64=%d,uint32=%d, uint16=%d):\n", sizeof(struct lpm_tree),sizeof(struct lpm_tree*), sizeof(struct internal_node),sizeof(struct data_node),sizeof(struct iplookup_node), sizeof(uint64_t), sizeof(uint32_t), sizeof(uint16_t));

    
    return serializedtree;
}
