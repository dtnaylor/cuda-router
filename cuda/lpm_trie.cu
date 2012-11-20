#include <router.h>
#include "../cuda-prefixtree/mymalloc.h"
#include "../cuda-prefixtree/tree.h"
#include "../cuda-prefixtree/cuda-lookup.cuh"

#ifdef USEDEBUG
    int lpmtriegpudebug = 1;
    #define DEBUG(...) do { if (lpmtriegpudebug) fprintf(stdout, __VA_ARGS__); } while (0)
    #else
    int lpmtriegpudebug = 0;
    #define DEBUG(...) (void)0
#endif



#ifdef LPM_TRIE

#define PREFIX_FILE "prefixes.txt"


__device__ __constant__ char *dd_serializedtree = NULL;
__device__ __constant__ int d_serializedtree_size;

struct lpm_serializedtree sTree;
char *d_serializedtree = NULL;


/**
 * A CUDA kernel to be executed on the GPU.
 * 
 * Fills in the results array with the outbound port for each packet
 */
__global__ void
process_packets(packet *p, int *results, int num_packets, int block_size)
{
	int packet_index = blockIdx.x * block_size + threadIdx.x;
    
	struct ip *ip_hdr = (struct ip*)p[packet_index].ip;
    int *ret = &(results[packet_index]);
    uint32_t ip_addr = *((uint32_t*)&ip_hdr->ip_dst);
}


void _setup_trie() {
    char* input = (char*) PREFIX_FILE;
	//int opt,i;
	struct lpm_tree* tree;
	//char* ifile = (char*) "/dev/stdin";
    FILE* in;

    DEBUG("Init serializer\n");
    
	init_myserializer((void**)&tree);
    
    //build_serializedtree(input);
    
    int insertions;
    int ret;
    
    DEBUG("Create tree (size of pointer=%d, sizeof char=%d)\n", (int)sizeof(tree), (int)sizeof(char));
	/* Create a fresh tree. */
	tree = lpm_init();
    
    /* Read in all prefixes. */
	in = fopen(input, "r");
    insertions = 0;
	while (1) {
		char ip_string[40]; /* Longest v6 string */
		int mask;
        int port;
		uint8_t rt;
		char line[4096];
		char* rtp;
        
		memset(line, '\0', 4096);
		rtp = fgets(line, 4096, in);
		/* EOL */
		if (rtp == NULL) {
			break;
		}
		rt = sscanf(line, "%39s %d %d%*[^\n]", ip_string, &mask, &port);
		if (rt < 2) {
			continue;
		}
        
		/* Doesn't handle IPv6; skip anything that looks like a v6 address. */
		if (strstr(ip_string, ":") != NULL) {
			continue;
		}
        
        while(1) {
            ret = lpm_insert(tree, ip_string, mask, port);
            if(ret) break;
            else if(cannotdouble_myserializer() == 1) {
                printf("ERROR: cannot double my serialized\n");
                exit(-1);
            }
            else //myserialized has been doubled, so repeat
                continue;
            
        }
        DEBUG("Inserted %d prefixes\n", ++insertions);
	}
	fclose(in);
        

    sTree.serializedtree_size = finalize_serialized((void**)(&(sTree.serialized_tree)));

    
    //tree = (struct lpm_tree*) sTree.serialized_tree;
}


void _setup_GPU() {
    d_serializedtree = _transfer_to_gpu(sTree.serialized_tree, sTree.serializedtree_size);
        
	check_error(cudaMemcpyToSymbol(dd_serializedtree, &d_serializedtree, sizeof(d_serializedtree)), "cudaMemcpyToSymbol (dd_serializedtree, d_serializedtree)", __LINE__);
	
	check_error(cudaMemcpyToSymbol(d_serializedtree_size, &sTree.serializedtree_size, sizeof(int)),  "cudaMemcpyToSymbol (d_serializedtree_size, sTree.serializedtree_size)", __LINE__);
}


/**
 * LPM-specific setup. This will be called a single time by router.cu 
 * before the kernel function runs for the first time
 */
void setup()
{
    _setup_trie();
    _setup_GPU();
}



/**
 * A CPU-only sequential version of the firewall packet processing function
 */
void process_packets_sequential(packet *p, int *results, int num_packets)
{
    int packet_index = 0;
    
    DEBUG("looking up %d packets\n", num_packets);
    while(packet_index < num_packets) {
        if(packet_index >= num_packets)
            return;
        
        struct ip *ip_hdr = (struct ip*)p[packet_index].ip;
        int *ret = &(results[packet_index]);
        uint32_t ip_addr = *((uint32_t*)&ip_hdr->ip_dst);

        struct internal_node* n = (struct internal_node*)((char*)sTree.serialized_tree + ((struct lpm_tree*)sTree.serialized_tree)->h_offset);
        
        *ret = 0;
        uint32_t b = MAX_BITS;
        struct internal_node* next = n;
        
        do {
            n = next;
            b--;
            //parent = (struct internal_node*)n;
            //uint32_t v_bit = ilun->ip & ((uint32_t)pow((double)2, (double)b));
            uint32_t v_bit = ip_addr & ((uint32_t)1 << b);
            
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
                
                if ((ip_addr & mask) == node->prefix) {
                    *ret = node->port;
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
        
        packet_index++;
    }
    DEBUG("packets looked-up\n");
#ifdef USEDEBUG
    for(packet_index=0; packet_index<num_packets; packet_index++) {
        DEBUG("packet %d port %d\n", packet_index, results[packet_index]);
    }
    DEBUG("results printed out\n");
#endif

}


/**
 * LPM-specific setup. This will be called a single time by router.cu
 * before the sequential CPU function runs for the first time
 */
void setup_sequential()
{
    _setup_trie();
}


/**
 * Firewall-specific teardown. This will be called a single time by router.cu 
 * after the kernel function runs last time
 */

void teardown()
{
	//free(h_rules);
    cudaFree(d_serializedtree);
}


#endif /* LPM_TRIE */
