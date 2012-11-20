#include <router.h>
#include "../cuda-prefixtree/mymalloc.h"
#include "../cuda-prefixtree/tree.h"


#ifdef LPM_TRIE

#define PREFIX_FILE "prefixes.txt"

/**
 * A CUDA kernel to be executed on the GPU.
 * 
 * Fills in the results array with the outbound port for each packet
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
    char* input = (char*) PREFIX_FILE;
	int opt,i;
	struct lpm_tree* tree;
    struct lpm_serializedtree sTree;
	char* ifile = (char*) "/dev/stdin";
    FILE* in;


    DEBUG("Init serializer\n");
    
	init_myserializer((void**)&tree);
    
    //build_serializedtree(input);
    
    
    int insertions, ret;
    
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

    
    tree = (struct lpm_tree*) sTree.serialized_tree;


}



/**
 * A CPU-only sequential version of the firewall packet processing function
 */
void process_packets_sequential(packet *p, int *results, int num_packets)
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
