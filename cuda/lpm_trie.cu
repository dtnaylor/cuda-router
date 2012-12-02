#include <router.h>
#include "../cuda-prefixtree/mymalloc.h"
#include "../cuda-prefixtree/tree.h"
#include "../cuda-prefixtree/cuda-lookup.cuh"


#ifdef LPM_TRIE

#ifdef USEDEBUG
    int lpmtriegpudebug = 1;
    #define DEBUG(...) do { if (lpmtriegpudebug) fprintf(stdout, __VA_ARGS__); } while (0)
    #else
    int lpmtriegpudebug = 0;
    #define DEBUG(...) (void)0
#endif

#define PREFIX_FILE "prefixes.txt"


__device__ __constant__ char *dd_serializedtree = NULL;
__device__ __constant__ int d_serializedtree_size;

struct lpm_serializedtree sTree;
char *d_serializedtree = NULL;

#define D_HTONL(n) (((((unsigned long)(n) & 0xFF)) << 24) | \
((((unsigned long)(n) & 0xFF00)) << 8) | \
((((unsigned long)(n) & 0xFF0000)) >> 8) | \
((((unsigned long)(n) & 0xFF000000)) >> 24))

/**
 * A CUDA kernel to be executed on the GPU.
 *
 * Fills in the results array with the outbound port for each packet
 */
__global__ void
process_packets(packet *p, int *results, int num_packets, int block_size)
{
	int packet_index = blockIdx.x * block_size + threadIdx.x;
    
    if(packet_index >= num_packets)
        return;
    
    struct ip *ip_hdr = (struct ip*)p[packet_index].ip;
    int *ret = &(results[packet_index]);
    
    uint32_t ip_addr = *((uint32_t*)&ip_hdr->ip_dst);
    ip_addr = D_HTONL(ip_addr);
    
    struct internal_node* n = (struct internal_node*)((char*)dd_serializedtree + ((struct lpm_tree*)dd_serializedtree)->h_offset);
    
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
    
    int ret;
    
    DEBUG("Create tree (size of pointer=%d, sizeof char=%d)\n", (int)sizeof(tree), (int)sizeof(char));
	/* Create a fresh tree. */
	tree = lpm_init();
    
    /* Read in all prefixes. */
	in = fopen(input, "r");
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
        ip_addr = HTONL(ip_addr);

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
        
        DEBUG("packet %d (addr %s) port %d\n", packet_index, inet_ntoa(ip_hdr->ip_dst/* *((struct in_addr*)& (HTONL(ip_addr)))*/), results[packet_index]);

        
        packet_index++;
    }
    DEBUG("packets looked-up\n");
}


/**
 * LPM-specific setup. This will be called a single time by router.cu
 * before the sequential CPU function runs for the first time
 */
void setup_sequential()
{
    _setup_trie();
    
//    char *ipaddr = "198.168.1.45";
//    uint32_t tmp, tmp2, tmp3, tmp4, tmp5;
//    inet_pton(AF_INET, ipaddr, &tmp);
//    inet_aton(ipaddr, (struct in_addr *)&tmp2);
//    packet pkt;
//    int results;
//    struct ip *ip_hdr = (struct ip*)pkt.ip;
//
//    ip_hdr->ip_dst = *((struct in_addr*) &tmp);
//    process_packets_sequential(&pkt, &results, 1);
//    printf("res: %d\n", results);
//    
//    ip_hdr->ip_dst = *((struct in_addr*) &tmp2);
//    process_packets_sequential(&pkt, &results, 1);
//    printf("res: %d\n", results);
//    
//    tmp3 = ntohl(tmp);
//    ip_hdr->ip_dst = *((struct in_addr*) &tmp3);
//    process_packets_sequential(&pkt, &results, 1);
//    printf("res: %d\n", results);
//    
//    tmp4 = ntohl(tmp2);
//    ip_hdr->ip_dst = *((struct in_addr*) &tmp4);
//    process_packets_sequential(&pkt, &results, 1);
//    printf("res: %d\n", results);
//    
//    tmp5 = HTONL(tmp4);
//    if(tmp5==tmp2)
//        printf("tmp5 equals tmp2");
//    else
//        printf("tmp5 NOT equals tmp2");
//    
//    if(tmp3==tmp4)
//        printf("tmp3 equals tmp4");
//    else
//        printf("tmp3 NOT equals tmp4");
    
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
