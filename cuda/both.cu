#include <router.h>
#include "../cuda-prefixtree/mymalloc.h"
#include "../cuda-prefixtree/tree.h"
#include "../cuda-prefixtree/cuda-lookup.cuh"

#if defined (BOTH_SEQ) || defined (BOTH_PAR)

#define PREFIX_FILE "prefixes.txt"

#ifdef USEDEBUG
    int lpmtriegpudebug = 1;
    #define DEBUG(...) do { if (lpmtriegpudebug) fprintf(stdout, __VA_ARGS__); } while (0)
    #else
    int lpmtriegpudebug = 0;
    #define DEBUG(...) (void)0
#endif

typedef struct _rule {
  uint32_t src_ip;
  uint32_t dst_ip;
  uint16_t src_port;
  uint16_t dst_port;
  uint8_t proto;
  int8_t action;
} rule;

int num_rules = 100;

int set_num_rules(int s) {
	if (s > 0) {
		num_rules = s;
	}
	return num_rules;
}

int get_num_rules() {
	return num_rules;
}

/* Global vars for firewall */
//unsigned long *h_rule_hashes;
//unsigned long *d_rule_hashes;
rule *h_rules;
rule *d_rules;
__device__ __constant__ rule *dd_rules;
int h_num_rules;
__device__ __constant__ int d_num_rules;


/* Global vars for lpm */
__device__ __constant__ char *dd_serializedtree = NULL;
__device__ __constant__ int d_serializedtree_size;

struct lpm_serializedtree sTree;
char *d_serializedtree = NULL;

#define D_HTONL(n) (((((unsigned long)(n) & 0xFF)) << 24) | \
((((unsigned long)(n) & 0xFF00)) << 8) | \
((((unsigned long)(n) & 0xFF0000)) >> 8) | \
((((unsigned long)(n) & 0xFF000000)) >> 24))




__device__ unsigned short d_ntohs(unsigned short n) {
	  return ((n & 0xFF) << 8) | ((n & 0xFF00) >> 8);
}

/**
 * A CUDA kernel to be executed on the GPU.
 * Checks each packet in the array p against a set of firewall rules.
 * Fill the array results with RESULT_FORWARD, RESULT_DROP, or RESULT_UNSET
 */
__global__ void
process_packets(packet *p, int *results, int num_packets, int block_size)
{
	int packet_index = blockIdx.x * block_size + threadIdx.x;

	struct ip *ip_hdr = (struct ip*)p[packet_index].ip;
	
#ifdef BOTH_PAR
if (blockIdx.y == 0)
{
#endif /* BOTH_SEQ */
	
	/**************************************************
	 *                    FIREWALL                    *
	 **************************************************/
	
	uint16_t sport, dport;
	if (ip_hdr->ip_p == 17) {
		struct udphdr *udp_hdr = (struct udphdr*)p[packet_index].udp;
		sport = d_ntohs(udp_hdr->uh_sport);
		dport = d_ntohs(udp_hdr->uh_dport);
	} else if (ip_hdr->ip_p == 6) {
		struct tcphdr *tcp_hdr = (struct tcphdr*)p[packet_index].udp; // FIXME
		sport = d_ntohs(tcp_hdr->th_sport);
		dport = d_ntohs(tcp_hdr->th_dport);
	} else {
		sport = 0;
		dport = 0;
	}

	// TODO: make this handle other protocols (TCP)
	int i; 
	for (i = 0; i < d_num_rules; i++) {
		if ((dd_rules[i].src_ip == 0 || dd_rules[i].src_ip == ip_hdr->ip_src.s_addr) &&
			(dd_rules[i].dst_ip == 0 || dd_rules[i].dst_ip == ip_hdr->ip_dst.s_addr) &&
			(dd_rules[i].src_port == 0 || dd_rules[i].src_port == sport) &&
			(dd_rules[i].dst_port == 0 || dd_rules[i].dst_port == dport) &&
			(dd_rules[i].proto == 0 || dd_rules[i].proto == ip_hdr->ip_p))
		{
			results[packet_index] = dd_rules[i].action;
			break;
		}
		else
		{
			results[packet_index] = RESULT_FORWARD;
		}
	}


#ifdef BOTH_PAR
}
else if (blockIdx.y == 1)
{
#endif /*BOTH_PAR*/


	/**************************************************
	 *             LONGEST PREFIX MATCH               *
	 **************************************************/
    if(packet_index >= num_packets)
        return;
    
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

#ifdef BOTH_PAR
}
#endif /* BOTH_PAR */

}


/**
 * Generates a set of firewall rules
 */
void generate_rules(int num_rules, rule* rules) 
{
	// Generate some random rules
	srand(time(NULL));
	int i;
	float r;
	for (i = 0; i < num_rules; i++) {

		/**
		 *	Pick a protocol:
		 *  * 6%,  TCP 75%,   UDP 14%,   ICMP 4%,   Other 1%
		 */
		r = rand()/(double)RAND_MAX * 100;	 
		if (r <= 75) {
			rules[i].proto = 6; // TCP
		} else if (r <= 89) {
			rules[i].proto = 17; // UDP
		} else if (r <= 93) {
			rules[i].proto = 1; // ICMP
		} else if (r <= 99) {
			rules[i].proto = 0; // *
		} else {
			rules[i].proto = rand() % 255; // Random other
		}

		/**
		 * Pick a source port:
		 *	* 98%,   range 1%,   single value 1%
		 * (note: ignoring ranges for now)
		 */
		 r = rand()/(double)RAND_MAX * 100;
		 if (r <= 98) {
			 rules[i].src_port = 0; // *
		 } else {
			 rules[i].src_port = rand() % 65535; // Random port
		 }
		
		/**
		 * Pick a dest port:
		 *	* 0%,   range 4%,   80 6.89%,   21 5.65%,   23 4.87%,   443 3.90%,   8080 2.25%,   139 2.16%,   Other
		 * (note: ignoring ranges for now)
		 */
		 r = rand()/(double)RAND_MAX * 100;
		 if (r <= 6.89) {
			 rules[i].dst_port = 80;
		 } else if (r <= 12.54) {
			 rules[i].dst_port = 21;
		 } else if (r <= 17.41) {
			 rules[i].dst_port = 23;
		 } else if (r <= 21.31) {
			 rules[i].dst_port = 443;
		 } else if (r <= 23.56) {
			 rules[i].dst_port = 8080;
		 } else if (r <= 25.72) {
			 rules[i].dst_port = 139;
		 } else {
			 rules[i].dst_port = rand() % 65535; // Random port
		 }
		
		/**
		 * Pick a src IP:
		 *	* 95%,   range 5%
		 * (note: ignoring ranges for now)
		 */
		rules[i].src_ip = rand() % 4294967295;
		
		/**
		 * Pick a dst IP:
		 *	single ip 45%,   range 15%,   Class B 10%,   Class C 30%
		 * (note: ignoring ranges for now)
		 */
		r = rand()/(double)RAND_MAX * 100;
		if (r <= 45) {
			rules[i].dst_ip = rand() % 4294967295;
		} else {
			rules[i].dst_ip = 0; // using wildcard instead of range
		}
	
		rules[i].action = RESULT_DROP;
	}

	/*inet_pton(AF_INET, "123.123.123.123", &(rules[0].src_ip));
	inet_pton(AF_INET, "210.210.210.210", &(rules[0].dst_ip));
	rules[0].src_port = 1234;
	rules[0].dst_port = 4321;
	rules[0].proto = 17;
	rules[0].action = RESULT_DROP;*/
	
	/*rules[0].src_ip = 0;
	rules[0].dst_ip = 0;
	rules[0].src_port = 0;
	rules[0].dst_port = 0;
	rules[0].proto = 17;
	rules[0].action = RESULT_DROP;*/

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
 * Firewall-specific setup. This will be called a single time by router.cu 
 * before the kernel function runs for the first time
 */
void setup()
{
	h_num_rules = num_rules;
	int rules_size = h_num_rules * sizeof(rule);
	h_rules = (rule*)malloc(rules_size);
	check_error(cudaMalloc((void **) &d_rules, rules_size), "cudaMalloc d_rules", __LINE__);

	
	generate_rules(h_num_rules, h_rules);


	// Copy firewall rules to GPU so the kernel function can use them
	check_error(cudaMemcpy(d_rules, h_rules, rules_size, cudaMemcpyHostToDevice), "cudaMemcpy (d_rules h_rules)", __LINE__);
	check_error(cudaMemcpyToSymbol(dd_rules, &d_rules, sizeof(d_rules)), "cudaMemcpyToSymbol (dd_rules, d_rules)", __LINE__);
	
	check_error(cudaMemcpyToSymbol(d_num_rules, &h_num_rules, sizeof(int)), "cudaMemcpyToSymbol (d_num_rules, h_num_rules)", __LINE__);
    
	_setup_trie();
    _setup_GPU();
}


/**
 * Firewall-specific teardown. This will be called a single time by router.cu 
 * after the kernel function runs last time
 */
void teardown()
{
	free(h_rules);
	cudaFree(d_rules);
    
	cudaFree(d_serializedtree);
}



/**
 * A CPU-only sequential version of the firewall packet processing function
 */
void process_packets_sequential(packet *p, int *results, int num_packets)
{ 
	/**************************************************
	 *                    FIREWALL                    *
	 **************************************************/
	int packet_index;
	for (packet_index = 0; packet_index < get_batch_size(); packet_index++) {
		struct ip *ip_hdr = (struct ip*)p[packet_index].ip;
		
		uint16_t sport, dport;
		if (ip_hdr->ip_p == 17) {
			struct udphdr *udp_hdr = (struct udphdr*)p[packet_index].udp;
			sport = ntohs(udp_hdr->uh_sport);
			dport = ntohs(udp_hdr->uh_dport);
		} else if (ip_hdr->ip_p == 6) {
			struct tcphdr *tcp_hdr = (struct tcphdr*)p[packet_index].udp; // FIXME
			sport = ntohs(tcp_hdr->th_sport);
			dport = ntohs(tcp_hdr->th_dport);
		} else {
			sport = 0;
			dport = 0;
		}

		// TODO: make this handle other protocols (TCP)
		int i; 
		for (i = 0; i < h_num_rules; i++) {
			if ((h_rules[i].src_ip == 0 || h_rules[i].src_ip == ip_hdr->ip_src.s_addr) &&
				(h_rules[i].dst_ip == 0 || h_rules[i].dst_ip == ip_hdr->ip_dst.s_addr) &&
				(h_rules[i].src_port == 0 || h_rules[i].src_port == sport) &&
				(h_rules[i].dst_port == 0 || h_rules[i].dst_port == dport) &&
				(h_rules[i].proto == 0 || h_rules[i].proto == ip_hdr->ip_p))
			{
				results[packet_index] = h_rules[i].action;
				break;
			}
			else
			{
				results[packet_index] = RESULT_FORWARD;
			}
		}
	}





	/**************************************************
	 *           LONGEST PREFIX MATCH                 *
	 **************************************************/

    packet_index = 0;
    
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
 * Firewall-specific setup. This will be called a single time by router.cu
 * before the sequential CPU function runs for the first time
 */
void setup_sequential()
{
	h_num_rules = num_rules;
	int rules_size = h_num_rules * sizeof(rule);
	h_rules = (rule*)malloc(rules_size);

	
	generate_rules(h_num_rules, h_rules);
    
	_setup_trie();
}

#endif /* defined (BOTH_SEQ) || defined (BOTH_PAR) */
