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

#include "tree.h"  

#include <stdio.h>
//#include <stdint.h>
#include <stdlib.h>
//#include <unistd.h>
#include <getopt.h>

#include <math.h>
#include <string.h>
//#include <fcntl.h>

#include <arpa/inet.h>
//#include <sys/types.h>
//#include <sys/stat.h>

#include "mymalloc.h"
#include "cuda-lookup.cuh"

#ifdef USEDEBUG
int debug = 1;
#else
int debug = 0;
#endif

#define ILUNARRAYSIZE 11000
struct iplookup_node *ilun_array;


/*#define DEBUG(fmt, ...) do { if (debug) fprintf(stderr, fmt, __VA_ARGS__); } while (0)*/
#define DEBUG(...) do { if (debug) fprintf(stdout, __VA_ARGS__); } while (0)

/*
 * This creates an data (value-holding) node which points nowhere.
 */
struct data_node* create_data_node(uint32_t prefix, uint8_t netmask, uint16_t port)
{
    if(!try_myserializedmalloc(sizeof(struct data_node))) // serialization array probably doubled but pointers probably changed, so nested function must restart
        return NULL;
    
	struct data_node* node = (struct data_node*)myserializedmalloc(sizeof(struct data_node));
	DEBUG("## Created data node %p for %d\n", (void*)node, prefix);
	node->type  = DAT_NODE;
	node->prefix = prefix;
	node->netmask  = netmask;
    node->port = port;
	//node->l = NULL;
	node->l_offset = 0;
	//node->r = NULL;
	node->r_offset = 0;
	return node;
}

/*
 * This creates an internal node that points nowhere.
 */
struct internal_node* create_internal_node()
{
    if(!try_myserializedmalloc(sizeof(struct internal_node))) // serialization array probably doubled but pointers probably changed, so nested function must restart
        return NULL;
    
	struct internal_node* tmp = (struct internal_node*)myserializedmalloc(sizeof(struct internal_node));
	DEBUG("## Created internal node %p\n", (void*)tmp);
	tmp->type = INT_NODE;
	//tmp->l = NULL;
	tmp->l_offset = 0;
	//tmp->r = NULL;
	tmp->r_offset = 0;

	return tmp;
}

/*
 * This function used internally; see lpm_insert().
 */
int insert(uint32_t prefix, uint32_t nm, uint16_t port, struct internal_node* n)
{
	uint8_t b = MAX_BITS;
	uint8_t depth = 0;
	struct internal_node* parent, *l,*r;
	struct internal_node* next = n;

    DEBUG("Inserting prefix %d\n", prefix);
    
	/* First, find the correct location for the prefix. Burrow down to
	   the correct depth, potentially creating internal nodes as I
	   go. */
	do {
		n = next;
		b--;
		depth++;

		parent = (struct internal_node*)n;
		//uint32_t v_bit = prefix & ((uint32_t)pow(2, b));
        uint32_t v_bit = prefix & ((uint32_t) 1 << b);


		/* Determine which direction to descend. */
		if (v_bit) {
			//if (n->r == NULL) {
			if (n->r_offset == 0) {
				//n->r = create_internal_node();
				r = create_internal_node();
                if(r == NULL) return 0; // interrupt immediately and perhaps restart later
				n->r_offset = ((char*)r - (char*)n);
                
                next = (struct internal_node*)((char*)n + n->r_offset);
                if(next!=r) printf("ERROR: BAD POINTERS! (n %p r %p off %d)\n", n, r, n->r_offset);
			}
			//next = n->r;
			next = (struct internal_node*)((char*)n + n->r_offset);
		}
		else {
			//if (n->l == NULL) {
			if (n->l_offset == 0) {
				l = create_internal_node();
                if(l == NULL) return 0; // interrupt immediately and perhaps restart later
				n->l_offset = ((char*)l - (char*)n);
                
                next = (struct internal_node*)((char*)n + n->l_offset);
                if(next!=l) printf("ERROR: BAD POINTERS (n %p l %p off %d)\n", n, r, n->l_offset);			}
			//next = n->l;
			next = (struct internal_node*)((char*)n + n->l_offset);
		}
        //DEBUG("n=%p, next=%p\n", (void*)n, (void*)next);
	} while (depth < nm);

	if (next == NULL) {
		/* The easy case. */
		struct data_node* node = create_data_node(prefix, nm, port);
        if(node == NULL) return 0; // interrupt immediately and perhaps restart later
		//uint32_t v_bit = prefix & ((uint32_t)pow(2, b));
		uint32_t v_bit = prefix & ((uint32_t) 1 << b);
		if (v_bit) {
			//parent->r = (struct internal_node*)node;
			parent->r_offset = ((char*)node - (char*)parent);
		}
		else {
			//parent->l = (struct internal_node*)node;
			parent->l_offset = ((char*)node - (char*)parent);
		}
	}
	else if (next->type == INT_NODE) {
		/* In this case, we've descended as far as we can. Attach the
		   prefix here. */
		//uint32_t v_bit = prefix & ((uint32_t)pow(2, b));
		uint32_t v_bit = prefix & ((uint32_t) 1 << b);
		struct data_node* newnode = create_data_node(prefix, nm, port);
        if(newnode == NULL) return 0; // interrupt immediately and perhaps restart later

		//newnode->l = next->l;
		newnode->l_offset = ((char*)next - (char*)newnode) + next->l_offset;
		//newnode->r = next->r;
		newnode->r_offset = ((char*)next - (char*)newnode) + next->r_offset;

		if (v_bit) {
			//n->r = (struct internal_node*)newnode;
			n->r_offset = ((char*)newnode - (char*)n);
		}
		else {
			//n->l = (struct internal_node*)newnode;
			n->l_offset = ((char*)newnode - (char*)n);
		}

		DEBUG("## Freeing %p\n", (void*)n);
		//myfree(next);
	}
    
    return 1;
}

/* destroy:
 * Recursively destroys nodes.
 */
void destroy(struct internal_node* node)
{
	if (node == NULL) return;

	//if (node->l != NULL) {
	if (node->l_offset != 0) {
		//destroy(node->l);
		destroy((struct internal_node*)((char*)node + node->l_offset));
	}
	//if (node->r != NULL) {
	if (node->r_offset != 0) {
		//destroy(node->r);
		destroy((struct internal_node*)((char*)node + node->r_offset));
	}

	myserializedfree(node);
}


/* lpm_destroy:
 * Frees the entire tree structure.
 */
void lpm_destroy(struct lpm_tree* tree)
{
	if (tree == NULL) return;
	destroy((struct internal_node*)((char*)tree + tree->h_offset));
	myserializedfree(tree);
}


/* lpm_init:
 * Constructs a fresh tree ready for use by the other functions.
 */
struct lpm_tree* lpm_init()
{
	struct lpm_tree* tree = (struct lpm_tree*)myserializedmalloc(sizeof(struct lpm_tree));
	DEBUG("## Created tree %p\n", (void*)tree);

	/* Build empty internal node, and attach it to new tree. */
	struct internal_node* node = create_internal_node();
    
	//tree->head = node;
	tree->h_offset = ((char*)node - (char*)tree);
	return tree;
}

/* lpm_insert:
 * Insert a new prefix ('ip_string' and 'netmask') into the tree. If
 * 'ip_string' does not contain a valid IPv4 address, or the netmask
 * is clearly invalid, the tree is not modified and the function
 * returns 0. Successful insertion returns 1.
 */
int lpm_insert(struct lpm_tree* tree, char* ip_string, uint32_t netmask, uint16_t port)
{
	uint32_t ip;
    int ret;
	if (!inet_pton(AF_INET, ip_string, &ip) || netmask > 32) {
		return 0;
	}
	ip = ntohl(ip);

	DEBUG(">> Inserting %s/%d===================================================\n", ip_string, netmask);

    //insert(ip, netmask, tree->head);
    ret = insert(ip, netmask, port, (struct internal_node*)((char*)tree + tree->h_offset));
 
	if(ret)
        DEBUG(">> Done inserting %s/%d =============================================\n", ip_string, netmask);
    
	return ret;
}

/*
 * Internal function; called by lpm_lookup()
 */
void lookup(struct iplookup_node *ilun, uint32_t address, char* output, struct internal_node* n)
{
	uint32_t b = MAX_BITS;
	struct internal_node* parent;
	struct internal_node* next = n;
    uint16_t iterations=0;

	uint32_t best_prefix = 0;
	uint8_t  best_netmask = 0;


	char addr_string[16];
	uint32_t tmp = htonl(address);
	inet_ntop(AF_INET, &tmp, addr_string, 16);
    
#ifdef USECUPRINTF
    printf("root is:%p, loffset %d roffset %d\n", next, next->l_offset, next->r_offset);
#endif

	do {
		n = next;
		b--;
        iterations++;
        
		parent = (struct internal_node*)n;
		//uint32_t v_bit = address & ((uint32_t)pow(2, b));
        uint32_t v_bit = address & ((uint32_t)1 << b);

#ifdef USECUPRINTF
        //printf("v_bit %u \t bits %u \t pow2 %u \t mypow %u\n", v_bit, b, (uint32_t)pow((double)2, (double)b), 1<<b);
#endif
		/* If we've found an internal node, determine which
		   direction to descend. */
		if (v_bit) {
			//next = n->r;
			next = (struct internal_node*)((char*)n + n->r_offset);
#ifdef USECUPRINTF
            printf("next right is:%p loffset %d roffset %d\n", next, next->l_offset, next->r_offset);
            
#endif
		}
		else {
			//next = n->l;
			next = (struct internal_node*)((char*)n + n->l_offset);
#ifdef USECUPRINTF
            printf("next left is:%p loffset %d roffset %d\n", next, next->l_offset, next->r_offset);
            
#endif
		}

		if (n->type == DAT_NODE) {
			struct data_node* node = (struct data_node*)n;

#ifdef USECUPRINTF
            printf("data node found , iter %d\n", iterations);
#endif
            
			char prefix[16];
			tmp = htonl(node->prefix);
			inet_ntop(AF_INET, &tmp, prefix, 16);

			uint32_t mask = 0xFFFFFFFF;

			//mask = mask - ((uint32_t)pow(2, 32 - node->netmask) - 1);
            mask = mask - (((uint32_t)1 << (32 - node->netmask)) - 1);


			if ((address & mask) == node->prefix) {
				best_prefix = node->prefix;
				best_netmask = node->netmask;
                ilun->port = node->port;
			}
			else {
				break;
			}
		}
    } while (next != n); //termination when offset is 0 and they are equal
	//} while (next != NULL);

    ilun->port2 = iterations;
    
	if (!best_prefix) {
		sprintf(output, "NF");
	}
	else {
		char prefix[16];
		tmp = htonl(best_prefix);
		inet_ntop(AF_INET, &tmp, prefix, 16);

		sprintf(output, "%s/%d", prefix, best_netmask);
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
int lpm_lookup(struct lpm_tree* tree, struct iplookup_node *ilun, char* output)
{
    /*
	uint32_t tmp;
	int rt;
	rt = inet_pton(AF_INET, ip_string, &tmp);
	if (!rt) {
		return 0;
	}
	uint32_t ip = ntohl(tmp);
     */
	//lookup(ip, output, tree->head);
	lookup(ilun, ilun->ip, output, (struct internal_node*)((char*)tree + tree->h_offset));

	return 1;
}




/* debug_print:
 * Prints out the current node's parent, the current node's value (if
 * it has one), and recurses down to the left then right children. The
 * 'left' parameter should indicate the direction of the hop to the
 * current node (1 if the left child was used, 0 if the right child
 * was used; -1 is used to indicate the root of the tree.)
 */
void debug_print(struct internal_node* parent, int left, int depth, struct internal_node* n)
{
	DEBUG("parent:%p", (void*)parent);
	if (left == 1) {
		DEBUG("->L");
	}
	else if (left == 0) {
		DEBUG("->R");
	}
	else {
		DEBUG("---");
	}

	//if (n == NULL) {
    if(parent == n) { // this happens when offset is 0
		DEBUG(" Reached a null bottom %p\n", (void*)n);
		return;
	}
	else if (n->type == INT_NODE) {
		DEBUG(" Internal node %p.\n", (void*)n);
	} 
	else {
		struct data_node* node = (struct data_node*)n;

		uint32_t tmp = htonl(node->prefix);
		char output[16];
		inet_ntop(AF_INET, &tmp, output, 16);

		DEBUG(" External node: %p, %s/%d, %d\n", (void*)n, output, node->netmask, node->port);
	}

    DEBUG("debugging n %p, offsets %d %d\n", (void*)n, n->l_offset, n->r_offset);

	debug_print(n, 1, depth+1, (struct internal_node*)((char*)n + n->l_offset));

	debug_print(n, 0, depth+1, (struct internal_node*)((char*)n + n->r_offset));

}

/* lpm_debug_print:
 * Traverses the tree and prints out node status, starting from the root.
 */
void lpm_debug_print(struct lpm_tree* tree)
{
	if (debug) {
		//debug_print((struct internal_node*)tree, -1, 0, tree->head);
        DEBUG("debugging tree %p, offset %d, head %p\n", (void*)tree, tree->h_offset, (void*)((char*)tree + tree->h_offset));
		debug_print((struct internal_node*)tree, -1, 0, (struct internal_node*)((char*)tree + tree->h_offset));
	}
}

/*
 * Educate the user via standard error.
 */
void print_usage(char* name)
{
	fprintf(stderr, "%s will replace any IPv4 address on standard input with the\n", name);
	fprintf(stderr, "\tmatching prefix in prefix_file, or 'NF'.\n");
	fprintf(stderr, "Usage: %s -f prefix_file [-d]\n\n", name);
}


void build_serializedtree(char *filename) {
    struct lpm_tree* tree;
    //struct lpm_serializedtree sTree;
    int insertions, ret;
    
    FILE* in;

    DEBUG("Create tree (size of pointer=%d, sizeof char=%d)\n", (int)sizeof(tree), (int)sizeof(char));
	/* Create a fresh tree. */
	tree = lpm_init();

    /* Read in all prefixes. */
	in = fopen(filename, "r");
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
    
}

#ifndef LPM_TRIE || FIREWALL

int main(int argc, char* argv[])
{
	char* input;
	int opt,i;
	struct lpm_tree* tree;
    struct lpm_serializedtree sTree;
	char* ifile = (char*) "/dev/stdin";
    FILE* in;

	/* Check inputs; print usage and exit if something is clearly
	   wrong. */
	if (argc < 3) {
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}
    
    DEBUG("Parsing options\n");
	/* Parse options */
	while ((opt = getopt(argc, argv, "df:")) != -1) {
		switch (opt) {
		case 'd':
			debug = 1;
			break;
		case 'f':
			input = optarg;
			break;
		default: /* '?' */
			print_usage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}

    DEBUG("Init serializer\n");
    
	init_myserializer((void**)&tree);
    
//    build_serializedtree(input);
    
    
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

    
    //lpm_debug_print((struct lpm_tree*)sTree.serialized_tree);
    
    printf("Tree serialized and checked, start lookup (ILUNARRAYSIZE %d)!\n", ILUNARRAYSIZE);
    
    ilun_array = (struct iplookup_node*)malloc(ILUNARRAYSIZE*sizeof(struct iplookup_node));
    
	/* Begin reading from standard input the lines of text to
	   convert. */
	in = fopen(ifile, "r");
    i=0;
    char output[16];

	while (1) {
		char line[4096];
		char* rtp;
		char address_string[16];
		char* pointer;
		char* strstart;
		char* strend;
		int rt;
        uint32_t tmp;
        struct iplookup_node ilun;

		/* Read line. */
		memset(line, '\0', 4096);
        DEBUG("LINEbef: line %p in %p\n", line, in);

		rtp = fgets(line, 4096, in);
		if (rtp == NULL) {
			break;
		}

		line[strlen(line)-1] = '\0';
        DEBUG("LINE: %s\n", line);
        
		pointer = line;
		strstart = pointer;
		strend = strstr(strstart, " ");

        if(!inet_pton(AF_INET, strstart, &tmp)) {
            ilun.port=0;
            rt = 0;
        }
        else {
            ilun.ip = ntohl(tmp);
            ilun.port = 0;
            ilun_array[i++] = ilun;
            rt = 1;
        }
        
		while (strend != NULL) {
            DEBUG("entering inner while\n");
			memset(address_string, '\0', 16);
			memcpy(address_string, strstart, strend - strstart);
            memset(output,         '\0', 16);

            if(!inet_pton(AF_INET, address_string, &tmp)) {
                ilun.port=0;
                rt = 0;
            }
            else {
                ilun.ip = ntohl(tmp);
                ilun.port = 0;
                ilun_array[i++] = ilun;

                rt = lpm_lookup((struct lpm_tree*)sTree.serialized_tree, &ilun, output);
            }
			if (rt) {
				printf("%s , port %d, port2 %d\n", output, ilun.port, ilun.port2);
			}
			else {
				printf("%s ", address_string);
			}

			strstart = strend + 1;
			strend = strstr(strstart, " ");
		}

		memset(output, '\0', 16);

        rt = lpm_lookup((struct lpm_tree*)sTree.serialized_tree, &ilun, output);

		if (rt) {
			printf("%s , port %d, port2 %d\n", output, ilun.port, ilun.port2);
		}
		else {
			printf("%s\n", strstart);
		}
	}
    
    
    int j = i;
    //fill array
    while(i!=ILUNARRAYSIZE) {
        ilun_array[i] = ilun_array[i%j];
        i++;
    }
    
    //lpm_lookup((struct lpm_tree*)tree->serialized_tree, &(ilun_array[0]), output);
    lpm_lookup((struct lpm_tree*)sTree.serialized_tree, &(ilun_array[0]), output);
    printf("custom lookup %s , port %d\n", output, ilun_array[0].port);


    DEBUG("go CUDA\n");

    go_cuda(sTree.serialized_tree, sTree.serializedtree_size , ilun_array, ILUNARRAYSIZE);
    
    DEBUG("end of CUDA, results:\n");
    
    //tree->serialized_tree = p;
    //sTree.serialized_tree = p;
    
    for(i=0; i<ILUNARRAYSIZE;i++) {
        printf("ip:%d to be routed to port %d, port2=%d\n", ilun_array[i].ip, ilun_array[i].port, ilun_array[i].port2);
    }

	//lpm_destroy(tree);

	fclose(in);
	return 1;
}

#endif
