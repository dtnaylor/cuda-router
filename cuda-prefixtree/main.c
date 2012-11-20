ca/**
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
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <arpa/inet.h>

#include "mymalloc.h"
#include "cuda-lookup.cuh"

#ifdef USEMAINDEBUG
int maindebug = 1;
#else
int maindebug = 0;
#endif

#define ILUNARRAYSIZE 11000
struct iplookup_node *ilun_array;


/*#define DEBUG(fmt, ...) do { if (debug) fprintf(stderr, fmt, __VA_ARGS__); } while (0)*/
#define DEBUG(...) do { if (maindebug) fprintf(stdout, __VA_ARGS__); } while (0)


/*
 * Educate the user via standard error.
 */
void print_usage(char* name)
{
	fprintf(stderr, "%s will replace any IPv4 address on standard input with the\n", name);
	fprintf(stderr, "\tmatching prefix in prefix_file, or 'NF'.\n");
	fprintf(stderr, "Usage: %s -f prefix_file [-d]\n\n", name);
}


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
			maindebug = 1;
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
