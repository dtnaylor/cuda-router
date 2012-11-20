/*
 * mymalloc.c
 *
 *  Created on: Nov 13, 2012
 *      Author: br1
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include "mymalloc.h"

#ifdef USEDEBUG
int mallocdebug = 1;
#else
int mallocdebug = 0;
#endif

#define DEBUG(...) do { if (mallocdebug) fprintf(stdout, __VA_ARGS__); } while (0)


#define INITIAL_BUFFER_SIZE 1024

unsigned int allocatedbytes=0;
serializer_t myserializer;
void **init_p;

void* mymalloc(size_t size) {
	void* p;
	p = malloc(size);
	if(p != NULL) allocatedbytes += size;
	return p;
}

void myfree(void* p) {
	if(p != NULL)
		free(p);
}

unsigned int getallocatedbytes(){
	return allocatedbytes;
}

void init_myserializer(void** p) {
    init_p = p;
	myserializer.buffer = (char*)malloc(INITIAL_BUFFER_SIZE*sizeof(char));
	if(myserializer.buffer==NULL) printf("ERROR: impossible to init array\n");
	myserializer.size = INITIAL_BUFFER_SIZE;
	myserializer.free_bytes = INITIAL_BUFFER_SIZE;
	myserializer.next_available_byte = myserializer.buffer;
    myserializer.cannotdouble=0;
    DEBUG("## myserialized inizialized, %d free bytes\n", (int)myserializer.free_bytes);
}

void uninit_myserializer() {
    free(myserializer.buffer);
    myserializer.size = 0;
    myserializer.free_bytes = 0;
    myserializer.next_available_byte = NULL;
    myserializer.cannotdouble=0;
}

void _doubleserializersize() {
	char *p;
	uint32_t next_available_offset = myserializer.next_available_byte - myserializer.buffer;
	p = (char*)realloc(myserializer.buffer, 2*myserializer.size);
	if(p==NULL) {
        printf("ERROR: impossible to double array\n");
        myserializer.cannotdouble = 1;
    }
    
    *init_p = p;
	
    myserializer.buffer = p;
	myserializer.free_bytes += myserializer.size;
	myserializer.size *= 2;
	myserializer.next_available_byte = myserializer.buffer + next_available_offset;
    DEBUG("## myserialized reallocated, %d free bytes\n", (int)myserializer.free_bytes);
}

int cannotdouble_myserializer() {
    return myserializer.cannotdouble;
}

void* myserializedmalloc(size_t size) {
	char *ret;
	while(size > myserializer.free_bytes)
		_doubleserializersize();

	ret = myserializer.next_available_byte;
	myserializer.next_available_byte += size;
	myserializer.free_bytes -= size;
    DEBUG("## myserialized allocation of %d bytes succeeded, %d free bytes, %p returned, %p next available\n", (int)size, (int)myserializer.free_bytes, ret, myserializer.next_available_byte);

	return ret;
}

int try_myserializedmalloc(size_t size) {
	if(size > myserializer.free_bytes) { // if not enough, it must make the nested functions fail and restart becuase the pointer may change
		_doubleserializersize();
        return 0;
    }
    
    return 1;
}

void myserializedfree(void *p) {
    p=p; //do nothing
    return;
}


size_t finalize_serialized(void **p) {
	//shrink array to the size effectively used
	*p = realloc(myserializer.buffer, myserializer.size - myserializer.free_bytes);
	if(*p==NULL) printf("ERROR: impossible to shrink array\n");

	myserializer.buffer = (char*)*p;
	myserializer.size -= myserializer.free_bytes;
	myserializer.free_bytes = 0;
	myserializer.next_available_byte = NULL;
    DEBUG("## myserialized finalized, %d bytes total, %d free bytes, %p returned, %p next available\n", (int)myserializer.size, (int)myserializer.free_bytes, *p, myserializer.next_available_byte);

	return myserializer.size;
}
