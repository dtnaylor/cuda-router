/*
 * mymalloc.h
 *
 *  Created on: Nov 13, 2012
 *      Author: br1
 */

#ifndef MYMALLOC_H_
#define MYMALLOC_H_


void* mymalloc(size_t size);
void myfree(void* p);
unsigned int getallocatedbytes();

void init_myserializer(void** p);
void uninit_myserializer();

void* myserializedmalloc(size_t size);
void myserializedfree(void *p); //just returns
size_t finalize_serialized(void **p);
int try_myserializedmalloc(size_t size);
int cannotdouble_myserializer();


typedef struct {
	char* buffer;
	size_t size;
	size_t free_bytes;
	char* next_available_byte;
    int cannotdouble;
} serializer_t;

#endif /* MYMALLOC_H_ */
