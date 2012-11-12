#ifndef PACKET_COLLECTOR_H
#define PACKET_COLLECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

#define PORT 9877
#define BUF_SIZE 4096
#define MAX_ENTRIES 1024

// in miliseconds
#define TIMEOUT 100

typedef struct _packet {
  int size;
  char buf[BUF_SIZE];
} packet;

int get_packets(int sockfd, packet* p);
int init_socket();

#endif /* PACKET_COLLECTOR_H */
