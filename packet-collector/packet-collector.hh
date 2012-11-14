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
#define DEFAULT_BATCH_SIZE 1024
// in milliseconds
#define DEFAULT_BATCH_WAIT 100

#define RESULT_ERROR -1
#define RESULT_DROP -2
#define RESULT_FORWARD -3
#define RESULT_UNSET -4


typedef struct _packet {
  int size;
  char buf[BUF_SIZE];
} packet;

int get_packets(int sockfd, packet* p);
int init_socket();
int set_batch_size(int s);
int get_batch_size();
int set_batch_wait(int s);
int get_batch_wait();

#endif /* PACKET_COLLECTOR_H */
