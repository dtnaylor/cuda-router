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
#define TIMEOUT 5000

typedef struct _packet {
  int size;
  char buf[BUF_SIZE];
} packet;

int init_socket() {
  int sockfd;
  struct sockaddr_in servaddr;

  if ((sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
	perror("socket");
	return -1;
  }

  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family      = AF_INET;
  servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  servaddr.sin_port        = htons(PORT);

  if (bind(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0) {
	perror("bind");
	return -1;
  }

  return sockfd;
}

void *timer(void *data) {
  usleep(TIMEOUT*1000);  
  *(int *)data = 1;
  return 0;
}

int get_packets(int sockfd, packet* p) {
  int timeout = 0;
  pthread_t thread;

  struct timeval tout;
  tout.tv_sec = 0;
  tout.tv_usec = 0;

  fd_set rfds;
  FD_ZERO(&rfds);

  pthread_create(&thread, NULL, timer, &timeout);
 
  int i;
  for(i = 0; i < MAX_ENTRIES;) {
    FD_SET(sockfd, &rfds);
    if(select(sockfd+1, &rfds, NULL, NULL, &tout) > 0) {
      p[i].size = recv(sockfd, &p[i].buf, BUF_SIZE, 0); 
      i++;
    }
    if(timeout == 1) break;
  }
  pthread_cancel(thread);
  return i;
}

int main() {  
  int sockfd = init_socket();
  if(sockfd == -1) {
    return -1;
  }

  packet* p = (packet *)malloc(sizeof(packet)*MAX_ENTRIES);
  
  while(1) {
    int num_packets = get_packets(sockfd, p);
    printf("i = %d\n", num_packets);
  }
}
