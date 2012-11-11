#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

#define PORT 9876  
#define BUF_SIZE 4096
#define MAX_ENTRIES 1024

// in miliseconds
#define TIMEOUT 5000

typedef struct _packet {
  int size;
  char buf[BUF_SIZE];
} packet;

typedef struct _timeout {
  pthread_mutex_t lock;
  int timeout;
} timeout;

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
  
  timeout *t = (timeout *)data;
  pthread_mutex_lock(&t->lock);
  t->timeout = 1;
  pthread_mutex_unlock(&t->lock);
}

int get_packets(int sockfd, packet* p) {
  timeout t;
  pthread_t thread;

  t.timeout = 0;
  pthread_mutex_init(&t.lock, NULL);

  struct timeval tout;
  tout.tv_sec = 0;
  tout.tv_usec = 0;

  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(sockfd, &rfds);

  pthread_create(&thread, NULL, timer, &t);
 
  int i;
  for(i = 0; i < MAX_ENTRIES;) {
    if(select(sockfd+1, &rfds, NULL, NULL, &tout)) {
      p[i].size = recv(sockfd, &p[i].buf, BUF_SIZE, 0); 
      i++;
    }
    pthread_mutex_lock(&t.lock);
    if(t.timeout == 1) break;
    pthread_mutex_unlock(&t.lock);
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
