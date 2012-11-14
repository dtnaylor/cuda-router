#include "packet-collector.hh"

int batch_size = DEFAULT_BATCH_SIZE;
int batch_wait = DEFAULT_BATCH_WAIT;

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

int set_batch_size(int s) {
	if (s > 0) {
		batch_size = s;
	}
	return batch_size;
}

int get_batch_size() {
	return batch_size;
}

int set_batch_wait(int s) {
	if (s > 0) {
		batch_wait = s;
	}
	return batch_wait;
}

int get_batch_wait() {
	return batch_wait;
}

void *timer(void *data) {
  usleep(batch_wait*1000);  
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
  for(i = 0; i < batch_size;) {
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

#ifndef CUDA_CODE
int main() {  
  int sockfd = init_socket();
  if(sockfd == -1) {
    return -1;
  }

  packet* p = (packet *)malloc(sizeof(packet)*batch_size);
  
  while(1) {
    int num_packets = get_packets(sockfd, p);
    printf("i = %d\n", num_packets);
  }
}
#endif
