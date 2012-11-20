#include <stdio.h>

int myrand() {
	int ret;
	while((ret=rand()%255)==0);
	return ret;
}

int main(int argc, char **argv) {
	int n = atoi(argv[1]);
	int i;
	for(i=0; i<n; i++) {
		printf("%d.%d.%d.%d\n", myrand(), myrand(), myrand(), myrand());
	}
}
