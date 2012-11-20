#include <stdio.h>
int main() {
	unsigned int n=1;
	n = n<<20;
	printf("n %u\n",n);
	n = 1<<31;
	printf("n %u\n",n);
}
