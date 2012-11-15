all: cplusplus

tree: cplusplus \
	cuda-lookup

cplusplus: cuda-lookup
	g++ -g -m64 -L/usr/local/cuda/lib -lcuda -lcudart -o tree tree.c mymalloc.c cuda-lookup.o

debug: cuda-lookup
	#nvcc -c cuPrintf.cu -m64
	#g++ -m64 -L/usr/local/cuda/lib -lcuda -lcudart -o tree tree.c mymalloc.c cuda-lookup.o cuPrintf.o
	g++ -m64 -L/usr/local/cuda/lib -lcuda -lcudart -o tree tree.c mymalloc.c cuda-lookup.o


cuda-lookup: 
	nvcc -c cuda-lookup.cu -m64

clean: 
	rm -rf *.o tree
