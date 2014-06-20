
include ../common.mk

all: build

build: benchmarksort

benchmarksort.o: benchmarksort.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

benchmarksort: benchmarksort.o 
	make -C ../lib -f Makefile
	nvcc $(NVCCFLAGS) -lmgpu -o $@ $<

run: build
	./benchmarksort

clean:
	rm -f benchmarksort.o benchmarksort

clobber:
	clean

