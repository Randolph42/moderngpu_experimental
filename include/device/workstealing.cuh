#pragma once

#include "deviceutil.cuh"

struct WorkStealingGrid {
	struct Storage {
		int counter;
	};

	// Dynamically 'steals' work 
	MGPU_DEVICE static int ProcessTile(int tid, int* gridNext_global, 
		Storage& storage) {

		if(!tid) {
			// The first thread atomically adds.
			int next = atomicSub(&gridNext_global, 1);
			storage.counter = next - 1;
		}
		__syncthreads();

		int next = storage.counter;
		__syncthreads();

		return next;
	}
};

template<bool DynamicScheduling>
struct FlexGrid {

	

};
