/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#pragma once

#include "../device/intrinsics.cuh"

namespace mgpu {

// Get the difference between two pointers in bytes.
MGPU_HOST_DEVICE ptrdiff_t PtrDiff(const void* a, const void* b) {
	return (const byte*)b - (const byte*)a;
}

// Offset a pointer by i bytes.
template<typename T> 
MGPU_HOST_DEVICE const T* PtrOffset(const T* p, ptrdiff_t i) {
	return (const T*)((const byte*)p + i);
}
template<typename T>
MGPU_HOST_DEVICE T* PtrOffset(T* p, ptrdiff_t i) {
	return (T*)((byte*)p + i);
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, bool DeviceMem>
struct ParamAccessor {
	T val;
	MGPU_HOST_DEVICE ParamAccessor(T x) : val(x) { }
	MGPU_HOST_DEVICE ParamAccessor(const ParamAccessor& rhs) : val(rhs.val) { }

	MGPU_HOST_DEVICE T Access() const { return val; }
};
template<typename T>
struct ParamAccessor<T, true> {
	const T* p_global;
	MGPU_HOST_DEVICE ParamAccessor(const T* x_global) : p_global(x_global) { }
	MGPU_HOST_DEVICE ParamAccessor(const ParamAccessor& rhs) :
		p_global(rhs.p_global) { }

	MGPU_DEVICE T Access() { return *p_global; }
};

template<typename T>
ParamAccessor<T, false> HostAccessor(T x) {
	return ParamAccessor<T, false>(x);
}
template<typename T>
ParamAccessor<T, true> DeviceAccessor(T* p) {
	return ParamAccessor<T, true>(p);
}

// Function for calling ParamAccessor::Access.
template<typename T, bool DeviceMem>
MGPU_DEVICE T ParamAccess(ParamAccessor<T, DeviceMem> x) {
	return x.Access();
}

// Unencapsulated type.
template<typename T>
struct ParamType {
	typedef T Type;
};

// Host-side encapsulated type.
template<typename T>
struct ParamType<ParamAccessor<T, false> > {
	typedef T Type;
};

// Device-side encapsulated type.
template<typename T>
struct ParamType<ParamAccessor<T, true> > {
	typedef T Type;
};

// Function for interface consistency with ParamAccessor but for unwrapped 
// types.
template<typename T>
MGPU_DEVICE T ParamAccess(T x) {
	return x;
}

// WorkDistribution assigns the next tile of work to the next CTA that waits on
// NextTile(). The number of tiles is typically computed inside the dynamic 
// kernel from arguments passed in through device memory.
// Counters are provided by MgpuContext::GetCounter, which maintains a large
// array of allocated and zero counters.
struct WorkDistribution {
	struct Storage {
		int counter;
	};

	// Dynamically pulls the next work tile for this CTA.
	// Note that you must initialize the tile reference to -1 on entry.
	// The first tile for each CTA is executed immediately without accessing
	// the shared counter.
	MGPU_DEVICE static bool WorkStealing(int tid, int numTiles, 
		int* counter_global, Storage& storage, int& tile) {

		if(!tid) {
			// The first thread atomically adds.
			int next = atomicAdd(counter_global, 1);
			storage.counter = next;
		}
		__syncthreads();

		tile = storage.counter;
		__syncthreads();

		return tile < numTiles;
	}

	MGPU_DEVICE static bool EvenShare(int tid, int numTiles,
		int* counter_global, Storage& storage, int& tile) {

		if(-1 == tile)
			tile = blockIdx.x;
		else
			tile += gridDim.x;

		return tile < numTiles;
	}
};




////////////////////////////////////////////////////////////////////////////////
// Task range support
// Evenly distributes variable-length arrays over a fixed number of CTAs.

MGPU_HOST int2 DivideTaskRange(int numItems, int numWorkers) {
	div_t d = div(numItems, numWorkers);
	return make_int2(d.quot, d.rem);
}

MGPU_HOST_DEVICE int2 ComputeTaskRange(int block, int2 task) {
	int2 range;
	range.x = task.x * block;
	range.x += min(block, task.y);
	range.y = range.x + task.x + (block < task.y);
	return range;
}

MGPU_HOST_DEVICE int2 ComputeTaskRange(int block, int2 task, int blockSize, 
	int count) {
	int2 range = ComputeTaskRange(block, task);
	range.x *= blockSize;
	range.y = min(count, range.y * blockSize);
	return range;
}

////////////////////////////////////////////////////////////////////////////////
// DeviceExtractHeadFlags
// Input array flags is a bit array with 32 head flags per word.
// ExtractThreadHeadFlags returns numBits flags starting at bit index.

MGPU_HOST_DEVICE uint DeviceExtractHeadFlags(const uint* flags, int index, 
	int numBits) {

	int index2 = index>> 5;
	int shift = 31 & index;
	uint headFlags = flags[index2]>> shift;
	int shifted = 32 - shift;

	if(shifted < numBits)
		// We also need to shift in the next set of bits.
		headFlags = bfi(flags[index2 + 1], headFlags, shifted, shift);
	headFlags &= (1<< numBits) - 1;
	return headFlags;
}

////////////////////////////////////////////////////////////////////////////////
// DevicePackHeadFlags
// Pack VT bits per thread at 32 bits/thread. Will consume an integer number of
// words, because CTA size is a multiple of 32. The first NT * VT / 32 threads
// return packed words.

template<int NT, int VT>
MGPU_DEVICE uint DevicePackHeadFlags(uint threadBits, int tid, 
	uint* flags_shared) {

	const int WordCount = NT * VT / 32;

	// Each thread stores its thread bits to flags_shared[tid].
	flags_shared[tid] = threadBits;
	__syncthreads();

	uint packed = 0;
	if(tid < WordCount) {
		const int Items = MGPU_DIV_UP(32, VT);
		int index = 32 * tid;
		int first = index / VT;
		int bit = 0;

		int rem = index - VT * first;
		packed = flags_shared[first]>> rem;
		bit = VT - rem;
		++first;
		
		#pragma unroll
		for(int i = 0; i < Items; ++i) {
			if(i < Items - 1 || bit < 32) {
				uint x = flags_shared[first + i];
				if(bit < 32) packed |= x<< bit;
				bit += VT;
			}
		}
	}
	__syncthreads();

	return packed;
}

} // namespace mgpu
