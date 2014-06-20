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

#include "../mgpuhost.cuh"
#include "../device/ctamerge.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelBinarySearch

enum { BINARY_SEARCH_CTA_SIZE = 64 };

MGPU_HOST_DEVICE int NumBinarySearchPartitions(int count, int nv) {
	return MGPU_DIV_UP(count, nv);
}
MGPU_HOST_DEVICE int NumBinarySearchBlocks(int numPartitions) {
	return MGPU_DIV_UP(numPartitions + 1, BINARY_SEARCH_CTA_SIZE);
}

template<int NT, MgpuBounds Bounds, typename It, typename Comp>
MGPU_DEVICE static void KernelBinarySearch(int tid, int block, int count, 
	It data_global, int numItems, int nv, int* partitions_global,
	int numSearches, Comp comp) {

	int gid = NT * block + tid;
	if(gid < numSearches) {
		int p = BinarySearch<Bounds>(data_global, numItems, 
			min(nv * gid, count), comp);
		partitions_global[gid] = p;
	}
}

// Static launch version.
template<int NT, MgpuBounds Bounds, typename It, typename Comp>
__global__ void KernelStaticBinarySearch(int count, It data_global,
	int numItems, int nv, int* partitions_global, int numSearches, Comp comp) {

	KernelBinarySearch<NT, Bounds>(threadIdx.x, blockIdx.x, count,
		data_global, numItems, nv, partitions_global, numSearches, comp);
}

template<MgpuBounds Bounds, typename It1, typename Comp>
MGPU_MEM(int) BinarySearchPartitions(int count, It1 data_global, int numItems,
	int nv, Comp comp, CudaContext& context) {

	int numPartitions = NumBinarySearchPartitions(count, nv);
	int numBlocks = NumBinarySearchBlocks(numPartitions);
	MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);

	KernelStaticBinarySearch<BINARY_SEARCH_CTA_SIZE, Bounds>
		<<<numBlocks, BINARY_SEARCH_CTA_SIZE, 0, context.Stream()>>>(count, 
			data_global, numItems, nv, partitionsDevice->get(), 
			numPartitions + 1, comp);
	MGPU_SYNC_CHECK("KernelStaticBinarySearch");

	return partitionsDevice;
}


// Dynamic launch version.
template<int NT, MgpuBounds Bounds, typename CountT, typename DataT,
	typename NumItems, typename NV, typename PartitionsGlobal,
	typename Comp>
__global__ void KernelDynamicBinarySearch(int* counter_global, CountT count_,
	DataT data_global_, NumItems numItems_, NV nv_, 
	PartitionsGlobal partitions_global_, Comp comp) {

	int tid = threadIdx.x;

	int count = ParamAccess(count_);
	typename ParamType<DataT>::Type data_global = ParamAccess(data_global_);
	int numItems = ParamAccess(numItems_);
	int nv = ParamAccess(nv_);
	int* partitions_global = ParamAccess(partitions_global_);
	
	int numPartitions = NumBinarySearchPartitions(count, nv);
	int numTiles = NumBinarySearchBlocks(numPartitions);

	WorkDistribution wd(numTiles);
	while(wd.EvenShare()) {
		KernelBinarySearch<NT, Bounds>(tid, wd.CurrentTile(), count, 
			data_global, numItems, nv, partitions_global, numPartitions + 1,
			comp);
	}
}

template<MgpuBounds Bounds, typename CountT, typename DataT,
	typename NumItems, typename NV, typename PartitionsGlobal, 
	typename Comp>
void BinarySearchPartitionsDynamic(CountT count_, DataT data_global_, 
	NumItems numItems_, NV nv_, PartitionsGlobal partitions_global_, 
	Comp comp, CudaContext& context) {

	// Compute the number of CTAs that can run concurrently.
	int occ = context.Device().OccupancyDevice(
		&KernelDynamicBinarySearch<BINARY_SEARCH_CTA_SIZE, Bounds, CountT,
			DataT, NumItems, NV, PartitionsGlobal, Comp>,
		BINARY_SEARCH_CTA_SIZE);

	// Retrieve a zero in device memory to keep the next tile to process.
	int* counter_global = context.GetCounter();

	// Load the kernel with dynamic work distribution.
	KernelDynamicBinarySearch<BINARY_SEARCH_CTA_SIZE, Bounds>
		<<<occ, BINARY_SEARCH_CTA_SIZE, 0, context.Stream()>>>(counter_global,
			count_, data_global_, numItems_, nv_, partitions_global_, comp);

	MGPU_SYNC_CHECK("KernelDynamicBinarySearch");
}

////////////////////////////////////////////////////////////////////////////////
// KernelMergePartition

template<int NT, MgpuBounds Bounds, typename It1, typename It2, typename Comp>
MGPU_DEVICE void KernelMergePartition(int tid, int block, It1 a_global, 
	int aCount, It2 b_global, int bCount, int nv, int coop, int* mp_global, 
	int numSearches, Comp comp) {

	int partition = NT * block + tid;
	if(partition < numSearches) {
		int a0 = 0, b0 = 0;
		int gid = nv * partition;
		if(coop) {
			int3 frame = FindMergesortFrame(coop, partition, nv);
			a0 = frame.x;
			b0 = min(aCount, frame.y);
			bCount = min(aCount, frame.y + frame.z) - b0;
			aCount = min(aCount, frame.x + frame.z) - a0;

			// Put the cross-diagonal into the coordinate system of the input
			// lists.
			gid -= a0;
		}
		int mp = MergePath<Bounds>(a_global + a0, aCount, b_global + b0, 
			bCount, min(gid, aCount + bCount), comp);
		mp_global[partition] = mp;
	}
}

template<int NT, MgpuBounds Bounds, typename It1, typename It2, typename Comp>
__global__ void KernelStaticMergePartition(It1 a_global, int aCount,
	It2 b_global, int bCount, int nv, int coop, int* mp_global, 
	int numSearches, Comp comp) {

	KernelMergePartition<NT, Bounds>(threadIdx.x, blockIdx.x, a_global, aCount,
		b_global, bCount, nv, coop, mp_global, numSearches, comp);
}

template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
MGPU_MEM(int) MergePathPartitions(It1 a_global, int aCount, It2 b_global,
	int bCount, int nv, int coop, Comp comp, CudaContext& context) {

	int numPartitions = NumBinarySearchPartitions(aCount + bCount, nv);
	int numBlocks = NumBinarySearchBlocks(numPartitions);
	MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);

	KernelStaticMergePartition<BINARY_SEARCH_CTA_SIZE, Bounds>
		<<<numBlocks, BINARY_SEARCH_CTA_SIZE, 0, context.Stream()>>>(a_global,
		aCount, b_global, bCount, nv, coop, partitionsDevice->get(),
		numPartitions + 1, comp);
	MGPU_SYNC_CHECK("KernelStaticMergePartition");

	return partitionsDevice;
}


template<int NT, MgpuBounds Bounds, typename AGlobal, typename ACount,
	typename BGlobal, typename BCount, typename NV, typename Coop, 
	typename MPGlobal, typename Comp>
__global__ void KernelDynamicMergePartition(int* counter_global, 
	AGlobal a_global_, ACount aCount_, BGlobal b_global_, BCount bCount_,
	NV nv_, Coop coop_, MPGlobal mp_global_, Comp comp) {

	int tid = threadIdx.x;

	typename ParamType<AGlobal>::Type a_global = ParamAccess(a_global_);
	int aCount = ParamAccess(aCount_);
	typename ParamType<BGlobal>::Type b_global = ParamAccess(b_global_);
	int bCount = ParamAccess(bCount_);
	int nv = ParamAccess(nv_);
	int coop = ParamAccess(coop_);
	typename ParamType<MPGlobal>::Type mp_global = ParamAccess(mp_global_);

	int numPartitions = NumBinarySearchPartitions(aCount + bCount, nv);
	int numTiles = NumBinarySearchBlocks(numPartitions);

	WorkDistribution wd(numTiles);
	while(wd.EvenShare()) {
		KernelMergePartition<NT, Bounds>(tid, wd.CurrentTile(), a_global,
			aCount, b_global, bCount, nv, coop, mp_global, numPartitions + 1,
			comp);
	}
}

template<MgpuBounds Bounds, typename AGlobal, typename ACount, typename BGlobal,
	typename BCount, typename NV, typename Coop, typename MPGlobal, 
	typename Comp>
void MergePathPartitionsDynamic(AGlobal a_global_, ACount aCount_, 
	BGlobal b_global_, BCount bCount_, NV nv_, Coop coop_, 
	MPGlobal mp_global_, Comp comp, CudaContext& context) {

	// Launch only enough CTAs to exactly fill the device.
	int occ = context.Device().OccupancyDevice(
		(void*)&KernelDynamicMergePartition<BINARY_SEARCH_CTA_SIZE, Bounds, 
			AGlobal, ACount, BGlobal, BCount, NV, Coop, MPGlobal, Comp>,
		BINARY_SEARCH_CTA_SIZE);

	// Get a zero-initialized counter for the work-distribution mechanism.
	int* counter_global = context.GetCounter();

	// Launch the dynamically-scheduled kernel.
	KernelDynamicMergePartition<BINARY_SEARCH_CTA_SIZE, Bounds>
		<<<occ, BINARY_SEARCH_CTA_SIZE, 0, context.Stream()>>>(counter_global,
			a_global_, aCount_, b_global_, bCount_, nv_, coop_, mp_global_, 
			comp);

	MGPU_SYNC_CHECK("KernelDynamicBinarySearch");
}

////////////////////////////////////////////////////////////////////////////////
// FindSetPartitions

template<int NT, bool Duplicates, typename InputIt1, typename InputIt2,
	typename Comp>
__global__ void KernelSetPartition(InputIt1 a_global, int aCount, 
	InputIt2 b_global, int bCount, int nv, int* bp_global, int numSearches,
	Comp comp) {

	int gid = NT * blockIdx.x + threadIdx.x;
	if(gid < numSearches) {
		int diag = min(aCount + bCount, gid * nv);

		// Search with level 4 bias. This helps the binary search converge 
		// quickly for small runs of duplicates (the common case).
		int2 bp = BalancedPath<Duplicates, int64>(a_global, aCount, b_global,
			bCount, diag, 4, comp);

		if(bp.y) bp.x |= 0x80000000;
		bp_global[gid] = bp.x;
	}
}

template<bool Duplicates, typename It1, typename It2, typename Comp>
MGPU_MEM(int) FindSetPartitions(It1 a_global, int aCount, It2 b_global,
	int bCount, int nv, Comp comp, CudaContext& context) {

	const int NT = 64;
	int numPartitions = MGPU_DIV_UP(aCount + bCount, nv);
	int numPartitionBlocks = MGPU_DIV_UP(numPartitions + 1, NT);
	MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);

	KernelSetPartition<NT, Duplicates>
		<<<numPartitionBlocks, NT, 0, context.Stream()>>>(a_global, aCount,
		b_global, bCount, nv, partitionsDevice->get(), numPartitions + 1, comp);
	MGPU_SYNC_CHECK("KernelSetPartition");

	return partitionsDevice;
}

} // namespace mgpu
