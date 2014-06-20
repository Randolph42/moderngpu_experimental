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
#include "../device/ctaloadbalance.cuh"
#include "../kernels/search.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelLoadBalance

template<int NT, int VT>
struct KernelLoadBalance {

	struct Storage {
		int indices[NT * (VT + 1)];
	};

	template<typename InputIt>
	MGPU_DEVICE static void Kernel(int tid, int block, int aCount,
		InputIt b_global, int bCount, const int* mp_global, 
		int* indices_global, Storage& storage) {

		int4 range = CTALoadBalance<NT, VT>(aCount, b_global, bCount, block, 
			tid, mp_global, storage.indices, false);
		aCount = range.y - range.x;

		DeviceSharedToGlobal<NT, VT>(aCount, storage.indices, tid, 
			indices_global + range.x);
	}
};


////////////////////////////////////////////////////////////////////////////////
// LoadBalanceSearch with static scheduling.

template<typename Tuning, typename InputIt>
MGPU_LAUNCH_BOUNDS void KernelStaticLoadBalance(int aCount, InputIt b_global,
	int bCount, const int* mp_global, int* indices_global) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	__shared__ typename KernelLoadBalance<NT, VT>::Storage storage;

	KernelLoadBalance<NT, VT>::Kernel(threadIdx.x, blockIdx.x, aCount,
		b_global, bCount, mp_global, indices_global, storage);
}

template<typename InputIt>
MGPU_HOST void LoadBalanceSearch(int aCount, InputIt b_global, int bCount,
	int* indices_global, CudaContext& context) {

	const int NT = 128;
	const int VT = 7;
	typedef LaunchBoxVT<NT, VT> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);
	const int NV = launch.x * launch.y;
	  
	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsUpper>(
		mgpu::counting_iterator<int>(0), aCount, b_global, bCount, NV, 0,
		mgpu::less<int>(), context);

	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
	KernelStaticLoadBalance<Tuning>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(aCount, b_global,
		bCount, partitionsDevice->get(), indices_global);
	MGPU_SYNC_CHECK("KernelStaticLoadBalance");
}


////////////////////////////////////////////////////////////////////////////////
// LoadBalanceSearch with dynamic scheduling.

template<typename Tuning, typename ACount, typename BGlobal, typename BCount,
	typename IndicesGlobal>
__global__ void KernelDynamicLoadBalance(int* counter_global, ACount aCount_,
	BGlobal b_global_, BCount bCount_, int* mp_global, 
	IndicesGlobal indices_global_) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;

	union Shared {
		typename WorkDistribution::Storage distribution;
		typename KernelLoadBalance<NT, VT>::Storage loadBalance;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int aCount = ParamAccess(aCount_);
	typename ParamType<BGlobal>::Type b_global = ParamAccess(b_global_);
	int bCount = ParamAccess(bCount_);
	int* indices_global = ParamAccess(indices_global_);

	int numTiles = MGPU_DIV_UP(aCount + bCount, NV);

	WorkDistribution wd(numTiles);
//	while(wd.WorkStealing(tid, counter_global, shared.distribution)) {
	while(wd.EvenShare()) {
		KernelLoadBalance<NT, VT>::Kernel(tid, wd.CurrentTile(), aCount, 
			b_global, bCount, mp_global, indices_global, shared.loadBalance);
	}
}

// The user must provide adequete storage for the tile partitionings.
// This is (aCount + bCount) / (NT * VT) + 1.
template<typename ACount, typename BGlobal, typename BCount,
	typename IndicesGlobal>
void LoadBalanceSearchDynamic(ACount aCount_, BGlobal b_global_,
	BCount bCount_, IndicesGlobal indices_global_, int* partitions_global, 
	CudaContext& context) {

	const int NT = 128;
	const int VT = 7;
	const int NV = NT * VT;
	typedef LaunchBoxVT<NT, VT> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	// Partition the tiles. Store into the user-provided partitions_global
	// buffer.
	MergePathPartitionsDynamic<MgpuBoundsUpper>(
		mgpu::counting_iterator<int>(0), aCount_, b_global_, bCount_, NV,
		0, partitions_global, mgpu::less<int>(), context);
		
	// Compute the occupancy of the load-balancing search kernel.
	int occ = context.Device().OccupancyDevice(
		(void*)&KernelDynamicLoadBalance<Tuning, ACount, BGlobal, BCount, 
			IndicesGlobal>, NT);

	// Get a zeroed counter.
	int* counter_global = context.GetCounter();

	// Generate the load-balancing search output.
	KernelDynamicLoadBalance<Tuning><<<occ, NT, 0, context.Stream()>>>(
		counter_global, aCount_, b_global_, bCount_, partitions_global,
		indices_global_);

	MGPU_SYNC_CHECK("KernelDynamicLoadBalance");
}

} // namespace mgpu
