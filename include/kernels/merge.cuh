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
#include "../kernels/search.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelMerge

template<int NT, int VT, typename KeyType>
struct KernelMerge {
	union Storage {
		KeyType keys[NT * (VT + 1)];
		int indices[NT * VT];
	};

	template<bool HasValues, bool LoadExtended, typename KeysIt1,
		typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
		typename ValsIt3, typename Comp>
	MGPU_DEVICE static void Kernel(int tid, int block, KeysIt1 aKeys_global,
		ValsIt1 aVals_global, int aCount, KeysIt2 bKeys_global,
		ValsIt2 bVals_global, int bCount, const int* mp_global, int coop,
		KeysIt3 keys_global, ValsIt3 vals_global, Comp comp, 
		Storage& storage) {

		int4 range = ComputeMergeRange(aCount, bCount, block, coop, NT * VT, 
			mp_global);

		DeviceMerge<NT, VT, HasValues, LoadExtended>(aKeys_global, aVals_global, 
			aCount, bKeys_global, bVals_global, bCount, tid, block, range, 
			storage.keys, storage.indices, keys_global, vals_global, comp);
	}
};

////////////////////////////////////////////////////////////////////////////////
// KernelStaticMerge. Use grid size configured by host.

template<typename Tuning, bool HasValues, bool LoadExtended, typename KeysIt1,
	typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
	typename ValsIt3, typename Comp>
MGPU_LAUNCH_BOUNDS void KernelStaticMerge(KeysIt1 aKeys_global, 
	ValsIt1 aVals_global, int aCount, KeysIt2 bKeys_global, 
	ValsIt2 bVals_global, int bCount, const int* mp_global, int coop, 
	KeysIt3 keys_global, ValsIt3 vals_global, Comp comp) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;

	typedef KernelMerge<NT, VT, KeyType> KM;
	__shared__ typename KM::Storage storage;

	KM::Kernel<HasValues, LoadExtended>(threadIdx.x, blockIdx.x, aKeys_global,
		aVals_global, aCount, bKeys_global, bVals_global, bCount, mp_global,
		coop, keys_global, vals_global, comp, storage);
}

////////////////////////////////////////////////////////////////////////////////
// MergeKeys static scheduling.

template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename Comp>
MGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
	int bCount, KeysIt3 keys_global, Comp comp, CudaContext& context) {
	
	typedef typename std::iterator_traits<KeysIt1>::value_type T;
	typedef LaunchBoxVT<
		128, 23, 0,
		128, 11, 0,
		128, (sizeof(T) > 4) ? 7 : 11, 0
	> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	const int NV = launch.x * launch.y;
	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);

	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
	KernelStaticMerge<Tuning, false, true>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(aKeys_global, 
		(const int*)0, aCount, bKeys_global, (const int*)0, bCount, 
		partitionsDevice->get(), 0, keys_global, (int*)0, comp);
	MGPU_SYNC_CHECK("KernelStaticMerge");
}
template<typename KeysIt1, typename KeysIt2, typename KeysIt3>
MGPU_HOST void MergeKeys(KeysIt1 aKeys_global, int aCount, KeysIt2 bKeys_global,
	int bCount, KeysIt3 keys_global, CudaContext& context) {

	typedef mgpu::less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
	return MergeKeys(aKeys_global, aCount, bKeys_global, bCount, keys_global,
		Comp(), context);
}

////////////////////////////////////////////////////////////////////////////////
// MergePairs static scheduling.

template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
	typename ValsIt2, typename ValsIt3, typename Comp>
MGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
	KeysIt3 keys_global, ValsIt3 vals_global, Comp comp, CudaContext& context) {

	typedef typename std::iterator_traits<KeysIt1>::value_type T;
	typedef LaunchBoxVT<
		128, 11, 0,
		128, 11, 0,
		128, (sizeof(T) > 4) ? 7 : 11, 0
	> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	const int NV = launch.x * launch.y;
	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);

	int numBlocks = MGPU_DIV_UP(aCount + bCount, NV);
	KernelStaticMerge<Tuning, true, false>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(aKeys_global,
		aVals_global, aCount, bKeys_global, bVals_global, bCount, 
		partitionsDevice->get(), 0, keys_global, vals_global, comp);
	MGPU_SYNC_CHECK("KernelStaticMerge");
}
template<typename KeysIt1, typename KeysIt2, typename KeysIt3, typename ValsIt1,
	typename ValsIt2, typename ValsIt3>
MGPU_HOST void MergePairs(KeysIt1 aKeys_global, ValsIt1 aVals_global, 
	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
	KeysIt3 keys_global, ValsIt3 vals_global, CudaContext& context) {

	typedef mgpu::less<typename std::iterator_traits<KeysIt1>::value_type> Comp;
	return MergePairs(aKeys_global, aVals_global, aCount, bKeys_global, 
		bVals_global, bCount, keys_global, vals_global, Comp(), context);
}

////////////////////////////////////////////////////////////////////////////////
// KernelDynamicMerge. Use constant grid size and dynamically provision work.

template<typename Tuning, bool HasValues, bool LoadExtended, typename KeysIt1,
	typename ACount, typename KeysIt2, typename BCount, typename KeysIt3,
	typename ValsIt1, typename ValsIt2, typename ValsIt3, typename Comp>
MGPU_LAUNCH_BOUNDS void KernelDynamicMerge(int* counter_global, 
	KeysIt1 aKeys_global_, ValsIt1 aVals_global_, ACount aCount_, 
	KeysIt2 bKeys_global_, ValsIt2 bVals_global_, BCount bCount_, 
	const int* mp_global, int coop, KeysIt3 keys_global_, ValsIt3 vals_global_, 
	Comp comp) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	typedef typename std::iterator_traits<KeysIt1>::value_type KeyType;

	typedef KernelMerge<NT, VT, KeyType> KM;
	union Shared {
		typename KM::Storage mergeStorage;
		typename WorkDistribution::Storage workDistribution;
	};
	__shared__ Shared shared;

	// Pull dynamic parameters into register.
	typename ParamType<KeysIt1>::Type aKeys_global = ParamAccess(aKeys_global_);
	typename ParamType<ValsIt1>::Type aVals_global = ParamAccess(aVals_global_);
	int aCount = ParamAccess(aCount_);

	typename ParamType<KeysIt2>::Type bKeys_global = ParamAccess(bKeys_global_);
	typename ParamType<ValsIt2>::Type bVals_global = ParamAccess(bVals_global_);
	int bCount = ParamAccess(bCount_);

	typename ParamType<KeysIt3>::Type keys_global = ParamAccess(keys_global_);
	typename ParamType<ValsIt3>::Type vals_global = ParamAccess(vals_global_);
	
	// Compute the number of tiles.
	int numTiles = MGPU_DIV_UP(aCount + bCount, NV);

	// Dynamically loop over tiles.
	int tid = threadIdx.x;
	int tile = -1;
	while(WorkDistribution::EvenShare(tid, numTiles, counter_global, 
		shared.workDistribution, tile)) {

		KM::Kernel<HasValues, LoadExtended>(tid, tile, aKeys_global,
			aVals_global, aCount, bKeys_global, bVals_global, bCount, 
			mp_global, coop, keys_global, vals_global, comp, 
			shared.mergeStorage);
	}
}

template<typename KeysIt1, typename ACount, typename KeysIt2, typename BCount,
	typename KeysIt3, typename Comp>
MGPU_HOST void MergeKeysDynamic(KeysIt1 aKeys_global_, ACount aCount_, 
	KeysIt2 bKeys_global_, BCount bCount_, KeysIt3 keys_global_, Comp comp, 
	int* partitions_global, CudaContext& context) {

	
	typedef typename std::iterator_traits<KeysIt1>::value_type T;
	typedef LaunchBoxVT<
		128, 23, 0,
		128, 11, 0,
		128, (sizeof(T) > 4) ? 7 : 11, 0
	> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	const int NV = launch.x * launch.y;
	MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsLower>(
		aKeys_global, aCount, bKeys_global, bCount, NV, 0, comp, context);

}

} // namespace mgpu
