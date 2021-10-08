#pragma once

#include "utils.h"

#ifdef CUDA_SUPPORT
	#include <vector_types.h>
#endif

namespace gpu {
	class WorkSize {
	public:
		WorkSize(unsigned int groupSizeX, unsigned int workSizeX)
		{
			init(1, groupSizeX, 1, 1, workSizeX, 1, 1);
		}

		WorkSize(unsigned int groupSizeX, unsigned int groupSizeY, unsigned int workSizeX, unsigned int workSizeY)
		{
			init(2, groupSizeX, groupSizeY, 1, workSizeX, workSizeY, 1);
		}

		WorkSize(unsigned int groupSizeX, unsigned int groupSizeY, unsigned int groupSizeZ,unsigned  int workSizeX, unsigned int workSizeY, unsigned int workSizeZ)
		{
			init(3, groupSizeX, groupSizeY, groupSizeZ, workSizeX, workSizeY, workSizeZ);
		}

#ifdef CUDA_SUPPORT
		const dim3 &cuBlockSize() const {
			return blockSize;
		}

		const dim3 &cuGridSize() const {
			return gridSize;
		}
#endif

		const size_t *clLocalSize() const {
			return localWorkSize;
		}

		const size_t *clGlobalSize() const {
			return globalWorkSize;
		}

		int clWorkDim() const {
			return workDims;
		}

	private:
		void init(int workDims, unsigned int groupSizeX, unsigned int groupSizeY, unsigned int groupSizeZ, unsigned  int workSizeX, unsigned int workSizeY, unsigned int workSizeZ)
		{
			this->workDims = workDims;

			localWorkSize[0] = groupSizeX;
			localWorkSize[1] = groupSizeY;
			localWorkSize[2] = groupSizeZ;

			workSizeX = gpu::divup(workSizeX, groupSizeX) * groupSizeX;
			workSizeY = gpu::divup(workSizeY, groupSizeY) * groupSizeY;
			workSizeZ = gpu::divup(workSizeZ, groupSizeZ) * groupSizeZ;

			globalWorkSize[0] = workSizeX;
			globalWorkSize[1] = workSizeY;
			globalWorkSize[2] = workSizeZ;

#ifdef CUDA_SUPPORT
			blockSize	= dim3(groupSizeX, groupSizeY, groupSizeZ);
			gridSize	= dim3(gpu::divup(workSizeX, groupSizeX),
							   gpu::divup(workSizeY, groupSizeY),
							   gpu::divup(workSizeZ, groupSizeZ));
#endif
		}

	private:
		size_t	localWorkSize[3];
		size_t	globalWorkSize[3];
		int workDims;

#ifdef CUDA_SUPPORT
		dim3	blockSize;
		dim3	gridSize;
#endif
	};
}