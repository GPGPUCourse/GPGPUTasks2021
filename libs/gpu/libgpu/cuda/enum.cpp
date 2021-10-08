#ifdef CUDA_SUPPORT
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "enum.h"
#include "utils.h"

bool CUDAEnum::printInfo(int id)
{
	cudaError_t status;

	cudaDeviceProp prop;
	status = cudaGetDeviceProperties(&prop, id);
	if (status != cudaSuccess)
		return false;

	int driverVersion = 239;
	status = cudaDriverGetVersion(&driverVersion);
	if (status != cudaSuccess)
		return false;

	int runtimeVersion = 239;
	status = cudaRuntimeGetVersion(&runtimeVersion);
	if (status != cudaSuccess)
		return false;

	std::cout << "Using device: " << prop.name << ", " << prop.multiProcessorCount << " compute units, " << (prop.totalGlobalMem >> 20) << " MB global memory, compute capability " <<  prop.major << "." << prop.minor << std::endl;
	std::cout << "  driver version: " << driverVersion << ", runtime version: " << runtimeVersion << std::endl;
	std::cout << "  max work group size " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "  max work item sizes [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
	return true;
}

CUDAEnum::CUDAEnum()
{
}

CUDAEnum::~CUDAEnum()
{
}

bool CUDAEnum::compareDevice(const Device &dev1, const Device &dev2)
{
	if (dev1.name	> dev2.name)	return false;
	if (dev1.name	< dev2.name)	return true;
	if (dev1.id		> dev2.id)		return false;
	return true;
}

bool CUDAEnum::enumDevices()
{
	int device_count = 0;

	cudaError_t res = cudaGetDeviceCount(&device_count);
	if (res == cudaErrorNoDevice || res == cudaErrorInsufficientDriver)
		return true;

	if (res != cudaSuccess) {
		std::cerr << "cudaGetDeviceCount failed: " << cuda::formatError(res) << std::endl;
		return false;
	}

	for (int device_index = 0; device_index < device_count; device_index++) {
		cudaDeviceProp prop;

		res = cudaGetDeviceProperties(&prop, device_index);
		if (res != cudaSuccess) {
			std::cerr << "cudaGetDeviceProperties failed: " << cuda::formatError(res) << std::endl;
			return false;
		}

		// we don't support CUDA devices with compute capability < 2.0
		if (prop.major < 2)
			continue;

		Device device;

		device.id				= device_index;
		device.name				= prop.name;
		device.compute_units	= prop.multiProcessorCount;
		device.mem_size			= prop.totalGlobalMem;
		device.clock			= prop.clockRate / 1000;
		device.pci_bus_id		= prop.pciBusID;
		device.pci_device_id	= prop.pciDeviceID;

		devices_.push_back(device);
	}

	std::sort(devices_.begin(), devices_.end(), compareDevice);

	return true;
}
#endif
