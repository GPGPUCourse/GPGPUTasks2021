#include "device.h"
#include "context.h"
#include <libgpu/opencl/enum.h>
#include <libgpu/opencl/utils.h>
#include <algorithm>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/enum.h>
#include <libgpu/cuda/utils.h>
#include <cuda_runtime.h>
#endif

namespace gpu {

std::vector<Device> enumDevices()
{
	std::vector<Device> devices;

#ifdef CUDA_SUPPORT
	CUDAEnum cuda_enum;
	cuda_enum.enumDevices();

	const std::vector<CUDAEnum::Device> &cuda_devices = cuda_enum.devices();
	for (size_t k = 0; k < cuda_devices.size(); k++) {
		const CUDAEnum::Device &cuda_device = cuda_devices[k];

		Device device;
		device.name				= cuda_device.name;
		device.compute_units	= cuda_device.compute_units;
		device.clock			= cuda_device.clock;
		device.mem_size			= cuda_device.mem_size;
		device.pci_bus_id		= cuda_device.pci_bus_id;
		device.pci_device_id	= cuda_device.pci_device_id;
		device.supports_opencl	= false;
		device.supports_cuda	= true;
		device.device_id_opencl	= 0;
		device.device_id_cuda	= cuda_device.id;
		devices.push_back(device);
	}
#endif

	OpenCLEnum opencl_enum;
	opencl_enum.enumDevices();

	const std::vector<OpenCLEnum::Device> &opencl_devices = opencl_enum.devices();
	for (size_t k = 0; k < opencl_devices.size(); k++) {
		const OpenCLEnum::Device &opencl_device = opencl_devices[k];

		Device device;
		device.name				= opencl_device.name;
		device.opencl_vendor	= opencl_device.vendor;
		device.opencl_version	= opencl_device.version;
		device.compute_units	= opencl_device.compute_units;
		device.clock			= opencl_device.clock;
		device.mem_size			= opencl_device.mem_size;
		device.pci_bus_id		= opencl_device.nvidia_pci_bus_id;
		device.pci_device_id	= opencl_device.nvidia_pci_slot_id;
		device.supports_opencl	= true;
		device.supports_cuda	= false;
		device.device_id_opencl	= opencl_device.id;
		device.device_id_cuda	= 0;
		devices.push_back(device);
	}

#ifdef CUDA_SUPPORT
	std::sort(devices.begin(), devices.end());

	// merge corresponding devices
	for (size_t k = 0; k + 1 < devices.size(); k++) {
		if (devices[k].name				!= devices[k + 1].name)				continue;
		if (devices[k].pci_bus_id		!= devices[k + 1].pci_bus_id)		continue;
		if (devices[k].pci_device_id	!= devices[k + 1].pci_device_id)	continue;

		if (!devices[k].supports_opencl && !devices[k + 1].supports_cuda) {
			devices[k].supports_opencl	= true;
			devices[k].device_id_opencl	= devices[k + 1].device_id_opencl;
			devices.erase(devices.begin() + k + 1);
		}
	}
#endif

	return devices;
}

bool Device::printInfo() const
{
#ifdef CUDA_SUPPORT
	if (supports_cuda) {
		return CUDAEnum::printInfo(device_id_cuda);
	}
#endif

	if (supports_opencl) {
		ocl::DeviceInfo device_info;
		device_info.init(device_id_opencl);
		device_info.print();
		return true;
	}

	return false;
}

bool Device::supportsFreeMemoryQuery() const
{
#ifdef CUDA_SUPPORT
	if (supports_cuda) {
		return true;
	} else
#endif
	if (supports_opencl) {
		ocl::DeviceInfo device_info;
		device_info.init(device_id_opencl);
		if (device_info.hasExtension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT)) {
			return true;
		}
	}

	return false;
}

unsigned long long Device::getFreeMemory() const
{
#ifdef CUDA_SUPPORT
	if (supports_cuda) {
		Context context;
		context.init(device_id_cuda);
		context.activate();

		size_t total_mem_size = 0;
		size_t free_mem_size = 0;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
		return free_mem_size;
	} else
#endif
	if (supports_opencl) {
		ocl::DeviceInfo device_info;
		device_info.init(device_id_opencl);
		if (device_info.device_type == CL_DEVICE_TYPE_GPU && device_info.hasExtension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT)) {
			cl_ulong free_mem = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device_id_opencl, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, sizeof(free_mem), &free_mem, NULL));
			return free_mem * 1024;
		} else {
			size_t free_mem_size = mem_size - mem_size / 5;
			return free_mem_size;
		}
	} else {
		return 0x40000000ull * 64;	// assume 64GB by default
	}
}

std::vector<Device> selectDevices(unsigned int mask, bool silent)
{
	if (!mask)
		return std::vector<Device>();

	std::vector<Device> devices = enumDevices();

	std::vector<Device> res;
	for (size_t k = 0; k < devices.size(); k++) {
		if (!(mask & (1 << k)))
			continue;

		Device &device = devices[k];
		if (!silent)
			if (!device.printInfo())
				continue;

		res.push_back(device);
	}

	return res;
}

}
