#pragma once

#include <string>
#include <vector>

typedef struct _cl_device_id *		cl_device_id;

namespace gpu {

class Device {
public:
	std::string			name;
	std::string			opencl_vendor;
	std::string			opencl_version;
	unsigned int		compute_units;
	unsigned int		clock;
	unsigned long long	mem_size;
	unsigned int		pci_bus_id;
	unsigned int		pci_device_id;

	bool				supports_opencl;
	bool				supports_cuda;

	cl_device_id		device_id_opencl;
	int					device_id_cuda;

	bool				printInfo() const;

	bool				supportsFreeMemoryQuery() const;
	unsigned long long	getFreeMemory() const;

	bool operator< (const Device &other) const
	{
		if (name			< other.name)				return true;
		if (name			> other.name)				return false;
		if (pci_bus_id		< other.pci_bus_id)			return true;
		if (pci_bus_id		> other.pci_bus_id)			return false;
		if (pci_device_id	< other.pci_device_id)		return true;
		if (pci_device_id	> other.pci_device_id)		return false;
		if (supports_opencl	< other.supports_opencl)	return true;
		if (supports_opencl	> other.supports_opencl)	return false;
		if (supports_cuda	< other.supports_cuda)		return true;
		if (supports_cuda	> other.supports_cuda)		return false;
		return false;
	}
};

std::vector<Device> enumDevices();
std::vector<Device> selectDevices(unsigned int mask, bool silent=false);

}
