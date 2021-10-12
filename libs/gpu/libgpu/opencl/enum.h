#pragma once

#include <string>
#include <vector>
#include <libgpu/opencl/engine.h>

typedef struct _cl_platform_id *	cl_platform_id;
typedef struct _cl_device_id *		cl_device_id;

class OpenCLEnum {
public:
	OpenCLEnum();
	~OpenCLEnum();

	class Device {
	public:
		Device()
		{
			device_type			= 0;
			compute_units		= 0;
			mem_size			= 0;
			clock				= 0;
			nvidia_pci_bus_id	= 0;
			nvidia_pci_slot_id	= 0;
			has_cl_khr_spir		= false;
		}

		cl_device_id			id;
		unsigned int			vendor_id;
		cl_platform_id			platform_id;
		cl_device_type			device_type;
		std::string				name;
		std::string				vendor;
		std::string				version;
		unsigned int			compute_units;
		unsigned long long		mem_size;
		unsigned int			clock;
		unsigned int			nvidia_pci_bus_id;
		unsigned int			nvidia_pci_slot_id;
		bool					has_cl_khr_spir;

		ocl::sh_ptr_ocl_engine	createEngine(bool printInfo=false);

		bool	isCPU(void)	{ return device_type == CL_DEVICE_TYPE_CPU;	}
		bool	isGPU(void)	{ return device_type == CL_DEVICE_TYPE_GPU;	}

		bool printInfo() const;
	};

	class Platform {
	public:
		cl_platform_id			id;
		std::string				name;
		std::string				vendor;
		std::string				version;
	};

	bool	enumDevices();
	std::vector<Device> &	devices()	{ return devices_;		}
	std::vector<Platform> &	platforms()	{ return platforms_;	}

protected:
	bool	enumPlatforms();
	bool	enumDevices(cl_platform_id platform_id);

	bool	queryDeviceInfo(Device &device);
	bool	queryDeviceInfo(cl_device_id device_id, unsigned int param, std::string &value, const std::string &param_name, size_t max_size = 0);
	template <typename T>
	bool	queryDeviceInfo(cl_device_id device_id, unsigned int param, T &value, const std::string &param_name);
	bool	queryPlatformInfo(cl_platform_id platform_id, unsigned int param, std::string &value, const std::string &param_name, size_t max_size);
	bool	queryExtensionList(cl_device_id device_id, std::set<std::string> &extensions);

	static	bool	compareDevice(const Device &dev1, const Device &dev2);

	std::vector<Device>		devices_;
	std::vector<Platform>	platforms_;
};
