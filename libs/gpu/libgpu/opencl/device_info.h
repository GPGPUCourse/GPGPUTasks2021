#pragma once

#include <cstddef>
#include <string>
#include <set>

typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;

namespace ocl {

class DeviceInfo {
public:
	DeviceInfo();

	void init(cl_device_id device_id);
	void print() const;

	bool 				isIntelGPU() const;
	bool				hasExtension(std::string extension)	{ return extensions.count(extension) > 0;}

	std::string				device_name;
	std::string				vendor_name;
	unsigned int			device_type;
	unsigned int			vendor_id;
	size_t					max_compute_units;
	size_t					max_mem_alloc_size;
	size_t					max_workgroup_size;
	size_t					max_work_item_sizes[3];
	size_t					global_mem_size;
	size_t 					device_address_bits;
	size_t					max_work_item_dimensions;
	unsigned int			warp_size;
	size_t					wavefront_width;
	std::string				driver_version;
	std::string				platform_version;

	int 					opencl_major_version;
	int 					opencl_minor_version;

	std::set<std::string>	extensions;

protected:
	void				initExtensions(cl_platform_id platform_id, cl_device_id device_id);
	void				initOpenCLVersion(cl_platform_id platform_id, cl_device_id device_id);
	void				parseOpenCLVersion(char* buffer, int buffer_limit, int& major_version, int& minor_verions);
};

}
