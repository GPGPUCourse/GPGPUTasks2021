#pragma once

#include <stdexcept>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <libutils/string_utils.h>
#include <libgpu/utils.h>

namespace ocl {

#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE	0x11B3  // since OpenCL 1.1

#define CL_NV_DEVICE_ATTRIBUTE_QUERY_EXT				"cl_nv_device_attribute_query"
#define CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT				"cl_amd_device_attribute_query"

#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV							0x4008
#endif

#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV						0x4009
#endif

#define CL_DEVICE_TOPOLOGY_AMD							0x4037
#define CL_DEVICE_BOARD_NAME_AMD						0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD				0x4039
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD					0x4043

enum VENDOR {
	ID_AMD		= 0x1002,
	ID_INTEL	= 0x8086,
	ID_NVIDIA	= 0x10de,
};

class ocl_exception : public gpu::gpu_exception {
public:
	ocl_exception(std::string msg) throw ()					: gpu_exception(msg)							{	}
	ocl_exception(const char *msg) throw ()					: gpu_exception(msg)							{	}
	ocl_exception() throw ()								: gpu_exception("OpenCL exception")				{	}
};

class ocl_bad_alloc : public gpu::gpu_bad_alloc {
public:
	ocl_bad_alloc(std::string msg) throw ()					: gpu_bad_alloc(msg)							{	}
	ocl_bad_alloc(const char *msg) throw ()					: gpu_bad_alloc(msg)							{	}
	ocl_bad_alloc() throw ()								: gpu_bad_alloc("OpenCL exception")				{	}
};

std::string errorString(cl_int code);

static inline void reportError(cl_int err, int line, std::string prefix="")
{
	if (CL_SUCCESS == err)
		return;

	std::string message = prefix + errorString(err) + " (" + to_string(err) + ")" + " at line " + to_string(line);

	switch (err) {
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		throw ocl_bad_alloc(message);
	default:
		throw ocl_exception(message);
	}
}

#define OCL_SAFE_CALL(expr)  ocl::reportError(expr, __LINE__, "")
#define OCL_SAFE_CALL_MESSAGE(expr, message)  ocl::reportError(expr, __LINE__, message)

}
