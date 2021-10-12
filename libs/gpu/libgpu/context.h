#pragma once

#include <vector>
#include <libgpu/opencl/engine.h>

typedef struct CUctx_st *cudaContext_t;
typedef struct CUstream_st *cudaStream_t;

#ifdef _MSC_VER
    #define THREAD_LOCAL __declspec(thread)
#else
    #define THREAD_LOCAL __thread
#endif

namespace gpu {

class Context {
public:
	Context();

	enum Type {
		TypeUndefined,
		TypeOpenCL,
		TypeCUDA
	};

	void	clear();
	void	init(int device);
	void	init(struct _cl_device_id *device);
	bool	isInitialized();
	bool	isGPU();
	bool	isIntelGPU();
	bool	isGoldChecksEnabled();

	void	activate();

	size_t 				getCoresEstimate();
	size_t				getTotalMemory();
	size_t				getFreeMemory();
	size_t				getMaxMemAlloc();
	size_t				getMaxWorkgroupSize();
	std::vector<size_t>	getMaxWorkItemSizes();

	Type	type() const;

	ocl::sh_ptr_ocl_engine	cl() const;
	cudaStream_t			cudaStream() const;

protected:
	class Data {
	public:
		Data();
		~Data();

		Type	type;

		int						cuda_device;
		cudaContext_t			cuda_context;
		cudaStream_t			cuda_stream;
		struct _cl_device_id *	ocl_device;
		ocl::sh_ptr_ocl_engine	ocl_engine;
		bool					activated;
	};

	Data *	data() const;

	Data *						data_;
	std::shared_ptr<Data>		data_ref_;
	static THREAD_LOCAL Data *	data_current_;
};

}
