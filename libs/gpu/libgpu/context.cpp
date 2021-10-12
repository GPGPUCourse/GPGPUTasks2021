#include "context.h"

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#include <libgpu/cuda/cuda_api.h>
#endif

namespace gpu {

THREAD_LOCAL Context::Data *Context::data_current_ = 0;

Context::Data::Data()
{
	type			= TypeUndefined;
	cuda_device		= 0;
	cuda_context	= 0;
	cuda_stream		= 0;
	ocl_device		= 0;
	activated		= false;
}

Context::Data::~Data()
{
	if (data_current_ != this) {
		if (data_current_ != 0) {
			std::cerr << "Another GPU context found on context destruction" << std::endl;
		}
	} else {
		data_current_ = 0;
	}

#ifdef CUDA_SUPPORT
	if (cuda_stream) {
		cudaError_t err = cudaStreamDestroy(cuda_stream);
		if (cudaSuccess != err)
			std::cerr << "Warning: cudaStreamDestroy failed: " << cuda::formatError(err) << std::endl;
	}

#ifndef CUDA_USE_PRIMARY_CONTEXT
	if (cuda_context) {
		CUresult err = cuCtxDestroy(cuda_context);
		if (CUDA_SUCCESS != err)
			std::cerr << "Warning: cuCtxDestroy failed: " << cuda::formatDriverError(err) << std::endl;
	}
#endif
#endif
}

Context::Context()
{
	data_ = data_current_;
}

void Context::clear()
{
	data_ = NULL;
}

void Context::init(int device)
{
#ifdef CUDA_SUPPORT
#ifndef CUDA_USE_PRIMARY_CONTEXT
	if (!cuda_api_init())
		throw cuda::cuda_exception("Can't load nvcuda library");
#endif
	std::shared_ptr<Data> data = std::make_shared<Data>();
	data->type				= TypeCUDA;
	data->cuda_device		= device;
	data_ref_	= data;
#endif
}

void Context::init(struct _cl_device_id *device)
{
	std::shared_ptr<Data> data = std::make_shared<Data>();
	data->type				= TypeOpenCL;
	data->ocl_device		= device;
	data_ref_	= data;
}

bool Context::isInitialized()
{
	return data_ref_.get() && data_ref_->type != TypeUndefined;
}

bool Context::isGPU()
{
	return (type() != TypeUndefined);
}

bool Context::isIntelGPU()
{
	if (type() != TypeOpenCL) {
		return false;
	}

	return cl()->deviceInfo().isIntelGPU();
}

bool Context::isGoldChecksEnabled()
{
	return false; // NOTTODO: Make it switchable
}

void Context::activate()
{
	if (!data_ref_)
		throw std::runtime_error("Unexpected GPU context activate call");

	// create cuda stream on first activate call
	if (!data_ref_->activated) {
#ifdef CUDA_SUPPORT
		if (data_ref_->type == TypeCUDA) {
#ifndef CUDA_USE_PRIMARY_CONTEXT
			// It is claimed that contexts are thread safe starting from CUDA 4.0.
			// Nevertheless, we observe crashes in nvcuda.dll if the same device is used in parallel from 2 threads using its primary context.
			// To avoid this problem we create a separate standard context for each processing thread.
			// https://devtalk.nvidia.com/default/topic/519087/cuda-programming-and-performance/cuda-context-and-threading/post/3689477/#3689477
			// http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#axzz4g8KX5QV5

			CUdevice device = 0;
			CU_SAFE_CALL( cuDeviceGet(&device, data_ref_->cuda_device) );
			CU_SAFE_CALL( cuCtxCreate(&data_ref_->cuda_context, 0, device) );
#else
			CUDA_SAFE_CALL( cudaSetDevice(data_ref_->cuda_device) );
#endif
			CUDA_SAFE_CALL( cudaStreamCreate(&data_ref_->cuda_stream) );
		}
#endif

		if (data_ref_->type == TypeOpenCL) {
			ocl::sh_ptr_ocl_engine engine = std::make_shared<ocl::OpenCLEngine>();
			engine->init(data_ref_->ocl_device);
			data_ref_->ocl_engine = engine;
		}

		data_ref_->activated = true;
	}

	if (data_current_ && data_current_ != data_ref_.get())
		throw std::runtime_error("Another GPU context is already active");

	data_			= data_ref_.get();
	data_current_	= data_;
}

Context::Data *Context::data() const
{
	if (!data_)
		throw std::runtime_error("Null context");

	return data_;
}

size_t Context::getCoresEstimate()
{
	size_t compute_units = 1;

	switch (type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			cudaDeviceProp deviceProp;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, data_->cuda_device));
			compute_units = (size_t) deviceProp.multiProcessorCount;
			break;
#endif
		case Context::TypeOpenCL:
			compute_units = cl()->maxComputeUnits();
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	return compute_units * 256;
}

size_t Context::getTotalMemory()
{
	size_t total_mem_size = 0;
	size_t free_mem_size = 0;

	switch (type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
			break;
#endif
		case Context::TypeOpenCL:
			total_mem_size = cl()->totalMemSize();
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	return total_mem_size;
}

size_t Context::getFreeMemory()
{
	size_t total_mem_size = 0;
	size_t free_mem_size = 0;

	switch (type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
			break;
#endif
		case Context::TypeOpenCL:
			total_mem_size = cl()->totalMemSize();
			free_mem_size = total_mem_size - total_mem_size / 5;
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	return free_mem_size;
}

size_t Context::getMaxMemAlloc()
{
	size_t max_mem_alloc_size = 0;

#ifdef CUDA_SUPPORT
	if (type() == gpu::Context::TypeCUDA) {
		size_t total_mem_size = 0;
		size_t free_mem_size = 0;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
		max_mem_alloc_size = total_mem_size / 2;
	} else
#endif
	if (type() == gpu::Context::TypeOpenCL) {
		max_mem_alloc_size = cl()->maxMemAllocSize();
	} else {
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	return max_mem_alloc_size;
}

size_t Context::getMaxWorkgroupSize()
{
	size_t max_workgroup_size = 0;

	switch (type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			int value;
			CUDA_SAFE_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, data_->cuda_device));
			max_workgroup_size = value;
			break;
#endif
		case Context::TypeOpenCL:
			max_workgroup_size = cl()->maxWorkgroupSize();
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	return max_workgroup_size;
}

std::vector<size_t> Context::getMaxWorkItemSizes()
{
	std::vector<size_t> work_item_sizes(3);

	switch (type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			int value[3];
			CUDA_SAFE_CALL(cudaDeviceGetAttribute(&value[0], cudaDevAttrMaxBlockDimX, data_->cuda_device));
			CUDA_SAFE_CALL(cudaDeviceGetAttribute(&value[1], cudaDevAttrMaxBlockDimY, data_->cuda_device));
			CUDA_SAFE_CALL(cudaDeviceGetAttribute(&value[2], cudaDevAttrMaxBlockDimZ, data_->cuda_device));
			for (int i = 0; i < 3; ++i) {
				work_item_sizes[i] = value[i];
			}
			break;
#endif
		case Context::TypeOpenCL:
			for (int i = 0; i < 3; ++i) {
				work_item_sizes[i] = cl()->maxWorkItemSizes(i);
			}
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	return work_item_sizes;
}

Context::Type Context::type() const
{
	if (data_)
		return data_->type;
	if (data_ref_)
		return data_ref_->type;
	return TypeUndefined;
}

ocl::sh_ptr_ocl_engine Context::cl() const
{
	return data()->ocl_engine;
}

cudaStream_t Context::cudaStream() const
{
	return data()->cuda_stream;
}

}
