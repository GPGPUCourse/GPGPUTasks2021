#pragma once

#include <iostream>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <libgpu/utils.h>
#include <libutils/string_utils.h>
#include <cfloat>

#define CUDA_KERNELS_ACCURATE_ERRORS_CHECKS true

#ifndef NDEBUG
#undef CUDA_KERNELS_ACCURATE_ERRORS_CHECKS
#define CUDA_KERNELS_ACCURATE_ERRORS_CHECKS true
#endif

namespace cuda {

	class cuda_exception : public gpu::gpu_exception {
	public:
		cuda_exception(std::string msg) throw ()					: gpu_exception(msg)							{	}
		cuda_exception(const char *msg) throw ()					: gpu_exception(msg)							{	}
		cuda_exception() throw ()									: gpu_exception("CUDA exception")				{	}
	};

	class cuda_bad_alloc : public gpu::gpu_bad_alloc {
	public:
		cuda_bad_alloc(std::string msg) throw ()					: gpu_bad_alloc(msg)							{	}
		cuda_bad_alloc(const char *msg) throw ()					: gpu_bad_alloc(msg)							{	}
		cuda_bad_alloc() throw ()									: gpu_bad_alloc("CUDA exception")				{	}
	};

	std::string formatError(cudaError_t code);

	static inline void reportError(cudaError_t err, int line, std::string prefix="")
	{
		if (cudaSuccess == err)
			return;

		std::string message = prefix + formatError(err) + " at line " + to_string(line);

		switch (err) {
		case cudaErrorMemoryAllocation:
			throw cuda_bad_alloc(message);
		default:
			throw cuda_exception(message);
		}
	}

	static inline void checkKernelErrors(cudaStream_t stream, int line)
	{
		reportError(cudaGetLastError(), line, "Kernel failed: ");
		if (CUDA_KERNELS_ACCURATE_ERRORS_CHECKS) {
			reportError(cudaStreamSynchronize(stream), line, "Kernel failed: ");
		}
	}

	#define CUDA_SAFE_CALL(expr)  cuda::reportError(expr, __LINE__)
	#define CUDA_CHECK_KERNEL(stream)  cuda::checkKernelErrors(stream, __LINE__)

	template <typename T>	class DataTypeRange					{ };
	template<>				class DataTypeRange<unsigned char>	{ public:	static __device__	unsigned char	min() { return 0; }			static __device__	unsigned char	max() {	return UCHAR_MAX;	}};
	template<>				class DataTypeRange<unsigned short>	{ public:	static __device__	unsigned short	min() { return 0; }			static __device__	unsigned short	max() {	return USHRT_MAX;	}};
	template<>				class DataTypeRange<unsigned int>	{ public:	static __device__	unsigned int	min() { return 0; }			static __device__	unsigned int	max() {	return UINT_MAX;	}};
	template<>				class DataTypeRange<float>			{ public:	static __device__	float			min() { return FLT_MIN; }	static __device__	float			max() {	return FLT_MAX;		}};
	template<>				class DataTypeRange<double>			{ public:	static __device__	double			min() { return DBL_MIN; }	static __device__	double			max() {	return DBL_MAX;		}};

	template <typename T>	class TypeHelper					{ };
	template<>				class TypeHelper<unsigned char>		{ public:	typedef unsigned int	type32; };
	template<>				class TypeHelper<unsigned short>	{ public:	typedef unsigned int	type32; };
	template<>				class TypeHelper<unsigned int>		{ public:	typedef unsigned int	type32; };
	template<>				class TypeHelper<float>				{ public:	typedef float			type32; };
	template<>				class TypeHelper<double>			{ public:	typedef float			type32; };

}
