#pragma once

#include <string>
#include <stdexcept>

namespace gpu {

	class gpu_exception : public std::runtime_error {
	public:
		gpu_exception(std::string msg) throw ()					: runtime_error(msg)							{	}
		gpu_exception(const char *msg) throw ()					: runtime_error(msg)							{	}
		gpu_exception() throw ()								: runtime_error("GPU exception")				{	}
	};

	class gpu_bad_alloc : public gpu_exception {
	public:
		gpu_bad_alloc(std::string msg) throw ()					: gpu_exception(msg)							{	}
		gpu_bad_alloc(const char *msg) throw ()					: gpu_exception(msg)							{	}
		gpu_bad_alloc() throw ()								: gpu_exception("GPU exception")				{	}
	};

	void raiseException(std::string file, int line, std::string message);

	template <typename T>
	size_t deviceTypeSize();

	template <typename T>
	T deviceTypeMax();

	template <typename T>
	T deviceTypeMin();

	inline unsigned int divup(unsigned int num, unsigned int denom) {
		return (num + denom - 1) / denom;
	}

	unsigned int calcNChunk(size_t n, size_t group_size, size_t max_size=1000*1000);
	unsigned int calcColsChunk(size_t width, size_t height, size_t group_size_x, size_t max_size=1000*1000);
	unsigned int calcRowsChunk(size_t width, size_t height, size_t group_size_y, size_t max_size=1000*1000);
	unsigned int calcZSlicesChunk(size_t x, size_t y, size_t z, size_t group_size_z, size_t max_size=1000*1000);
}

#define GPU_CHECKED_VERBOSE(x, message)	if (!(x)) {gpu::raiseException(__FILE__, __LINE__, message);}
#define GPU_CHECKED(x)					if (!(x)) {gpu::raiseException(__FILE__, __LINE__, "");}
