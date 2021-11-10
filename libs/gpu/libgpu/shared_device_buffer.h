#pragma once

#include <cstddef>
#include "shared_host_buffer.h"

typedef struct _cl_mem *cl_mem;

namespace gpu {

class shared_device_buffer {
public:
	shared_device_buffer();
	~shared_device_buffer();
	shared_device_buffer(const shared_device_buffer &other, size_t offset = 0);
	shared_device_buffer &operator= (const shared_device_buffer &other);

	void			swap(shared_device_buffer &other);
	void			reset();
	size_t			size() const;
	void			resize(size_t size);
	void			grow(size_t size, float reserveMultiplier=1.1f);
	bool 			isNull() const;

	void *			cuptr() const;
	cl_mem			clmem() const;
	size_t			cloffset() const;

	void 			write(const void *data, size_t size);
	void			write(const shared_device_buffer &buffer, size_t size);
	void			write(const shared_host_buffer &buffer, size_t size);
	void			write2D(size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);

	void			read(void *data, size_t size, size_t offset = 0) const;
	void 			read2D(size_t spitch, void *dst, size_t dpitch, size_t width, size_t height) const;

	void 			copyTo(shared_device_buffer &that, size_t size) const;

	static shared_device_buffer create(size_t size);

protected:
	void	incref();
	void	decref();

	unsigned char *	buffer_;
	void *			data_;
	int				type_;
	size_t			size_;
	size_t			offset_;
};

template <typename T>
class shared_device_buffer_typed : public shared_device_buffer {
public:
	shared_device_buffer_typed() : shared_device_buffer() {}
	shared_device_buffer_typed(const shared_device_buffer_typed &other, size_t offset) : shared_device_buffer(other, offset * sizeof(T)) {}
	explicit shared_device_buffer_typed(const shared_device_buffer &other) : shared_device_buffer(other) {}

	size_t			number() const;

	void			resizeN(size_t number);
	void			growN(size_t number, float reserveMultiplier=1.1f);

	T *				cuptr() const;

	void 			writeN(const T* data, size_t number);

	void			readN(T* data, size_t number, size_t offset = 0) const;

	void			copyToN(shared_device_buffer_typed<T> &that, size_t number) const;

	static shared_device_buffer_typed<T> createN(size_t number);
};

typedef shared_device_buffer						gpu_mem_any;

typedef shared_device_buffer_typed<int8_t>			gpu_mem_8i;
typedef shared_device_buffer_typed<int16_t>			gpu_mem_16i;
typedef shared_device_buffer_typed<int32_t>			gpu_mem_32i;
typedef shared_device_buffer_typed<uint8_t>			gpu_mem_8u;
typedef shared_device_buffer_typed<uint16_t>		gpu_mem_16u;
typedef shared_device_buffer_typed<uint32_t>		gpu_mem_32u;
typedef shared_device_buffer_typed<float>			gpu_mem_32f;
typedef shared_device_buffer_typed<double>			gpu_mem_64f;

#define gpu_mem shared_device_buffer_typed

}
