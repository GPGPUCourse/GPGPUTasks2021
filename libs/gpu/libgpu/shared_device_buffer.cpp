#include "shared_device_buffer.h"
#include "context.h"
#include <libgpu/utils.h>
#include <algorithm>
#include <stdexcept>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace gpu {

shared_device_buffer::shared_device_buffer()
{
	buffer_	= 0;
	data_	= 0;
	type_	= Context::TypeUndefined;
	size_	= 0;
	offset_	= 0;
}

shared_device_buffer::~shared_device_buffer()
{
	decref();
}

shared_device_buffer::shared_device_buffer(const shared_device_buffer &other, size_t offset)
{
	buffer_	= other.buffer_;
	data_	= other.data_;
	type_	= other.type_;
	size_	= other.size_;
	offset_	= other.offset_ + offset;
	incref();
}

shared_device_buffer &shared_device_buffer::operator= (const shared_device_buffer &other)
{
	if (this != &other) {
		decref();
		buffer_	= other.buffer_;
		data_	= other.data_;
		type_	= other.type_;
		size_	= other.size_;
		offset_	= other.offset_;
		incref();
	}

	return *this;
}

void shared_device_buffer::swap(shared_device_buffer &other)
{
	std::swap(buffer_,	other.buffer_);
	std::swap(data_,	other.data_);
	std::swap(type_,	other.type_);
	std::swap(size_,	other.size_);
	std::swap(offset_,	other.offset_);
}

void shared_device_buffer::incref()
{
	if (!buffer_)
		return;

#if defined(_WIN64)
	InterlockedIncrement64((LONGLONG *) buffer_);
#elif defined(_WIN32)
	InterlockedIncrement((LONG *) buffer_);
#else
	__sync_add_and_fetch((long long *) buffer_, 1);
#endif
}

void shared_device_buffer::decref()
{
	if (!buffer_)
		return;

	long long count = 0;

#if defined(_WIN64)
	count = InterlockedDecrement64((LONGLONG *) buffer_);
#elif defined(_WIN32)
	count = InterlockedDecrement((LONG *) buffer_);
#else
	count = __sync_sub_and_fetch((long long *) buffer_, 1);
#endif

	if (!count) {
		switch (type_) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			cudaFree(data_);
			break;
#endif
		case Context::TypeOpenCL:
			clReleaseMemObject((cl_mem) data_);
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
		}

		delete [] buffer_;
	}

	buffer_ = 0;
	data_	= 0;
	type_	= Context::TypeUndefined;
	size_	= 0;
	offset_	= 0;
}

shared_device_buffer shared_device_buffer::create(size_t size)
{
	shared_device_buffer res;
	res.resize(size);
	return res;
}

void *shared_device_buffer::cuptr() const
{
	if (type_ == Context::TypeOpenCL)
		throw gpu_exception("GPU buffer type mismatch");

	return (char *) data_ + offset_;
}

cl_mem shared_device_buffer::clmem() const
{
	if (type_ == Context::TypeCUDA)
		throw gpu_exception("GPU buffer type mismatch");

	return (cl_mem) data_;
}

size_t shared_device_buffer::cloffset() const
{
	if (type_ == Context::TypeCUDA)
		throw gpu_exception("GPU buffer type mismatch");

	return offset_;
}

size_t shared_device_buffer::size() const
{
	return size_;
}

bool shared_device_buffer::isNull() const
{
	return data_ == NULL;
}

void shared_device_buffer::reset()
{
	decref();
}

void shared_device_buffer::resize(size_t size)
{
	if (size == size_)
		return;

	decref();

	Context context;
	Context::Type type = context.type();

	switch (type) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL( cudaMalloc(&data_, size) );
		break;
#endif
	case Context::TypeOpenCL:
		data_ = context.cl()->createBuffer(CL_MEM_READ_WRITE, size);
		break;
	default:
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}

	buffer_	= new unsigned char [8];
	* (long long *) buffer_ = 0;
	incref();

	type_	= type;
	size_	= size;
	offset_	= 0;
}

void shared_device_buffer::grow(size_t size, float reserveMultiplier)
{
	if (size > size_)
		resize((size_t) (size * reserveMultiplier));
}

void shared_device_buffer::write(const void *data, size_t size)
{
	if (size == 0)
		return;

	if (size > size_)
		throw gpu_exception("Too many data for this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(cuptr(), data, size, cudaMemcpyHostToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->writeBuffer((cl_mem) data_, CL_TRUE, offset_, size, data);
		break;
	default:
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

void shared_device_buffer::write(const shared_device_buffer &buffer, size_t size)
{
	if (!size)
		return;

	if (size > size_)
		throw gpu_exception("Too many data for this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(cuptr(), buffer.cuptr(), size, cudaMemcpyDeviceToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->copyBuffer(buffer.clmem(), clmem(), buffer.cloffset(), cloffset(), size);
		break;
	default:
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

void shared_device_buffer::write(const shared_host_buffer &buffer, size_t size)
{
	if (!size)
		return;

	if (size > size_)
		throw gpu_exception("Too many data for this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(cuptr(), buffer.get(), size, cudaMemcpyHostToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->writeBuffer((cl_mem) data_, CL_TRUE, offset_, size, buffer.get());
		break;
	default:
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

void shared_device_buffer::write2D(size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
	if (spitch == width && dpitch == width) {
		write(src, width * height);
		return;
	}

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy2D(cuptr(), dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
		break;
#endif
	case Context::TypeOpenCL:
		{
			size_t buffer_origin[3] = { offset_, 0, 0 };
			size_t host_origin[3] = { 0, 0, 0 };
			size_t region[3] = { width, height, 1 };
			context.cl()->writeBufferRect((cl_mem) data_, CL_TRUE, buffer_origin, host_origin, region, dpitch, 0, spitch, 0, src);
		}
		break;
	default:
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

void shared_device_buffer::read(void *data, size_t size, size_t offset) const
{
	if (size == 0)
		return;
	if (size > size_)
		throw gpu_exception("Not enough data in this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
		CUDA_SAFE_CALL(cudaMemcpy(data, (char *) cuptr() + offset, size, cudaMemcpyDeviceToHost));
		break;
#endif
	case Context::TypeOpenCL:
		context.cl()->readBuffer((cl_mem) data_, CL_TRUE, offset_ + offset, size, data);
		break;
	default:
		gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

void shared_device_buffer::read2D(size_t spitch, void *dst, size_t dpitch, size_t width, size_t height) const
{
	if (spitch == width && dpitch == width) {
		read(dst, width * height);
		return;
	}

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaMemcpy2D(dst, dpitch, cuptr(), spitch, width, height, cudaMemcpyDeviceToHost));
			break;
#endif
		case Context::TypeOpenCL:
		{
			size_t buffer_origin[3] = { offset_, 0, 0 };
			size_t host_origin[3] = { 0, 0, 0 };
			size_t region[3] = { width, height, 1 };
			context.cl()->readBufferRect((cl_mem) data_, CL_TRUE, buffer_origin, host_origin, region, spitch, 0, dpitch, 0, dst);
		}
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

void shared_device_buffer::copyTo(shared_device_buffer &that, size_t size) const
{
	if (size == 0)
		return;
	if (size > size_)
		throw gpu_exception("Not enough data in this device buffer: " + to_string(size) + " > " + to_string(size_));

	Context context;
	switch (context.type()) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaMemcpy((char *) that.cuptr(), (char *) cuptr(), size, cudaMemcpyDeviceToDevice));
			break;
#endif
		case Context::TypeOpenCL:
			context.cl()->copyBuffer(clmem(), that.clmem(), offset_, that.offset_, size);
			break;
		default:
			gpu::raiseException(__FILE__, __LINE__, "No GPU context!");
	}
}

template <typename T>
shared_device_buffer_typed<T> shared_device_buffer_typed<T>::createN(size_t number)
{
	shared_device_buffer_typed<T> res;
	res.resizeN(number);
	return res;
}

template <typename T>
size_t shared_device_buffer_typed<T>::number() const
{
	return size_ / sizeof(T);
}

template <typename T>
void shared_device_buffer_typed<T>::resizeN(size_t number)
{
	this->resize(number * sizeof(T));
}

template <typename T>
void shared_device_buffer_typed<T>::growN(size_t number, float reserveMultiplier)
{
	this->grow(number * sizeof(T), reserveMultiplier);
}

template <typename T>
T *shared_device_buffer_typed<T>::cuptr() const
{
	return (T *) shared_device_buffer::cuptr();
}

template<typename T>
void shared_device_buffer_typed<T>::writeN(const T* data, size_t number) {
	this->write(data, number * sizeof(T));
}

template<typename T>
void shared_device_buffer_typed<T>::readN(T* data, size_t number, size_t offset) const
{
	this->read(data, number * sizeof(T), offset * sizeof(T));
}

template<typename T>
void shared_device_buffer_typed<T>::copyToN(shared_device_buffer_typed<T> &that, size_t number) const
{
	this->copyTo(that, number * sizeof(T));
}

template class shared_device_buffer_typed<int8_t>;
template class shared_device_buffer_typed<int16_t>;
template class shared_device_buffer_typed<int32_t>;
template class shared_device_buffer_typed<uint8_t>;
template class shared_device_buffer_typed<uint16_t>;
template class shared_device_buffer_typed<uint32_t>;
template class shared_device_buffer_typed<float>;
template class shared_device_buffer_typed<double>;

}
