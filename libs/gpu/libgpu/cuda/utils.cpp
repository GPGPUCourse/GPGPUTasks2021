#ifdef CUDA_SUPPORT
#include "utils.h"
#include "cuda_api.h"

namespace cuda {

std::string formatError(cudaError_t code)
{
	return std::string(cudaGetErrorString(code)) + " (" + to_string(code) + ")";
}

}

#endif
