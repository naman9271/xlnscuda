// 32-bit XLNS CUDA batch host wrappers.
//
// Include xlnscpp/xlns32.cpp and xlns32d.cu before this file. The raw kernels
// are file-local so callers use the wrapper API instead of choosing launches.

#include <stddef.h>

#ifndef XLNSCUDA_BATCH_THREADS
#define XLNSCUDA_BATCH_THREADS 256
#endif

static inline unsigned xlns32d_batch_blocks(size_t n)
{
	size_t blocks = (n + XLNSCUDA_BATCH_THREADS - 1) / XLNSCUDA_BATCH_THREADS;
	if (blocks == 0)
		blocks = 1;
	if (blocks > 65535)
		blocks = 65535;
	return (unsigned)blocks;
}

static __global__ void xlns32d_batch_from_float_kernel(const float *src, xlns32 *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_from_float(src[i]);
}

static __global__ void xlns32d_batch_to_float_kernel(const xlns32 *src, float *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_to_float(src[i]);
}

static __global__ void xlns32d_batch_mul_kernel(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_mul(a[i], b[i]);
}

static __global__ void xlns32d_batch_add_kernel(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_add(a[i], b[i]);
}

static __global__ void xlns32d_batch_sub_kernel(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_sub(a[i], b[i]);
}

static __global__ void xlns32d_batch_div_kernel(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_div(a[i], b[i]);
}

static __global__ void xlns32d_batch_scale_kernel(const xlns32 *a, xlns32 scalar, xlns32 *dst,
                                                  size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32d_mul(a[i], scalar);
}

static __global__ void xlns32d_batch_neg_kernel(const xlns32 *src, xlns32 *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32_neg(src[i]);
}

static __global__ void xlns32d_batch_abs_kernel(const xlns32 *src, xlns32 *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns32_abs(src[i]);
}

inline cudaError_t xlns32d_batch_from_float_async(const float *src, xlns32 *dst, size_t n,
                                                  cudaStream_t stream)
{
	xlns32d_batch_from_float_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_to_float_async(const xlns32 *src, float *dst, size_t n,
                                                cudaStream_t stream)
{
	xlns32d_batch_to_float_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_mul_async(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns32d_batch_mul_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_add_async(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns32d_batch_add_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_sub_async(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns32d_batch_sub_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_div_async(const xlns32 *a, const xlns32 *b, xlns32 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns32d_batch_div_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_scale_async(const xlns32 *a, xlns32 scalar, xlns32 *dst,
                                             size_t n, cudaStream_t stream)
{
	xlns32d_batch_scale_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, scalar, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_neg_async(const xlns32 *src, xlns32 *dst, size_t n,
                                           cudaStream_t stream)
{
	xlns32d_batch_neg_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_abs_async(const xlns32 *src, xlns32 *dst, size_t n,
                                           cudaStream_t stream)
{
	xlns32d_batch_abs_kernel<<<xlns32d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns32d_batch_from_float(const float *src, xlns32 *dst, size_t n)
{
	return xlns32d_batch_from_float_async(src, dst, n, 0);
}

inline cudaError_t xlns32d_batch_to_float(const xlns32 *src, float *dst, size_t n)
{
	return xlns32d_batch_to_float_async(src, dst, n, 0);
}

inline cudaError_t xlns32d_batch_mul(const xlns32 *a, const xlns32 *b, xlns32 *dst, size_t n)
{
	return xlns32d_batch_mul_async(a, b, dst, n, 0);
}

inline cudaError_t xlns32d_batch_add(const xlns32 *a, const xlns32 *b, xlns32 *dst, size_t n)
{
	return xlns32d_batch_add_async(a, b, dst, n, 0);
}

inline cudaError_t xlns32d_batch_sub(const xlns32 *a, const xlns32 *b, xlns32 *dst, size_t n)
{
	return xlns32d_batch_sub_async(a, b, dst, n, 0);
}

inline cudaError_t xlns32d_batch_div(const xlns32 *a, const xlns32 *b, xlns32 *dst, size_t n)
{
	return xlns32d_batch_div_async(a, b, dst, n, 0);
}

inline cudaError_t xlns32d_batch_scale(const xlns32 *a, xlns32 scalar, xlns32 *dst, size_t n)
{
	return xlns32d_batch_scale_async(a, scalar, dst, n, 0);
}

inline cudaError_t xlns32d_batch_neg(const xlns32 *src, xlns32 *dst, size_t n)
{
	return xlns32d_batch_neg_async(src, dst, n, 0);
}

inline cudaError_t xlns32d_batch_abs(const xlns32 *src, xlns32 *dst, size_t n)
{
	return xlns32d_batch_abs_async(src, dst, n, 0);
}
