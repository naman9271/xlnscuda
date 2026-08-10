// 16-bit XLNS CUDA batch host wrappers.
//
// Include xlnscpp/xlns16.cpp and xlns16d.cu before this file. The raw kernels
// are file-local so callers use the wrapper API instead of choosing launches.

#include <stddef.h>

#ifndef XLNSCUDA_BATCH_THREADS
#define XLNSCUDA_BATCH_THREADS 256
#endif

static inline unsigned xlns16d_batch_blocks(size_t n)
{
	size_t blocks = (n + XLNSCUDA_BATCH_THREADS - 1) / XLNSCUDA_BATCH_THREADS;
	if (blocks == 0)
		blocks = 1;
	if (blocks > 65535)
		blocks = 65535;
	return (unsigned)blocks;
}

static __global__ void xlns16d_batch_from_float_kernel(const float *src, xlns16 *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_from_float(src[i]);
}

static __global__ void xlns16d_batch_to_float_kernel(const xlns16 *src, float *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_to_float(src[i]);
}

static __global__ void xlns16d_batch_mul_kernel(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_mul(a[i], b[i]);
}

static __global__ void xlns16d_batch_add_kernel(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_add(a[i], b[i]);
}

static __global__ void xlns16d_batch_sub_kernel(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_sub(a[i], b[i]);
}

static __global__ void xlns16d_batch_div_kernel(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                                size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_div(a[i], b[i]);
}

static __global__ void xlns16d_batch_scale_kernel(const xlns16 *a, xlns16 scalar, xlns16 *dst,
                                                  size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16d_mul(a[i], scalar);
}

static __global__ void xlns16d_batch_neg_kernel(const xlns16 *src, xlns16 *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16_neg(src[i]);
}

static __global__ void xlns16d_batch_abs_kernel(const xlns16 *src, xlns16 *dst, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (; i < n; i += stride)
		dst[i] = xlns16_abs(src[i]);
}

inline cudaError_t xlns16d_batch_from_float_async(const float *src, xlns16 *dst, size_t n,
                                                  cudaStream_t stream)
{
	xlns16d_batch_from_float_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_to_float_async(const xlns16 *src, float *dst, size_t n,
                                                cudaStream_t stream)
{
	xlns16d_batch_to_float_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_mul_async(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns16d_batch_mul_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_add_async(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns16d_batch_add_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_sub_async(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns16d_batch_sub_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_div_async(const xlns16 *a, const xlns16 *b, xlns16 *dst,
                                           size_t n, cudaStream_t stream)
{
	xlns16d_batch_div_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, b, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_scale_async(const xlns16 *a, xlns16 scalar, xlns16 *dst,
                                             size_t n, cudaStream_t stream)
{
	xlns16d_batch_scale_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		a, scalar, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_neg_async(const xlns16 *src, xlns16 *dst, size_t n,
                                           cudaStream_t stream)
{
	xlns16d_batch_neg_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_abs_async(const xlns16 *src, xlns16 *dst, size_t n,
                                           cudaStream_t stream)
{
	xlns16d_batch_abs_kernel<<<xlns16d_batch_blocks(n), XLNSCUDA_BATCH_THREADS, 0, stream>>>(
		src, dst, n);
	return cudaGetLastError();
}

inline cudaError_t xlns16d_batch_from_float(const float *src, xlns16 *dst, size_t n)
{
	return xlns16d_batch_from_float_async(src, dst, n, 0);
}

inline cudaError_t xlns16d_batch_to_float(const xlns16 *src, float *dst, size_t n)
{
	return xlns16d_batch_to_float_async(src, dst, n, 0);
}

inline cudaError_t xlns16d_batch_mul(const xlns16 *a, const xlns16 *b, xlns16 *dst, size_t n)
{
	return xlns16d_batch_mul_async(a, b, dst, n, 0);
}

inline cudaError_t xlns16d_batch_add(const xlns16 *a, const xlns16 *b, xlns16 *dst, size_t n)
{
	return xlns16d_batch_add_async(a, b, dst, n, 0);
}

inline cudaError_t xlns16d_batch_sub(const xlns16 *a, const xlns16 *b, xlns16 *dst, size_t n)
{
	return xlns16d_batch_sub_async(a, b, dst, n, 0);
}

inline cudaError_t xlns16d_batch_div(const xlns16 *a, const xlns16 *b, xlns16 *dst, size_t n)
{
	return xlns16d_batch_div_async(a, b, dst, n, 0);
}

inline cudaError_t xlns16d_batch_scale(const xlns16 *a, xlns16 scalar, xlns16 *dst, size_t n)
{
	return xlns16d_batch_scale_async(a, scalar, dst, n, 0);
}

inline cudaError_t xlns16d_batch_neg(const xlns16 *src, xlns16 *dst, size_t n)
{
	return xlns16d_batch_neg_async(src, dst, n, 0);
}

inline cudaError_t xlns16d_batch_abs(const xlns16 *src, xlns16 *dst, size_t n)
{
	return xlns16d_batch_abs_async(src, dst, n, 0);
}
