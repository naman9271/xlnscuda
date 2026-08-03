#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define xlns32_ideal
#include "../xlnscpp/xlns32.cpp"
#include "../xlns32d.cu"

#define CHECK_CUDA(call) \
	do { \
		cudaError_t err = (call); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA error %s:%d: %s\n", \
				__FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(1); \
		} \
	} while (0)

static int check_from_float(const char *name, const float *src, size_t n)
{
	xlns32 *expected = (xlns32 *)malloc(n * sizeof(expected[0]));
	xlns32 *got = (xlns32 *)malloc(n * sizeof(got[0]));
	float *d_src = 0;
	xlns32 *d_dst = 0;
	int wrong = 0;

	/*
	 * CPU fp2xlns32 accepts double, while CUDA fp2xlns32d accepts float.
	 * The intended parity path here is "the exact IEEE single input value,
	 * promoted to double for the CPU reference."
	 */
	for (size_t i = 0; i < n; i++)
		expected[i] = fp2xlns32((double)src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns32d_batch_from_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_dst, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n; i++) {
		if (expected[i] != got[i]) {
			printf("%s from_float mismatch i=%zu src=%e expected=%08x got=%08x\n",
			       name, i, src[i], expected[i], got[i]);
			wrong++;
		}
	}

	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dst));
	free(expected);
	free(got);
	return wrong;
}

static int check_to_float(const char *name, const xlns32 *src, size_t n)
{
	float *expected = (float *)malloc(n * sizeof(expected[0]));
	float *got = (float *)malloc(n * sizeof(got[0]));
	xlns32 *d_src = 0;
	float *d_dst = 0;
	int wrong = 0;

	for (size_t i = 0; i < n; i++)
		expected[i] = xlns322fp(src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns32d_batch_to_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_dst, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n; i++) {
		float tol = fmaxf(1.0e-6f, fabsf(expected[i]) * 1.0e-6f);
		if (fabsf(expected[i] - got[i]) > tol) {
			printf("%s to_float mismatch i=%zu src=%08x expected=%e got=%e\n",
			       name, i, src[i], expected[i], got[i]);
			wrong++;
		}
	}

	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dst));
	free(expected);
	free(got);
	return wrong;
}

static int check_empty(void)
{
	xlns32d_batch_from_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	xlns32d_batch_to_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	return 0;
}

static int check_case(const char *name, const float *src, size_t n)
{
	xlns32 *lns = (xlns32 *)malloc(n * sizeof(lns[0]));
	int wrong = 0;

	for (size_t i = 0; i < n; i++)
		lns[i] = fp2xlns32((double)src[i]);

	wrong += check_from_float(name, src, n);
	wrong += check_to_float(name, lns, n);
	free(lns);
	return wrong;
}

int main(void)
{
	const float one[] = { -2.5f };
	const float odd[] = { -3.0f, -0.5f, 0.0f, 0.25f, 7.0f };
	const float pow2[] = {
		-8.0f, -4.0f, -1.0f, -1.0e-40f,
		 0.0f,  1.0e-40f,  1.0f,  FLT_MAX
	};
	int wrong = 0;

	wrong += check_empty();
	wrong += check_case("one", one, sizeof(one) / sizeof(one[0]));
	wrong += check_case("odd", odd, sizeof(odd) / sizeof(odd[0]));
	wrong += check_case("pow2_extremes", pow2, sizeof(pow2) / sizeof(pow2[0]));

	printf("chkxlns32d_batch_convert %s (%d wrong)\n", wrong ? "FAIL" : "PASS", wrong);
	return wrong ? 1 : 0;
}
