#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define xlns16_alt
#include "../xlnscpp/xlns16.cpp"
#include "../xlns16d.cu"

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
	xlns16 *expected = (xlns16 *)malloc(n * sizeof(expected[0]));
	xlns16 *got = (xlns16 *)malloc(n * sizeof(got[0]));
	float *d_src = 0;
	xlns16 *d_dst = 0;
	int wrong = 0;

	for (size_t i = 0; i < n; i++)
		expected[i] = fp2xlns16((double)src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns16d_batch_from_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_dst, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n; i++) {
		if (expected[i] != got[i]) {
			printf("%s from_float mismatch i=%zu src=%e expected=%04x got=%04x\n",
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

static int check_to_float(const char *name, const xlns16 *src, size_t n)
{
	float *expected = (float *)malloc(n * sizeof(expected[0]));
	float *got = (float *)malloc(n * sizeof(got[0]));
	xlns16 *d_src = 0;
	float *d_dst = 0;
	int wrong = 0;

	for (size_t i = 0; i < n; i++)
		expected[i] = xlns162fp(src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns16d_batch_to_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_dst, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n; i++) {
		float tol = fmaxf(1.0e-6f, fabsf(expected[i]) * 1.0e-6f);
		if (fabsf(expected[i] - got[i]) > tol) {
			printf("%s to_float mismatch i=%zu src=%04x expected=%e got=%e\n",
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
	xlns16d_batch_from_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	xlns16d_batch_to_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	return 0;
}

static int check_case(const char *name, const float *src, size_t n)
{
	xlns16 *lns = (xlns16 *)malloc(n * sizeof(lns[0]));
	int wrong = 0;

	for (size_t i = 0; i < n; i++)
		lns[i] = fp2xlns16((double)src[i]);

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

	printf("chkxlns16d_batch_convert %s (%d wrong)\n", wrong ? "FAIL" : "PASS", wrong);
	return wrong ? 1 : 0;
}
