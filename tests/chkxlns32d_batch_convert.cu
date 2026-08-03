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

#define FROM_FLOAT_MAX_ERROR_PERCENT 0.001f
#define TO_FLOAT_MAX_ERROR_PERCENT 0.001f

static float percent_error(float expected, float got)
{
	if (expected == got) return 0.0f;
	if (isinf(expected) || isinf(got)) return INFINITY;
	float denom = fmaxf(fabsf(expected), FLT_MIN);
	return fabsf(expected - got) / denom * 100.0f;
}

static int check_from_float(const char *name, const float *src, size_t n, int *bit_diffs)
{
	xlns32 *expected = (xlns32 *)malloc(n * sizeof(expected[0]));
	xlns32 *got = (xlns32 *)malloc(n * sizeof(got[0]));
	float *d_src = 0;
	xlns32 *d_dst = 0;
	int failures = 0;

	/*
	 * CPU fp2xlns32 accepts double, while CUDA fp2xlns32d accepts float.
	 * The intended parity path here is the exact IEEE single input value,
	 * promoted to double for the CPU reference. CPU and GPU libm can still
	 * land on adjacent LNS codes, so numeric error is reported separately.
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

	printf("\n[%s from_float]\n", name);
	printf("idx | src          | CPU bits | GPU bits | CPU fp       | GPU fp       | err %%     | status\n");
	printf("----|--------------|----------|----------|--------------|--------------|-----------|--------\n");
	for (size_t i = 0; i < n; i++) {
		float expected_fp = xlns322fp(expected[i]);
		float got_fp = xlns322fp(got[i]);
		float err = percent_error(expected_fp, got_fp);
		int bit_same = expected[i] == got[i];
		int fail = err > FROM_FLOAT_MAX_ERROR_PERCENT;
		if (!bit_same) (*bit_diffs)++;
		if (fail) failures++;
		printf("%3zu | %+12.5e | %08x | %08x | %+12.5e | %+12.5e | %9.6f | %s\n",
		       i, src[i], expected[i], got[i], expected_fp, got_fp, err,
		       fail ? "FAIL" : (bit_same ? "OK" : "ROUND"));
	}

	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dst));
	free(expected);
	free(got);
	return failures;
}

static int check_to_float(const char *name, const xlns32 *src, size_t n)
{
	float *expected = (float *)malloc(n * sizeof(expected[0]));
	float *got = (float *)malloc(n * sizeof(got[0]));
	xlns32 *d_src = 0;
	float *d_dst = 0;
	int failures = 0;

	for (size_t i = 0; i < n; i++)
		expected[i] = xlns322fp(src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns32d_batch_to_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_dst, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	printf("\n[%s to_float]\n", name);
	printf("idx | src bits | CPU fp       | GPU fp       | err %%     | status\n");
	printf("----|----------|--------------|--------------|-----------|--------\n");
	for (size_t i = 0; i < n; i++) {
		float err = percent_error(expected[i], got[i]);
		int fail = err > TO_FLOAT_MAX_ERROR_PERCENT;
		if (fail) failures++;
		printf("%3zu | %08x | %+12.5e | %+12.5e | %9.6f | %s\n",
		       i, src[i], expected[i], got[i], err, fail ? "FAIL" : "OK");
	}

	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dst));
	free(expected);
	free(got);
	return failures;
}

static int check_empty(void)
{
	printf("\n[empty]\n");
	xlns32d_batch_from_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	xlns32d_batch_to_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	printf("empty kernels completed\n");
	return 0;
}

static int check_case(const char *name, const float *src, size_t n, int *bit_diffs)
{
	xlns32 *lns = (xlns32 *)malloc(n * sizeof(lns[0]));
	int failures = 0;

	for (size_t i = 0; i < n; i++)
		lns[i] = fp2xlns32((double)src[i]);

	failures += check_from_float(name, src, n, bit_diffs);
	failures += check_to_float(name, lns, n);
	free(lns);
	return failures;
}

int main(void)
{
	const float one[] = { -2.5f };
	const float odd[] = { -3.0f, -0.5f, 0.0f, 0.25f, 7.0f };
	const float pow2[] = {
		-8.0f, -4.0f, -1.0f, -1.0e-40f,
		 0.0f,  1.0e-40f,  1.0f,  FLT_MAX
	};
	int failures = 0;
	int bit_diffs = 0;

	printf("=== xlns32d batch conversion CPU vs GPU ===\n");
	failures += check_empty();
	failures += check_case("one", one, sizeof(one) / sizeof(one[0]), &bit_diffs);
	failures += check_case("odd", odd, sizeof(odd) / sizeof(odd[0]), &bit_diffs);
	failures += check_case("pow2_extremes", pow2, sizeof(pow2) / sizeof(pow2[0]), &bit_diffs);

	printf("\nchkxlns32d_batch_convert %s (%d failures, %d tolerated bit differences)\n",
	       failures ? "FAIL" : "PASS", failures, bit_diffs);
	return failures ? 1 : 0;
}
