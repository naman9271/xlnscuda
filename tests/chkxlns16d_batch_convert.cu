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

#define FROM_FLOAT_MAX_ERROR_PERCENT 1.0f
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
	xlns16 *expected = (xlns16 *)malloc(n * sizeof(expected[0]));
	xlns16 *got = (xlns16 *)malloc(n * sizeof(got[0]));
	float *d_src = 0;
	xlns16 *d_dst = 0;
	int failures = 0;

	for (size_t i = 0; i < n; i++)
		expected[i] = fp2xlns16((double)src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns16d_batch_from_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_dst, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	printf("\n[%s from_float]\n", name);
	printf("idx | src          | CPU bits | GPU bits | CPU fp       | GPU fp       | err %%     | status\n");
	printf("----|--------------|----------|----------|--------------|--------------|-----------|--------\n");
	for (size_t i = 0; i < n; i++) {
		float expected_fp = xlns162fp(expected[i]);
		float got_fp = xlns162fp(got[i]);
		float err = percent_error(expected_fp, got_fp);
		int bit_same = expected[i] == got[i];
		int fail = err > FROM_FLOAT_MAX_ERROR_PERCENT;
		if (!bit_same) (*bit_diffs)++;
		if (fail) failures++;
		printf("%3zu | %+12.5e | %04x     | %04x     | %+12.5e | %+12.5e | %9.6f | %s\n",
		       i, src[i], expected[i], got[i], expected_fp, got_fp, err,
		       fail ? "FAIL" : (bit_same ? "OK" : "ROUND"));
	}

	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dst));
	free(expected);
	free(got);
	return failures;
}

static int check_to_float(const char *name, const xlns16 *src, size_t n)
{
	float *expected = (float *)malloc(n * sizeof(expected[0]));
	float *got = (float *)malloc(n * sizeof(got[0]));
	xlns16 *d_src = 0;
	float *d_dst = 0;
	int failures = 0;

	for (size_t i = 0; i < n; i++)
		expected[i] = xlns162fp(src[i]);

	CHECK_CUDA(cudaMalloc((void **)&d_src, n * sizeof(src[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_dst, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_src, src, n * sizeof(src[0]), cudaMemcpyHostToDevice));

	xlns16d_batch_to_float_kernel<<<(unsigned)((n + 31) / 32), 32>>>(d_src, d_dst, n);
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
		printf("%3zu | %04x     | %+12.5e | %+12.5e | %9.6f | %s\n",
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
	xlns16d_batch_from_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	xlns16d_batch_to_float_kernel<<<1, 32>>>(0, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	printf("empty kernels completed\n");
	return 0;
}

static int check_case(const char *name, const float *src, size_t n, int *bit_diffs)
{
	xlns16 *lns = (xlns16 *)malloc(n * sizeof(lns[0]));
	int failures = 0;

	for (size_t i = 0; i < n; i++)
		lns[i] = fp2xlns16((double)src[i]);

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

	printf("=== xlns16d batch conversion CPU vs GPU ===\n");
	failures += check_empty();
	failures += check_case("one", one, sizeof(one) / sizeof(one[0]), &bit_diffs);
	failures += check_case("odd", odd, sizeof(odd) / sizeof(odd[0]), &bit_diffs);
	failures += check_case("pow2_extremes", pow2, sizeof(pow2) / sizeof(pow2[0]), &bit_diffs);

	printf("\nchkxlns16d_batch_convert %s (%d failures, %d tolerated bit differences)\n",
	       failures ? "FAIL" : "PASS", failures, bit_diffs);
	return failures ? 1 : 0;
}
