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

#define NUM_VALUES 8
#define NUM_REDUCE_CASES 5
#define NUM_REDUCE_LNS_RESULTS 4
#define NUM_SCALAR_RESULTS 9
#define LNS_MAX_ERROR_PERCENT 0.001f
#define F32_MAX_ERROR_PERCENT 0.001f

static const char *reduce_lns_names[NUM_REDUCE_LNS_RESULTS] = {
	"sum", "vec_dot", "max_array", "min_array"
};

static const char *scalar_names[NUM_SCALAR_RESULTS] = {
	"sigmoid", "tanh", "silu", "gelu", "exp", "log", "exp2", "log2", "pow"
};

static float percent_error(float expected, float got)
{
	if (expected == got) return 0.0f;
	if (isinf(expected) || isinf(got)) return INFINITY;
	float denom = fmaxf(fabsf(expected), FLT_MIN);
	return fabsf(expected - got) / denom * 100.0f;
}

__global__ void xlns32d_reduce_kernel(const xlns32 *a, const xlns32 *b,
				      const float *fa, const float *fb,
				      const size_t *sizes,
				      xlns32 *lns_results,
				      float *f32_results,
				      int num_cases)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_cases) return;

	size_t n = sizes[i];
	int base = i * NUM_REDUCE_LNS_RESULTS;
	lns_results[base + 0] = xlns32d_sum(a, n);
	lns_results[base + 1] = xlns32d_vec_dot(a, b, n);
	lns_results[base + 2] = xlns32d_max_array(a, n);
	lns_results[base + 3] = xlns32d_min_array(a, n);
	f32_results[i] = xlns32d_vec_dot_f32(fa, fb, n);
}

__global__ void xlns32d_scalar_kernel(const xlns32 *x, const xlns32 *base,
				      const xlns32 *exponent,
				      xlns32 *results, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	int out = i * NUM_SCALAR_RESULTS;
	results[out + 0] = xlns32d_sigmoid(x[i]);
	results[out + 1] = xlns32d_tanh(x[i]);
	results[out + 2] = xlns32d_silu(x[i]);
	results[out + 3] = xlns32d_gelu(x[i]);
	results[out + 4] = xlns32d_exp(x[i]);
	results[out + 5] = xlns32d_log(x[i]);
	results[out + 6] = xlns32d_exp2(x[i]);
	results[out + 7] = xlns32d_log2(x[i]);
	results[out + 8] = xlns32d_pow(base[i], exponent[i]);
}

static int check_lns_numeric(const char *group, const char *name, int case_idx,
			     xlns32 expected, xlns32 got)
{
	float expected_fp = xlns322fp(expected);
	float got_fp = xlns322fp(got);
	float err = percent_error(expected_fp, got_fp);
	if (err <= LNS_MAX_ERROR_PERCENT) return 0;
	printf("%s %s mismatch case=%d expected=%08x got=%08x expected_fp=%+.8e got_fp=%+.8e err=%f%%\n",
	       group, name, case_idx, expected, got, expected_fp, got_fp, err);
	return 1;
}

static int check_f32_numeric(int case_idx, float expected, float got)
{
	float err = percent_error(expected, got);
	if (err <= F32_MAX_ERROR_PERCENT) return 0;
	printf("vec_dot_f32 mismatch case=%d expected=%+.8e got=%+.8e err=%f%%\n",
	       case_idx, expected, got, err);
	return 1;
}

static void print_reduce_table(const size_t *sizes,
			       const xlns32 *expected_lns, const xlns32 *got_lns,
			       const float *expected_f32, const float *got_f32)
{
	printf("\n=== xlns32d helper reductions CPU vs GPU ===\n");
	for (int f = 0; f < NUM_REDUCE_LNS_RESULTS; f++) {
		printf("\n[%s]\n", reduce_lns_names[f]);
		printf("case | n | CPU bits | GPU bits | CPU fp       | GPU fp\n");
		printf("-----|---|----------|----------|--------------|--------------\n");
		for (int i = 0; i < NUM_REDUCE_CASES; i++) {
			int idx = i * NUM_REDUCE_LNS_RESULTS + f;
			printf("%4d | %zu | %08x | %08x | %+12.5e | %+12.5e\n",
			       i, sizes[i], expected_lns[idx], got_lns[idx],
			       xlns322fp(expected_lns[idx]), xlns322fp(got_lns[idx]));
		}
	}

	printf("\n[vec_dot_f32]\n");
	printf("case | n | CPU fp       | GPU fp       | err %%\n");
	printf("-----|---|--------------|--------------|-----------\n");
	for (int i = 0; i < NUM_REDUCE_CASES; i++) {
		printf("%4d | %zu | %+12.5e | %+12.5e | %9.6f\n",
		       i, sizes[i], expected_f32[i], got_f32[i],
		       percent_error(expected_f32[i], got_f32[i]));
	}
}

static void print_scalar_table(const xlns32 *x,
			       const xlns32 *expected, const xlns32 *got)
{
	printf("\n=== xlns32d helper scalar CPU vs GPU ===\n");
	for (int f = 0; f < NUM_SCALAR_RESULTS; f++) {
		printf("\n[%s]\n", scalar_names[f]);
		printf("case | x bits   | x fp         | CPU bits | GPU bits | CPU fp       | GPU fp       | err %%\n");
		printf("-----|----------|--------------|----------|----------|--------------|--------------|-----------\n");
		for (int i = 0; i < NUM_VALUES; i++) {
			int idx = i * NUM_SCALAR_RESULTS + f;
			printf("%4d | %08x | %+12.5e | %08x | %08x | %+12.5e | %+12.5e | %9.6f\n",
			       i, x[i], xlns322fp(x[i]), expected[idx], got[idx],
			       xlns322fp(expected[idx]), xlns322fp(got[idx]),
			       percent_error(xlns322fp(expected[idx]), xlns322fp(got[idx])));
		}
	}
}

int main(void)
{
	const float input_fp[NUM_VALUES] = {
		-4.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f
	};
	const float b_fp[NUM_VALUES] = {
		0.5f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f
	};
	const float base_fp[NUM_VALUES] = {
		0.125f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 4.0f, 8.0f
	};
	const float exponent_fp[NUM_VALUES] = {
		3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 2.0f, 3.0f, 0.25f
	};
	const size_t sizes[NUM_REDUCE_CASES] = { 0, 1, 2, 5, NUM_VALUES };

	xlns32 h_a[NUM_VALUES];
	xlns32 h_b[NUM_VALUES];
	xlns32 h_base[NUM_VALUES];
	xlns32 h_exponent[NUM_VALUES];
	xlns32 expected_reduce[NUM_REDUCE_CASES * NUM_REDUCE_LNS_RESULTS];
	xlns32 got_reduce[NUM_REDUCE_CASES * NUM_REDUCE_LNS_RESULTS];
	float expected_f32[NUM_REDUCE_CASES];
	float got_f32[NUM_REDUCE_CASES];
	xlns32 expected_scalar[NUM_VALUES * NUM_SCALAR_RESULTS];
	xlns32 got_scalar[NUM_VALUES * NUM_SCALAR_RESULTS];

	for (int i = 0; i < NUM_VALUES; i++) {
		h_a[i] = fp2xlns32(input_fp[i]);
		h_b[i] = fp2xlns32(b_fp[i]);
		h_base[i] = fp2xlns32(base_fp[i]);
		h_exponent[i] = fp2xlns32(exponent_fp[i]);
	}

	for (int i = 0; i < NUM_REDUCE_CASES; i++) {
		size_t n = sizes[i];
		int base = i * NUM_REDUCE_LNS_RESULTS;
		expected_reduce[base + 0] = xlns32_sum(h_a, n);
		expected_reduce[base + 1] = xlns32_vec_dot(h_a, h_b, n);
		expected_reduce[base + 2] = xlns32_max_array(h_a, n);
		expected_reduce[base + 3] = xlns32_min_array(h_a, n);
		expected_f32[i] = xlns32_vec_dot_f32(input_fp, b_fp, n);
	}

	for (int i = 0; i < NUM_VALUES; i++) {
		int out = i * NUM_SCALAR_RESULTS;
		expected_scalar[out + 0] = xlns32_sigmoid(h_a[i]);
		expected_scalar[out + 1] = xlns32_tanh(h_a[i]);
		expected_scalar[out + 2] = xlns32_silu(h_a[i]);
		expected_scalar[out + 3] = xlns32_gelu(h_a[i]);
		expected_scalar[out + 4] = xlns32_exp(h_a[i]);
		expected_scalar[out + 5] = xlns32_log(h_a[i]);
		expected_scalar[out + 6] = xlns32_exp2(h_a[i]);
		expected_scalar[out + 7] = xlns32_log2(h_a[i]);
		expected_scalar[out + 8] = xlns32_pow(h_base[i], h_exponent[i]);
	}

	xlns32 *d_a = 0;
	xlns32 *d_b = 0;
	xlns32 *d_base = 0;
	xlns32 *d_exponent = 0;
	float *d_fa = 0;
	float *d_fb = 0;
	size_t *d_sizes = 0;
	xlns32 *d_reduce = 0;
	float *d_f32 = 0;
	xlns32 *d_scalar = 0;

	CHECK_CUDA(cudaMalloc((void **)&d_a, sizeof(h_a)));
	CHECK_CUDA(cudaMalloc((void **)&d_b, sizeof(h_b)));
	CHECK_CUDA(cudaMalloc((void **)&d_base, sizeof(h_base)));
	CHECK_CUDA(cudaMalloc((void **)&d_exponent, sizeof(h_exponent)));
	CHECK_CUDA(cudaMalloc((void **)&d_fa, sizeof(input_fp)));
	CHECK_CUDA(cudaMalloc((void **)&d_fb, sizeof(b_fp)));
	CHECK_CUDA(cudaMalloc((void **)&d_sizes, sizeof(sizes)));
	CHECK_CUDA(cudaMalloc((void **)&d_reduce, sizeof(got_reduce)));
	CHECK_CUDA(cudaMalloc((void **)&d_f32, sizeof(got_f32)));
	CHECK_CUDA(cudaMalloc((void **)&d_scalar, sizeof(got_scalar)));

	CHECK_CUDA(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_base, h_base, sizeof(h_base), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_exponent, h_exponent, sizeof(h_exponent), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_fa, input_fp, sizeof(input_fp), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_fb, b_fp, sizeof(b_fp), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_sizes, sizes, sizeof(sizes), cudaMemcpyHostToDevice));

	xlns32d_reduce_kernel<<<1, 32>>>(d_a, d_b, d_fa, d_fb, d_sizes,
					 d_reduce, d_f32, NUM_REDUCE_CASES);
	CHECK_CUDA(cudaGetLastError());
	xlns32d_scalar_kernel<<<1, 32>>>(d_a, d_base, d_exponent,
					 d_scalar, NUM_VALUES);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(got_reduce, d_reduce, sizeof(got_reduce), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(got_f32, d_f32, sizeof(got_f32), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(got_scalar, d_scalar, sizeof(got_scalar), cudaMemcpyDeviceToHost));

	print_reduce_table(sizes, expected_reduce, got_reduce, expected_f32, got_f32);
	print_scalar_table(h_a, expected_scalar, got_scalar);

	int wrong = 0;
	for (int i = 0; i < NUM_REDUCE_CASES; i++) {
		for (int j = 0; j < NUM_REDUCE_LNS_RESULTS; j++) {
			int idx = i * NUM_REDUCE_LNS_RESULTS + j;
			wrong += check_lns_numeric("reduce", reduce_lns_names[j], i,
						   expected_reduce[idx], got_reduce[idx]);
		}
		wrong += check_f32_numeric(i, expected_f32[i], got_f32[i]);
	}
	for (int i = 0; i < NUM_VALUES; i++) {
		for (int j = 0; j < NUM_SCALAR_RESULTS; j++) {
			int idx = i * NUM_SCALAR_RESULTS + j;
			wrong += check_lns_numeric("scalar", scalar_names[j], i,
						   expected_scalar[idx], got_scalar[idx]);
		}
	}

	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_base));
	CHECK_CUDA(cudaFree(d_exponent));
	CHECK_CUDA(cudaFree(d_fa));
	CHECK_CUDA(cudaFree(d_fb));
	CHECK_CUDA(cudaFree(d_sizes));
	CHECK_CUDA(cudaFree(d_reduce));
	CHECK_CUDA(cudaFree(d_f32));
	CHECK_CUDA(cudaFree(d_scalar));

	printf("\nchkxlns32d_helper_functions %s (%d wrong)\n", wrong ? "FAIL" : "PASS", wrong);
	return wrong ? 1 : 0;
}
