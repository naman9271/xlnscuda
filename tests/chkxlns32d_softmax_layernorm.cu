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

#define NUM_CASES 4
#define MAX_N 5
#define NUM_SOFTMAX_RESULTS 4
#define NUM_LAYER_RESULTS 4
#define XLNS32_MAX_ERROR_PERCENT 0.01f

static const char *softmax_names[NUM_SOFTMAX_RESULTS] = {
	"softmax_exp", "softmax", "softmax_masked", "softmax_inplace"
};

static const char *layer_names[NUM_LAYER_RESULTS] = {
	"layernorm", "layernorm_gamma", "layernorm_beta", "layernorm_gamma_beta"
};

static float percent_error(float expected, float got)
{
	if (expected == got) return 0.0f;
	if (isinf(expected) || isinf(got)) return INFINITY;
	float denom = fmaxf(fabsf(expected), FLT_MIN);
	return fabsf(expected - got) / denom * 100.0f;
}

__global__ void xlns32d_softmax_layernorm_kernel(const xlns32 *a,
						 const xlns32 *mask,
						 const xlns32 *gamma,
						 const xlns32 *beta,
						 const size_t *sizes,
						 xlns32 scale,
						 float eps,
						 xlns32 *softmax_results,
						 xlns32 *layer_results,
						 int num_cases)
{
	int case_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (case_idx >= num_cases) return;

	size_t n = sizes[case_idx];
	const xlns32 *case_a = a + case_idx * MAX_N;
	const xlns32 *case_mask = mask + case_idx * MAX_N;
	const xlns32 *case_gamma = gamma + case_idx * MAX_N;
	const xlns32 *case_beta = beta + case_idx * MAX_N;
	xlns32 *soft_base = softmax_results + case_idx * NUM_SOFTMAX_RESULTS * MAX_N;
	xlns32 *layer_base = layer_results + case_idx * NUM_LAYER_RESULTS * MAX_N;

	xlns32d_softmax_exp(case_a, soft_base + 0 * MAX_N, n);
	xlns32d_softmax(case_a, soft_base + 1 * MAX_N, n, scale);
	xlns32d_softmax_masked(case_a, case_mask, soft_base + 2 * MAX_N, n, scale);

	for (size_t i = 0; i < n; i++) soft_base[3 * MAX_N + i] = case_a[i];
	xlns32d_softmax(soft_base + 3 * MAX_N, soft_base + 3 * MAX_N, n, scale);

	xlns32d_layernorm(case_a, layer_base + 0 * MAX_N, 0, 0, n, eps);
	xlns32d_layernorm(case_a, layer_base + 1 * MAX_N, case_gamma, 0, n, eps);
	xlns32d_layernorm(case_a, layer_base + 2 * MAX_N, 0, case_beta, n, eps);
	xlns32d_layernorm(case_a, layer_base + 3 * MAX_N, case_gamma, case_beta, n, eps);
}

static int check_value(const char *group, const char *name, int case_idx,
		       int elem_idx, xlns32 expected, xlns32 got)
{
	float expected_fp = xlns322fp(expected);
	float got_fp = xlns322fp(got);
	float err = percent_error(expected_fp, got_fp);
	if (err <= XLNS32_MAX_ERROR_PERCENT) return 0;
	printf("%s %s mismatch case=%d elem=%d expected=%08x got=%08x expected_fp=%+.8e got_fp=%+.8e err=%f%%\n",
	       group, name, case_idx, elem_idx, expected, got, expected_fp, got_fp, err);
	return 1;
}

static void print_table(const char *title, const char **names, int num_results,
			const size_t *sizes, const xlns32 *expected,
			const xlns32 *got)
{
	printf("\n=== xlns32d %s CPU vs GPU ===\n", title);
	for (int f = 0; f < num_results; f++) {
		printf("\n[%s]\n", names[f]);
		printf("case | elem | CPU bits | GPU bits | CPU fp       | GPU fp       | err %%\n");
		printf("-----|------|----------|----------|--------------|--------------|-----------\n");
		for (int c = 0; c < NUM_CASES; c++) {
			for (size_t i = 0; i < sizes[c]; i++) {
				int idx = c * num_results * MAX_N + f * MAX_N + (int)i;
				printf("%4d | %4zu | %08x | %08x | %+12.5e | %+12.5e | %9.6f\n",
				       c, i, expected[idx], got[idx],
				       xlns322fp(expected[idx]), xlns322fp(got[idx]),
				       percent_error(xlns322fp(expected[idx]), xlns322fp(got[idx])));
			}
		}
	}
}

int main(void)
{
	const size_t sizes[NUM_CASES] = { 0, 1, 3, 5 };
	const float input_fp[NUM_CASES][MAX_N] = {
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 2.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ -1.0f, 0.0f, 1.0f, 0.0f, 0.0f },
		{ -2.0f, 0.5f, 1.5f, -0.75f, 3.0f }
	};
	const float mask_fp[NUM_CASES][MAX_N] = {
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, -INFINITY, 0.25f, 0.0f, 0.0f },
		{ 0.0f, -INFINITY, -0.25f, 0.5f, 0.0f }
	};
	const float gamma_fp[MAX_N] = { 1.0f, 0.5f, 1.5f, -1.0f, 2.0f };
	const float beta_fp[MAX_N] = { 0.0f, 0.25f, -0.5f, 1.0f, -1.0f };
	const xlns32 scale = fp2xlns32(0.75f);
	const float eps = 1.0e-5f;

	xlns32 h_a[NUM_CASES * MAX_N];
	xlns32 h_mask[NUM_CASES * MAX_N];
	xlns32 h_gamma[NUM_CASES * MAX_N];
	xlns32 h_beta[NUM_CASES * MAX_N];
	xlns32 expected_softmax[NUM_CASES * NUM_SOFTMAX_RESULTS * MAX_N] = { 0 };
	xlns32 got_softmax[NUM_CASES * NUM_SOFTMAX_RESULTS * MAX_N] = { 0 };
	xlns32 expected_layer[NUM_CASES * NUM_LAYER_RESULTS * MAX_N] = { 0 };
	xlns32 got_layer[NUM_CASES * NUM_LAYER_RESULTS * MAX_N] = { 0 };

	for (int c = 0; c < NUM_CASES; c++) {
		for (int i = 0; i < MAX_N; i++) {
			int idx = c * MAX_N + i;
			h_a[idx] = fp2xlns32(input_fp[c][i]);
			h_mask[idx] = isinf(mask_fp[c][i]) && mask_fp[c][i] < 0.0f ?
				      xlns32_neg_inf : fp2xlns32(mask_fp[c][i]);
			h_gamma[idx] = fp2xlns32(gamma_fp[i]);
			h_beta[idx] = fp2xlns32(beta_fp[i]);
		}
	}

	for (int c = 0; c < NUM_CASES; c++) {
		size_t n = sizes[c];
		const xlns32 *case_a = h_a + c * MAX_N;
		const xlns32 *case_mask = h_mask + c * MAX_N;
		const xlns32 *case_gamma = h_gamma + c * MAX_N;
		const xlns32 *case_beta = h_beta + c * MAX_N;
		xlns32 *soft_base = expected_softmax + c * NUM_SOFTMAX_RESULTS * MAX_N;
		xlns32 *layer_base = expected_layer + c * NUM_LAYER_RESULTS * MAX_N;

		xlns32_softmax_exp(case_a, soft_base + 0 * MAX_N, n);
		xlns32_softmax(case_a, soft_base + 1 * MAX_N, n, scale);
		xlns32_softmax_masked(case_a, case_mask, soft_base + 2 * MAX_N, n, scale);
		for (size_t i = 0; i < n; i++) soft_base[3 * MAX_N + i] = case_a[i];
		xlns32_softmax(soft_base + 3 * MAX_N, soft_base + 3 * MAX_N, n, scale);

		xlns32_layernorm(case_a, layer_base + 0 * MAX_N, 0, 0, n, eps);
		xlns32_layernorm(case_a, layer_base + 1 * MAX_N, case_gamma, 0, n, eps);
		xlns32_layernorm(case_a, layer_base + 2 * MAX_N, 0, case_beta, n, eps);
		xlns32_layernorm(case_a, layer_base + 3 * MAX_N, case_gamma, case_beta, n, eps);
	}

	xlns32 *d_a = 0;
	xlns32 *d_mask = 0;
	xlns32 *d_gamma = 0;
	xlns32 *d_beta = 0;
	size_t *d_sizes = 0;
	xlns32 *d_softmax = 0;
	xlns32 *d_layer = 0;

	CHECK_CUDA(cudaMalloc((void **)&d_a, sizeof(h_a)));
	CHECK_CUDA(cudaMalloc((void **)&d_mask, sizeof(h_mask)));
	CHECK_CUDA(cudaMalloc((void **)&d_gamma, sizeof(h_gamma)));
	CHECK_CUDA(cudaMalloc((void **)&d_beta, sizeof(h_beta)));
	CHECK_CUDA(cudaMalloc((void **)&d_sizes, sizeof(sizes)));
	CHECK_CUDA(cudaMalloc((void **)&d_softmax, sizeof(got_softmax)));
	CHECK_CUDA(cudaMalloc((void **)&d_layer, sizeof(got_layer)));
	CHECK_CUDA(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_mask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_beta, h_beta, sizeof(h_beta), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_sizes, sizes, sizeof(sizes), cudaMemcpyHostToDevice));

	xlns32d_softmax_layernorm_kernel<<<1, 32>>>(d_a, d_mask, d_gamma, d_beta,
						    d_sizes, scale, eps,
						    d_softmax, d_layer, NUM_CASES);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(got_softmax, d_softmax, sizeof(got_softmax), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(got_layer, d_layer, sizeof(got_layer), cudaMemcpyDeviceToHost));

	print_table("softmax", softmax_names, NUM_SOFTMAX_RESULTS, sizes,
		    expected_softmax, got_softmax);
	print_table("layernorm", layer_names, NUM_LAYER_RESULTS, sizes,
		    expected_layer, got_layer);

	int wrong = 0;
	for (int c = 0; c < NUM_CASES; c++) {
		for (int f = 0; f < NUM_SOFTMAX_RESULTS; f++) {
			for (size_t i = 0; i < sizes[c]; i++) {
				int idx = c * NUM_SOFTMAX_RESULTS * MAX_N + f * MAX_N + (int)i;
				wrong += check_value("softmax", softmax_names[f], c, (int)i,
						     expected_softmax[idx], got_softmax[idx]);
			}
		}
		for (int f = 0; f < NUM_LAYER_RESULTS; f++) {
			for (size_t i = 0; i < sizes[c]; i++) {
				int idx = c * NUM_LAYER_RESULTS * MAX_N + f * MAX_N + (int)i;
				wrong += check_value("layernorm", layer_names[f], c, (int)i,
						     expected_layer[idx], got_layer[idx]);
			}
		}
	}

	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_mask));
	CHECK_CUDA(cudaFree(d_gamma));
	CHECK_CUDA(cudaFree(d_beta));
	CHECK_CUDA(cudaFree(d_sizes));
	CHECK_CUDA(cudaFree(d_softmax));
	CHECK_CUDA(cudaFree(d_layer));

	printf("\nchkxlns32d_softmax_layernorm %s (%d wrong)\n",
	       wrong ? "FAIL" : "PASS", wrong);
	return wrong ? 1 : 0;
}
