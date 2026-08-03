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

#define NUM_CASES 8
#define NUM_BOOL_RESULTS 8
#define NUM_LNS_RESULTS 5

static const char *bool_names[NUM_BOOL_RESULTS] = {
	"is_zero", "is_negative", "is_positive", "gt", "lt", "eq", "ge", "le"
};

static const char *lns_names[NUM_LNS_RESULTS] = {
	"max", "min", "copysign", "fma", "relu"
};

__global__ void xlns32d_utils_kernel(const xlns32 *a, const xlns32 *b,
				     const xlns32 *c,
				     int *bool_results, xlns32 *lns_results,
				     int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	xlns32 av = a[i];
	xlns32 bv = b[i];
	xlns32 cv = c[i];
	int base_bool = i * NUM_BOOL_RESULTS;
	int base_lns = i * NUM_LNS_RESULTS;

	bool_results[base_bool + 0] = xlns32d_is_zero(av);
	bool_results[base_bool + 1] = xlns32d_is_negative(av);
	bool_results[base_bool + 2] = xlns32d_is_positive(av);
	bool_results[base_bool + 3] = xlns32d_gt(av, bv);
	bool_results[base_bool + 4] = xlns32d_lt(av, bv);
	bool_results[base_bool + 5] = xlns32d_eq(av, bv);
	bool_results[base_bool + 6] = xlns32d_ge(av, bv);
	bool_results[base_bool + 7] = xlns32d_le(av, bv);

	lns_results[base_lns + 0] = xlns32d_max(av, bv);
	lns_results[base_lns + 1] = xlns32d_min(av, bv);
	lns_results[base_lns + 2] = xlns32d_copysign(av, bv);
	lns_results[base_lns + 3] = xlns32d_fma(av, bv, cv);
	lns_results[base_lns + 4] = xlns32d_relu(av);
}

static int check_bool_result(int case_idx, int result_idx, int expected, int got)
{
	if (expected == got) return 0;
	printf("bool mismatch case=%d result=%d expected=%d got=%d\n",
	       case_idx, result_idx, expected, got);
	return 1;
}

static int check_lns_result(int case_idx, int result_idx, xlns32 expected, xlns32 got)
{
	if (expected == got) return 0;
	printf("lns mismatch case=%d result=%d expected=%08x got=%08x\n",
	       case_idx, result_idx, expected, got);
	return 1;
}

static void print_results_table(const xlns32 *a, const xlns32 *b,
				const int *expected_bool, const int *got_bool,
				const xlns32 *expected_lns, const xlns32 *got_lns)
{
	printf("\n=== xlns32d utility CPU vs GPU table (grouped by function) ===\n");
	for (int f = 0; f < NUM_BOOL_RESULTS; f++) {
		printf("\n[%s]\n", bool_names[f]);
		printf("case | a bits   | b bits   | a fp | b fp | CPU | GPU\n");
		printf("-----|----------|----------|------|------|-----|-----\n");
		for (int i = 0; i < NUM_CASES; i++) {
			int idx = i * NUM_BOOL_RESULTS + f;
			printf("%4d | %08x | %08x | %+7.4f | %+7.4f | %3d | %3d\n",
			       i, a[i], b[i], xlns322fp(a[i]), xlns322fp(b[i]),
			       expected_bool[idx], got_bool[idx]);
		}
	}
	for (int f = 0; f < NUM_LNS_RESULTS; f++) {
		printf("\n[%s]\n", lns_names[f]);
		printf("case | a bits   | b bits   | a fp | b fp | CPU | GPU\n");
		printf("-----|----------|----------|------|------|-----|-----\n");
		for (int i = 0; i < NUM_CASES; i++) {
			int idx = i * NUM_LNS_RESULTS + f;
			printf("%4d | %08x | %08x | %+7.4f | %+7.4f | %08x | %08x\n",
			       i, a[i], b[i], xlns322fp(a[i]), xlns322fp(b[i]),
			       expected_lns[idx], got_lns[idx]);
		}
	}
	printf("\n");
}

int main(void)
{
	xlns32 h_a[NUM_CASES] = {
		/* zero */
		xlns32_zero,
		/* positive / fma(2, 3, 4) */
		xlns32_two,
		/* negative / fma(-2, 3, 4) */
		xlns32_neg_two,
		/* gt/lt */
		xlns32_two,
		/* eq/ge/le */
		xlns32_half,
		/* max/min */
		fp2xlns32(0.25f),
		/* copysign */
		fp2xlns32(-0.25f),
		/* mixed edge */
		fp2xlns32(3.5f)
	};
	xlns32 h_b[NUM_CASES] = {
		/* zero */
		xlns32_zero,
		/* positive / fma(2, 3, 4) */
		fp2xlns32(3.0f),
		/* negative / fma(-2, 3, 4) */
		fp2xlns32(3.0f),
		/* gt/lt */
		xlns32_one,
		/* eq/ge/le */
		xlns32_half,
		/* max/min */
		fp2xlns32(0.25f),
		/* copysign */
		fp2xlns32(0.25f),
		/* mixed edge */
		fp2xlns32(-4.0f)
	};
	xlns32 h_c[NUM_CASES] = {
		/* zero */
		xlns32_zero,
		/* positive */
		fp2xlns32(4.0f),
		/* negative */
		fp2xlns32(4.0f),
		/* gt/lt */
		xlns32_zero,
		/* eq/ge/le */
		xlns32_zero,
		/* max/min */
		xlns32_zero,
		/* copysign */
		xlns32_zero,
		/* mixed edge */
		xlns32_zero
	};

	int expected_bool[NUM_CASES * NUM_BOOL_RESULTS];
	xlns32 expected_lns[NUM_CASES * NUM_LNS_RESULTS];
	int got_bool[NUM_CASES * NUM_BOOL_RESULTS];
	xlns32 got_lns[NUM_CASES * NUM_LNS_RESULTS];

	for (int i = 0; i < NUM_CASES; i++) {
		int base_bool = i * NUM_BOOL_RESULTS;
		int base_lns = i * NUM_LNS_RESULTS;
		expected_bool[base_bool + 0] = xlns32_is_zero(h_a[i]);
		expected_bool[base_bool + 1] = xlns32_is_negative(h_a[i]);
		expected_bool[base_bool + 2] = xlns32_is_positive(h_a[i]);
		expected_bool[base_bool + 3] = xlns32_gt(h_a[i], h_b[i]);
		expected_bool[base_bool + 4] = xlns32_lt(h_a[i], h_b[i]);
		expected_bool[base_bool + 5] = xlns32_eq(h_a[i], h_b[i]);
		expected_bool[base_bool + 6] = xlns32_ge(h_a[i], h_b[i]);
		expected_bool[base_bool + 7] = xlns32_le(h_a[i], h_b[i]);

		expected_lns[base_lns + 0] = xlns32_max(h_a[i], h_b[i]);
		expected_lns[base_lns + 1] = xlns32_min(h_a[i], h_b[i]);
		expected_lns[base_lns + 2] = xlns32_copysign(h_a[i], h_b[i]);
		expected_lns[base_lns + 3] = xlns32_add(xlns32_mul(h_a[i], h_b[i]), h_c[i]);
		expected_lns[base_lns + 4] = xlns32_is_negative(h_a[i]) ? xlns32_zero : h_a[i];
	}

	xlns32 *d_a = 0;
	xlns32 *d_b = 0;
	xlns32 *d_c = 0;
	int *d_bool = 0;
	xlns32 *d_lns = 0;

	CHECK_CUDA(cudaMalloc((void **)&d_a, sizeof(h_a)));
	CHECK_CUDA(cudaMalloc((void **)&d_b, sizeof(h_b)));
	CHECK_CUDA(cudaMalloc((void **)&d_c, sizeof(h_c)));
	CHECK_CUDA(cudaMalloc((void **)&d_bool, sizeof(got_bool)));
	CHECK_CUDA(cudaMalloc((void **)&d_lns, sizeof(got_lns)));
	CHECK_CUDA(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_c, h_c, sizeof(h_c), cudaMemcpyHostToDevice));

	xlns32d_utils_kernel<<<1, 32>>>(d_a, d_b, d_c, d_bool, d_lns, NUM_CASES);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(got_bool, d_bool, sizeof(got_bool), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(got_lns, d_lns, sizeof(got_lns), cudaMemcpyDeviceToHost));

	print_results_table(h_a, h_b, expected_bool, got_bool, expected_lns, got_lns);

	int wrong = 0;
	for (int i = 0; i < NUM_CASES; i++) {
		for (int j = 0; j < NUM_BOOL_RESULTS; j++) {
			int idx = i * NUM_BOOL_RESULTS + j;
			wrong += check_bool_result(i, j, expected_bool[idx], got_bool[idx]);
		}
		for (int j = 0; j < NUM_LNS_RESULTS; j++) {
			int idx = i * NUM_LNS_RESULTS + j;
			wrong += check_lns_result(i, j, expected_lns[idx], got_lns[idx]);
		}
	}

	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_c));
	CHECK_CUDA(cudaFree(d_bool));
	CHECK_CUDA(cudaFree(d_lns));

	printf("chkxlns32d_utils %s (%d wrong)\n", wrong ? "FAIL" : "PASS", wrong);
	return wrong ? 1 : 0;
}
