// Device Monte (xlns16d_monte.cu) and LPVIP (xlns32d_lpvip.cu) vs xlnscpp.
// Bit-exact results and LFSR states are required.
//
//   nvcc -std=c++14 -I xlnscpp tests/chkxlns16d_sum_modes.cu -o chk
//   nvcc -std=c++14 -I xlnscpp -Dxlns16d_monte_bylpvip tests/chkxlns16d_sum_modes.cu -o chk_bylpvip
//
// Exit codes: 0 pass, 1 fail, 77 no CUDA device.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define xlns16_table
#define xlns16_alt
#define xlns16d_table
#include "xlns16.cpp"
#include "xlns32lpvip.cpp"
#ifdef xlns16d_monte_bylpvip
#include "xlns16monte_bylpvip.cpp"
#else
#include "xlns16monte.cpp"
#endif
#include "../xlns16d.cu"
#include "../xlns32d_lpvip.cu"
#include "../xlns16d_monte.cu"

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
	fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static int failures = 0;
static void report(const char *name, long long bad, long long total)
{
	printf("  %-50s %s (%lld / %lld mismatches)\n", name, bad ? "FAIL" : "ok", bad, total);
	failures += bad != 0;
}

static unsigned rng = 0x2545F491u;
static unsigned rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

template <class T> static T *to_dev(const std::vector<T> &v)
{
	T *d; CHECK_CUDA(cudaMalloc(&d, v.size() * sizeof(T) + 1));
	CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
	return d;
}
template <class T> static std::vector<T> to_host(const T *d, size_t n)
{
	std::vector<T> v(n); CHECK_CUDA(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
	return v;
}

__global__ void k_monte_add(const xlns16 *x, const xlns16 *y, xlns16 *s, xlns16 *r, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = xlns16d_add_monte(x[i], y[i], &s[i]);
}

__global__ void k_lpvip_add(const xlns32 *x, const xlns32 *y, xlns32 *r, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) r[i] = xlns32d_add_lpvip(x[i], y[i]);
}

__global__ void k_db_ideal(xlns32 *r, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1 && i < n) r[i] = xlns32d_lpvip_db_ideal(i);
}

// one dot product per row; monte rows start at jump(base, row * (n-1))
__global__ void k_dots(const xlns16 *a, const xlns16 *b, int rows, int n, xlns16 base, xlns16 *monte, xlns16 *lpvip)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	xlns16 st = xlns16_lfsr_jump(base, xlns16_lfsr_offset(r, n - 1));
	monte[r] = xlns16d_vec_dot_monte(a + (size_t) r * n, b + (size_t) r * n, n, &st);
	lpvip[r] = xlns16d_vec_dot_lpvip32(a + (size_t) r * n, b + (size_t) r * n, n);
}

int main(void)
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { printf("SKIP: no CUDA device\n"); return 77; }
	CHECK_CUDA(xlns16d_table_init());

	// LFSR: one host monte add == one step; jump == stepping
	long long bad = 0;
	for (unsigned s = 1; s <= 0xFFFF; s++) {
		xlns16_randombits = (xlns16) s;
		xlns16_add_monte(xlns16_one, xlns16_half);
		bad += xlns16_randombits != xlns16_lfsr_next((xlns16) s);
	}
	for (int t = 0; t < 2000; t++) {
		xlns16 s0 = (xlns16) (rnd() % 65535 + 1), v = s0;
		unsigned long long n = rnd() % 200000;
		for (unsigned long long i = 0; i < n % 65535; i++) v = xlns16_lfsr_next(v);
		bad += v != xlns16_lfsr_jump(s0, n);
	}
	report("LFSR step and jump-ahead", bad, 65535 + 2000);

	// Monte add: all x against y stride 61, random states
	std::vector<xlns16> x, y, s;
	for (unsigned yi = 0; yi < 65536; yi += 61)
		for (unsigned xi = 0; xi < 65536; xi++) { x.push_back(xi); y.push_back(yi); s.push_back(rnd() % 65535 + 1); }
	int n = (int) x.size();
	xlns16 *dx = to_dev(x), *dy = to_dev(y), *ds = to_dev(s), *dr;
	CHECK_CUDA(cudaMalloc(&dr, n * sizeof(xlns16)));
	k_monte_add<<<(n + 255) / 256, 256>>>(dx, dy, ds, dr, n);
	std::vector<xlns16> r = to_host(dr, n), s2 = to_host(ds, n);
	bad = 0;
	for (int i = 0; i < n; i++) {
		xlns16_randombits = s[i];
		bad += xlns16_add_monte(x[i], y[i]) != r[i] || xlns16_randombits != s2[i];
	}
	report("Monte add (result and state)", bad, n);

	// LPVIP: singularity branch exhaustively, then widened and random pairs
	const int nd = 1 << xlns32_F;
	xlns32 *dd; CHECK_CUDA(cudaMalloc(&dd, nd * sizeof(xlns32)));
	k_db_ideal<<<(nd + 255) / 256, 256>>>(dd, nd);
	std::vector<xlns32> dv = to_host(dd, nd);
	bad = 0;
	for (int i = 1; i < nd; i++) bad += dv[i] != xlns32_db_ideal(i);
	report("LPVIP singularity log/pow, all 2^23 inputs", bad, nd - 1);

	std::vector<xlns32> px, py;
	for (size_t i = 0; i < x.size(); i++) { px.push_back(xlns16_lpvip32_widen(x[i])); py.push_back(xlns16_lpvip32_widen(y[i])); }
	for (int i = 0; i < (1 << 21); i++) {
		xlns32 a = rnd(), d = rnd() % ((i & 1) ? (1u << xlns32_F) : 64u);
		px.push_back(a); py.push_back(((a & xlns32_logmask) - (d < (a & xlns32_logmask) ? d : 0)) ^ ((i & 2) ? xlns32_signmask : 0));
	}
	n = (int) px.size();
	xlns32 *dpx = to_dev(px), *dpy = to_dev(py), *dpr;
	CHECK_CUDA(cudaMalloc(&dpr, n * sizeof(xlns32)));
	k_lpvip_add<<<(n + 255) / 256, 256>>>(dpx, dpy, dpr, n);
	std::vector<xlns32> pr = to_host(dpr, n);
	bad = 0;
	for (int i = 0; i < n; i++) bad += pr[i] != xlns32_add_lpvip(px[i], py[i]);
	report("LPVIP add (widened, near/exact cancellation)", bad, n);

	// dot products: host sequential (one global LFSR) vs device parallel rows,
	// including rows longer than the LFSR period
	const int shapes[][2] = { { 512, 1 }, { 256, 1536 }, { 3, 70001 } };
	for (auto &sh : shapes) {
		int rows = sh[0], k = sh[1];
		std::vector<xlns16> a((size_t) rows * k), b((size_t) rows * k);
		for (size_t i = 0; i < a.size(); i++) {
			a[i] = fp2xlns16((int) (rnd() % 20001 - 10000) / 5000.0f);
			b[i] = fp2xlns16((int) (rnd() % 20001 - 10000) / 5000.0f);
		}
		xlns16 base = rnd() % 65535 + 1, *dm, *dl;
		xlns16 *da = to_dev(a), *db = to_dev(b);
		CHECK_CUDA(cudaMalloc(&dm, rows * sizeof(xlns16)));
		CHECK_CUDA(cudaMalloc(&dl, rows * sizeof(xlns16)));
		k_dots<<<(rows + 63) / 64, 64>>>(da, db, rows, k, base, dm, dl);
		std::vector<xlns16> gm = to_host(dm, rows), gl = to_host(dl, rows);
		bad = 0;
		xlns16_randombits = base;
		for (int i = 0; i < rows; i++) {
			bad += xlns16_vec_dot_monte(&a[(size_t) i * k], &b[(size_t) i * k], k) != gm[i];
			bad += xlns16_vec_dot_lpvip32(&a[(size_t) i * k], &b[(size_t) i * k], k) != gl[i];
		}
		bad += xlns16_randombits != xlns16_lfsr_jump(base, (unsigned long long) rows * (k - 1));
		char name[64];
		snprintf(name, sizeof(name), "Monte + LPVIP dots, %d rows x K=%d", rows, k);
		report(name, bad, 2LL * rows + 1);
	}

	printf("%s\n", failures ? "FAIL" : "PASS");
	return failures ? 1 : 0;
}
