// Device port of xlnscpp's xlns16monte.cpp (or, with xlns16d_monte_bylpvip,
// xlns16monte_bylpvip.cpp) with explicit random state instead of the host's
// global xlns16_randombits. Every Monte add advances the 16-bit LFSR
// (taps 0xB400, period 65535, seed 7; 0 is invalid) by one step, as on the
// host. xlns16_lfsr_jump advances a state by n steps, so parallel reductions
// can each start where a sequential host loop would be.
// Include after xlns16.cpp and xlns16d.cu (+ xlns32d_lpvip.cu for bylpvip).

#define xlns16_lfsr_period 65535u

__host__ __device__ inline xlns16 xlns16_lfsr_next(xlns16 s)
{
	xlns16 bit = s & 1;
	s >>= 1;
	if (bit) s ^= 0xB400u;
	return s;
}

// [k][c]: the state 2^k steps after the basis state 1<<c (the step is linear)
#define xlns16_lfsr_jump_init { \
  { 0xb400, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000 }, \
  { 0x5a00, 0xb400, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000 }, \
  { 0x1680, 0x2d00, 0x5a00, 0xb400, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800 }, \
  { 0x0168, 0x02d0, 0x05a0, 0x0b40, 0x1680, 0x2d00, 0x5a00, 0xb400, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080 }, \
  { 0x7c41, 0xf882, 0x9905, 0x5a0b, 0xb416, 0x002d, 0x005a, 0x00b4, 0x0168, 0x02d0, 0x05a0, 0x0b40, 0x1680, 0x2d00, 0x5a00, 0xb400 }, \
  { 0x137b, 0x26f6, 0x4dec, 0x9bd8, 0x5fb1, 0xbf62, 0x16c5, 0x2d8a, 0x5b14, 0xb628, 0x0451, 0x08a2, 0x1144, 0x2288, 0x4510, 0x8a20 }, \
  { 0xa43b, 0x2077, 0x40ee, 0x81dc, 0x6bb9, 0xd772, 0xc6e5, 0xe5cb, 0xa397, 0x2f2f, 0x5e5e, 0xbcbc, 0x1179, 0x22f2, 0x45e4, 0x8bc8 }, \
  { 0x4e3f, 0x9c7e, 0x50fd, 0xa1fa, 0x2bf5, 0x57ea, 0xafd4, 0x37a9, 0x6f52, 0xdea4, 0xd549, 0xc293, 0xed27, 0xb24f, 0x0c9f, 0x193e }, \
  { 0x9ab8, 0x5d71, 0xbae2, 0x1dc5, 0x3b8a, 0x7714, 0xee28, 0xb451, 0x00a3, 0x0146, 0x028c, 0x0518, 0x0a30, 0x1460, 0x28c0, 0x5180 }, \
  { 0xbae4, 0x1dc9, 0x3b92, 0x7724, 0xee48, 0xb491, 0x0123, 0x0246, 0x048c, 0x0918, 0x1230, 0x2460, 0x48c0, 0x9180, 0x4b01, 0x9602 }, \
  { 0xee5c, 0xb4b9, 0x0173, 0x02e6, 0x05cc, 0x0b98, 0x1730, 0x2e60, 0x5cc0, 0xb980, 0x1b01, 0x3602, 0x6c04, 0xd808, 0xd811, 0xd823 }, \
  { 0x5dd0, 0xbba0, 0x1f41, 0x3e82, 0x7d04, 0xfa08, 0x9c11, 0x5023, 0xa046, 0x288d, 0x511a, 0xa234, 0x2c69, 0x58d2, 0xb1a4, 0x0b49 }, \
  { 0x7f93, 0xff26, 0x964d, 0x449b, 0x8936, 0x7a6d, 0xf4da, 0x81b5, 0x6b6b, 0xd6d6, 0xc5ad, 0xe35b, 0xaeb7, 0x356f, 0x6ade, 0xd5bc }, \
  { 0xe27b, 0xacf7, 0x31ef, 0x63de, 0xc7bc, 0xe779, 0xa6f3, 0x25e7, 0x4bce, 0x979c, 0x4739, 0x8e72, 0x74e5, 0xe9ca, 0xbb95, 0x1f2b }, \
  { 0x718c, 0xe318, 0xae31, 0x3463, 0x68c6, 0xd18c, 0xcb19, 0xfe33, 0x9467, 0x40cf, 0x819e, 0x6b3d, 0xd67a, 0xc4f5, 0xe1eb, 0xabd7 }, \
  { 0x0362, 0x06c4, 0x0d88, 0x1b10, 0x3620, 0x6c40, 0xd880, 0xd901, 0xda03, 0xdc07, 0xd00f, 0xc81f, 0xf83f, 0x987f, 0x58ff, 0xb1fe }  \
}

static const xlns16 xlns16_lfsr_jump_tbl[16][16] = xlns16_lfsr_jump_init;
__constant__ xlns16 xlns16d_lfsr_jump_tbl[16][16] = xlns16_lfsr_jump_init;

// (index * per) mod period without overflow
__host__ __device__ inline unsigned long long xlns16_lfsr_offset(unsigned long long index, unsigned long long per)
{
	return ((index % xlns16_lfsr_period) * (per % xlns16_lfsr_period)) % xlns16_lfsr_period;
}

__host__ __device__ inline xlns16 xlns16_lfsr_jump(xlns16 s, unsigned long long n)
{
	unsigned int m = (unsigned int) (n % xlns16_lfsr_period);
	for (int k = 0; m != 0; k++, m >>= 1) {
		if (!(m & 1)) continue;
		xlns16 r = 0;
		for (int c = 0; c < 16; c++) {
			if ((s >> c) & 1) {
			#ifdef __CUDA_ARCH__
				r ^= xlns16d_lfsr_jump_tbl[k][c];
			#else
				r ^= xlns16_lfsr_jump_tbl[k][c];
			#endif
			}
		}
		s = r;
	}
	return s;
}

#ifdef xlns16d_monte_bylpvip

__device__ inline xlns16 xlns16d_add_monte(xlns16 x, xlns16 y, xlns16 *state)
{
	const xlns16 r = *state = xlns16_lfsr_next(*state);
	return (xlns16) (xlns32d_add_lpvip((((xlns32)x)<<16)|r, (((xlns32)y)<<16)|r)>>16);
}

#else

// xlns16_mitch with a defined shift (shift count from the signed value)
__device__ inline xlns16 xlns16d_monte_mitch(xlns16 z)
{
	const int zs = (xlns16_signed) z;
	const int shift = -(zs >> xlns16_F);
	if (shift < 0 || shift >= 16) return 0;
	return (xlns16) (((1<<xlns16_F)+(zs&((1<<xlns16_F)-1)))>>shift);
}

__device__ inline xlns16 xlns16d_add_monte(xlns16 x, xlns16 y, xlns16 *state)
{
	const xlns16 normres = xlns16d_add(x, y);
	const xlns16 r = *state = xlns16_lfsr_next(*state);   // one draw per call, always
	if ((normres != x) && (normres != y))
		return normres;
	// xlns16_monte_modifier
	xlns16 xl = x & xlns16_logmask, yl = y & xlns16_logmask;
	xlns16 minxyl = (yl>xl) ? xl : yl;
	xlns16 maxxy  = (xl>yl) ? x  : y;
	xlns16_signed z = minxyl - (maxxy&xlns16_logmask);
	xlns16 result = (r>>(16-xlns16_F)) + xlns16d_monte_mitch(z + (xlns16_F<<xlns16_F) + (7<<(xlns16_F-4)));
	result = result >> xlns16_F;
	result &= (z < -((xlns16_F)<<xlns16_F));
	return normres + ((xlns16_signmask&(x^y)) ? (xlns16)-result : result);
}

#endif

__device__ inline xlns16 xlns16d_vec_dot_monte(const xlns16 *a, const xlns16 *b, size_t n, xlns16 *state)
{
	if (n == 0) return xlns16_zero;
	xlns16 sum = xlns16d_mul(a[0], b[0]);
	for (size_t i = 1; i < n; i++)
		sum = xlns16d_add_monte(sum, xlns16d_mul(a[i], b[i]), state);
	return sum;
}
