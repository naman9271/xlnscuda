// Device port of xlnscpp's xlns32lpvip.cpp (xlns32_add_lpvip) and of the
// xlns16_*_lpvip32 summation: xlns16 terms widened into a 32-bit LPVIP
// accumulator, narrowed once at the end.
// Include after xlns16.cpp, xlns32lpvip.cpp and xlns16d.cu.

#ifndef xlns32_F
  #error "xlns32d_lpvip.cu: include xlns32lpvip.cpp first"
#endif

__device__ inline xlns32 xlns32d_lpvip_mul(xlns32 x, xlns32 y)
{
   xlns32 t = (xlns32_logmask&x) + (xlns32_logmask&y) - xlns32_logsignmask;
   if (xlns32_signmask&t)
      return (xlns32_logsignmask&t) ? (xlns32_signmask&(x^y)) : ((xlns32_signmask&(x^y)) | xlns32_logmask);
   return (xlns32_signmask&(x^y)) | t;
}

__device__ inline xlns32 xlns32d_lpvip_mitch(xlns32 z)
{
   const int shift = -(((xlns32_signed)z)>>xlns32_F);
   return (shift >= 32) ? 0 : (((1<<xlns32_F)+(z&((1<<xlns32_F)-1)))>>shift);
}

__device__ inline xlns32 xlns32d_lpvip_db_ideal(xlns32_signed z)
{
	return ((xlns32_signed) ((log( pow(2.0, ((double) z) / xlns32_scale) - 1 )/log(2.0))*xlns32_scale+.5));
}

__device__ inline xlns32 xlns32d_add_lpvip(xlns32 x, xlns32 y)
{
    xlns32 minxyl, maxxy, xl, yl, usedb, adjust, adjustez;
    xlns32_signed z;
    xl = x & xlns32_logmask;
    yl = y & xlns32_logmask;
    minxyl = (yl>xl) ? xl : yl;
    maxxy  = (xl>yl) ? x  : y;
    z = minxyl - (maxxy&xlns32_logmask);
    usedb = xlns32_signmask&(x^y);
    if ((z==0) && usedb)
       return xlns32_zero;
    xlns32_signed precond = (usedb==0) ? (
                                         #ifdef xlns32_aicasb
                                           (z > -(7<<(xlns32_F-1))) ? ((-z)>>3) : (7<<(xlns32_F-4))
                                         #else
                                           ((-z)>>3)
                                         #endif
                                         ) :
             (z < -(2<<xlns32_F)) ? 5<<(xlns32_F-3) : (z >> 2) + (9 << (xlns32_F-3));
    xlns32_signed postcond = (z <= -(3<<xlns32_F)) ? 0 :
                             z >= -(3<<(xlns32_F-2)) ? -(1<<(xlns32_F-6)) : +(1<<(xlns32_F-6));
    xlns32_signed mitch = (-z >= 1<<xlns32_F)||(usedb==0) ? xlns32d_lpvip_mitch(z+precond) :
                                       -xlns32d_lpvip_db_ideal(-z)-z;
    adjust = usedb ? -mitch : (z==0) ? 1<<(xlns32_F) : mitch + postcond;
    adjustez = (z < -xlns32_esszer) ? 0 : adjust;
    return xlns32d_lpvip_mul(maxxy, xlns32_logsignmask + adjustez);
}

// xlns16 terms, 32-bit accumulator (shift 16 for the default xlns32 F=23)
__host__ __device__ inline xlns32 xlns16_lpvip32_widen(xlns16 v)   { return ((xlns32)v)<<(xlns32_canonshift-15); }
__host__ __device__ inline xlns16 xlns16_lpvip32_narrow(xlns32 a)  { return (xlns16)(a>>(xlns32_canonshift-15)); }
__device__ inline xlns32 xlns16d_lpvip32_add(xlns32 acc, xlns16 t) { return xlns32d_add_lpvip(acc, xlns16_lpvip32_widen(t)); }

__device__ inline xlns16 xlns16d_vec_dot_lpvip32(const xlns16 *a, const xlns16 *b, size_t n)
{
	if (n == 0) return xlns16_zero;
	xlns32 sum = xlns16_lpvip32_widen(xlns16d_mul(a[0], b[0]));
	for (size_t i = 1; i < n; i++)
		sum = xlns16d_lpvip32_add(sum, xlns16d_mul(a[i], b[i]));
	return xlns16_lpvip32_narrow(sum);
}
