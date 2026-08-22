// 16-bit XLNS for CUDA
// copyright 1999-2025 Mark G. Arnold
// these routines 
//    ran on 16-bit Turbo C/C++ (the file may have CR/LFs from that system) 
//    were used in my PhD research and for several later papers on 32-bit LNS
// they were ported to Linux gcc and g++ around 2015 on 32-bit x86
// they were ported again for 64-bit arch in 2025, 
// they were copied and modified as __device__ CUDA routines
// they are based on similar math foundation (Gaussian logs, sb and db) as Python xlns,
//    but use different internal storage format:
//    +------+-------------------------+
//    | sign | int(log2) . frac(log2)  |
//    +------+-------------------------+
//    the int(log2) is not twos complement; it is offset (logsignmask XORed)
//    for the 32-bit format in this file, this is roughly similar to float32
//    1 sign bit, 8 int(log2) bits, 23 frac(log2) bits
//    there is an exact representation of 0.0, but no subnormals or NaNs


/* PORTABLE CODE STARTS HERE*/

inline __device__ xlns32 xlns32d_overflow(xlns32 xlns32_x, xlns32 xlns32_y, xlns32 xlns32_temp)
{       
	if (xlns32_logsignmask&xlns32_temp)
		return (xlns32_signmask&(xlns32_x^xlns32_y));
	else
		return (xlns32_signmask&(xlns32_x^xlns32_y))| xlns32_logmask;
}

inline __device__ xlns32 xlns32d_mul(xlns32 x, xlns32 y)
{
   xlns32 xlns32_temp;
   xlns32_temp=(xlns32_logmask&(x))+(xlns32_logmask&(y))-xlns32_logsignmask; 
   return  (xlns32_signmask&(xlns32_temp)) ? xlns32d_overflow(x,y,xlns32_temp) 
                                       :(xlns32_signmask&(x^y))|xlns32_temp;
}

inline __device__ xlns32 xlns32d_div(xlns32 x, xlns32 y)
{
   xlns32 xlns32_temp;
   xlns32_temp=(xlns32_logmask&(x))-(xlns32_logmask&(y))+xlns32_logsignmask; 
   return  (xlns32_signmask&(xlns32_temp)) ? xlns32d_overflow(x,y,xlns32_temp) 
                                       :(xlns32_signmask&(x^y))|xlns32_temp;
}

#ifdef xlns32_ideal
  #define xlns32d_sb xlns32d_sb_ideal
  #define xlns32d_db xlns32d_db_ideal
  #include <math.h>
  inline __device__ xlns32 xlns32d_sb_ideal(xlns32_signed z)
  {
	return ((xlns32) ((log(1+ pow(2.0, ((double) z) / xlns32_scale) )/log(2.0))*xlns32_scale+.5));
  }
  inline __device__ xlns32 xlns32d_db_ideal(xlns32_signed z)
  {
	return ((xlns32_signed) ((log( pow(2.0, ((double) z) / xlns32_scale) - 1 )/log(2.0))*xlns32_scale+.5));
  }
//#else
// no implementation on CUDA for non-ideal yet
#endif



#ifdef xlns32_alt

__device__ inline xlns32 xlns32d_add(xlns32 x, xlns32 y)
{
    xlns32 minxyl, maxxy, xl, yl, usedb, adjust, adjustez;
    xlns32_signed z;

    xl = x & xlns32_logmask;

    yl = y & xlns32_logmask;
    minxyl = (yl>xl) ? xl : yl;
    maxxy  = (xl>yl) ? x  : y;
    z = minxyl - (maxxy&xlns32_logmask);
    usedb = xlns32_signmask&(x^y);

    #ifdef xlns32_ideal

     float pm1 = usedb ? -1.0 : 1.0;

     adjust = z+((xlns32_signed)(log(pm1+pow(2.0,-((double)z)/xlns32_scale))/log(2.0)*xlns32_scale+.5));

    //#else
    // no implementation on CUDA for non-ideal yet
    #endif 
    adjustez = (z < -xlns32_esszer) ? 0 : adjust;
    return ((z==0) && usedb) ?
                     xlns32_zero :
                      xlns32d_mul(maxxy, xlns32_logsignmask + adjustez);
}

#else



//++++ X-X ERROR fixed

__device__ xlns32 xlns32d_add(xlns32 x, xlns32 y)
{
	xlns32 t;
	xlns32_signed z;

	z = (x&xlns32_logmask) - (y&xlns32_logmask);
	if (z<0)
	{
		z = -z;
		t = x;
		x = y;
		y = t;
	}
	if (xlns32_signmask&(x^y))
	{
		if (z == 0)
			return xlns32_zero;
		if (z < xlns32_esszer)
			return xlns32_neg(y + xlns32d_db(z));
		else
			return xlns32_neg(y+z);
	}
	else
	{
		return y + xlns32d_sb(z);
	}
}

#endif

#define xlns32d_sub(x,y) xlns32d_add(x,xlns32_neg(y))

__device__ inline int xlns32d_is_zero(xlns32 x)
{
	return (xlns32_abs(x) == xlns32_zero);
}

__device__ inline int xlns32d_is_negative(xlns32 x)
{
	return (xlns32_sign(x) != 0) && !xlns32d_is_zero(x);
}

__device__ inline int xlns32d_is_positive(xlns32 x)
{
	return (xlns32_sign(x) == 0) && !xlns32d_is_zero(x);
}

__device__ inline int xlns32d_gt(xlns32 a, xlns32 b)
{
	return xlns32_canon(a) > xlns32_canon(b);
}

__device__ inline int xlns32d_lt(xlns32 a, xlns32 b)
{
	return xlns32_canon(a) < xlns32_canon(b);
}

__device__ inline int xlns32d_eq(xlns32 a, xlns32 b)
{
	return a == b;
}

__device__ inline int xlns32d_ge(xlns32 a, xlns32 b)
{
	return xlns32_canon(a) >= xlns32_canon(b);
}

__device__ inline int xlns32d_le(xlns32 a, xlns32 b)
{
	return xlns32_canon(a) <= xlns32_canon(b);
}

__device__ inline xlns32 xlns32d_max(xlns32 a, xlns32 b)
{
	return xlns32d_gt(a, b) ? a : b;
}

__device__ inline xlns32 xlns32d_min(xlns32 a, xlns32 b)
{
	return xlns32d_lt(a, b) ? a : b;
}

__device__ inline xlns32 xlns32d_copysign(xlns32 x, xlns32 y)
{
	return xlns32_abs(x) | xlns32_sign(y);
}

__device__ inline xlns32 xlns32d_fma(xlns32 a, xlns32 b, xlns32 c)
{
	return xlns32d_add(xlns32d_mul(a, b), c);
}

__device__ inline xlns32 xlns32d_relu(xlns32 x)
{
	return xlns32d_is_negative(x) ? xlns32_zero : x;
}

/*END OF PORTABLE CODE*/

/*START OF PORTABLE CODE THAT DEPENDS ON <math.h>*/

#include <math.h>
#include <stddef.h>

/*END OF PORTABLE CODE THAT DEPENDS ON <math.h>*/


__device__ xlns32 fp2xlns32d(double x)
{
	if ((x>-2.938747e-39)&&(x<2.938747e-39))
		return(xlns32_zero);
        else if (x> 3.40282286e+38)
		return(xlns32_pos_inf);
        else if (x< -3.40282286e+38)
		return(xlns32_neg_inf);
	else if (x > 0.0)
		return xlns32_abs((xlns32_signed) ((log(x)/log(2.0))*xlns32_scale))
		       ^xlns32_logsignmask;
	else
		return (((xlns32_signed) ((log(fabs(x))/log(2.0))*xlns32_scale))
			  |xlns32_signmask)^xlns32_logsignmask;
}

__device__ float xlns32d2fp(xlns32 x)
{
	if (xlns32_abs(x) == xlns32_zero)
		return (0.0);
	else if (xlns32_sign(x))
		return (float) (-pow(2.0,((double) (((xlns32_signed) (xlns32_abs(x)-xlns32_logsignmask))))
					/((float) xlns32_scale)));
	else {
		return (float) (+pow(2.0,((double) (((xlns32_signed) (xlns32_abs(x)-xlns32_logsignmask))))
					/((float) xlns32_scale)));
	}
}

__device__ inline xlns32 xlns32d_from_float(float x)
{
	return fp2xlns32d(x);
}

__device__ inline float xlns32d_to_float(xlns32 x)
{
	return xlns32d2fp(x);
}

__device__ inline xlns32 xlns32d_sum(const xlns32 *a, size_t n)
{
	if (n == 0) return xlns32_zero;
	xlns32 sum = a[0];
	for (size_t i = 1; i < n; i++) {
		sum = xlns32d_add(sum, a[i]);
	}
	return sum;
}

__device__ inline xlns32 xlns32d_vec_dot(const xlns32 *a, const xlns32 *b, size_t n)
{
	if (n == 0) return xlns32_zero;
	xlns32 sum = xlns32d_mul(a[0], b[0]);
	for (size_t i = 1; i < n; i++) {
		sum = xlns32d_add(sum, xlns32d_mul(a[i], b[i]));
	}
	return sum;
}

__device__ inline float xlns32d_vec_dot_f32(const float *a, const float *b, size_t n)
{
	if (n == 0) return 0.0f;
	xlns32 sum = xlns32d_mul(fp2xlns32d(a[0]), fp2xlns32d(b[0]));
	for (size_t i = 1; i < n; i++) {
		xlns32 prod = xlns32d_mul(fp2xlns32d(a[i]), fp2xlns32d(b[i]));
		sum = xlns32d_add(sum, prod);
	}
	return xlns32d2fp(sum);
}

__device__ inline xlns32 xlns32d_max_array(const xlns32 *a, size_t n)
{
	if (n == 0) return xlns32_zero;
	xlns32 maxval = a[0];
	for (size_t i = 1; i < n; i++) {
		if (xlns32d_gt(a[i], maxval)) {
			maxval = a[i];
		}
	}
	return maxval;
}

__device__ inline xlns32 xlns32d_min_array(const xlns32 *a, size_t n)
{
	if (n == 0) return xlns32_zero;
	xlns32 minval = a[0];
	for (size_t i = 1; i < n; i++) {
		if (xlns32d_lt(a[i], minval)) {
			minval = a[i];
		}
	}
	return minval;
}

__device__ inline xlns32 xlns32d_sigmoid(xlns32 x)
{
	float fx = xlns32d2fp(x);
	float z = -fx * 1.4426950408889634f;
	xlns32_signed sz = (xlns32_signed)(z * xlns32_scale + (z >= 0 ? 0.5f : -0.5f));
	return xlns32_logsignmask - xlns32d_sb(sz);
}

__device__ inline xlns32 xlns32d_tanh(xlns32 x)
{
	float fx = xlns32d2fp(x);
	float result = tanh(fx);
	return fp2xlns32d(result);
}

__device__ inline xlns32 xlns32d_silu(xlns32 x)
{
	float fx = xlns32d2fp(x);
	float result = fx / (1.0f + exp(-fx));
	return fp2xlns32d(result);
}

__device__ inline xlns32 xlns32d_gelu(xlns32 x)
{
	float fx = xlns32d2fp(x);
	const float sqrt_2_over_pi = 0.7978845608f;
	float inner = sqrt_2_over_pi * (fx + 0.044715f * fx * fx * fx);
	float result = 0.5f * fx * (1.0f + tanh(inner));
	return fp2xlns32d(result);
}

__device__ inline xlns32 xlns32d_exp(xlns32 x)
{
	float fx = xlns32d2fp(x);
	return fp2xlns32d(exp(fx));
}

__device__ inline xlns32 xlns32d_log(xlns32 x)
{
	float fx = xlns32d2fp(x);
	if (fx <= 0.0f) return xlns32_zero;
	return fp2xlns32d(log(fx));
}

__device__ inline xlns32 xlns32d_exp2(xlns32 x)
{
	float fx = xlns32d2fp(x);
	return fp2xlns32d(pow(2.0, fx));
}

__device__ inline xlns32 xlns32d_log2(xlns32 x)
{
	float fx = xlns32d2fp(x);
	if (fx <= 0.0f) return xlns32_zero;
	return fp2xlns32d(log(fx) / log(2.0));
}

__device__ inline xlns32 xlns32d_pow(xlns32 base, xlns32 exponent)
{
	float fbase = xlns32d2fp(base);
	float fexp = xlns32d2fp(exponent);
	if (fbase <= 0.0f) return xlns32_zero;
	return fp2xlns32d(powf(fbase, fexp));
}

__device__ inline void xlns32d_softmax_exp(const xlns32 *a, xlns32 *c, size_t n)
{
	xlns32 maxval = xlns32d_max_array(a, n);
	for (size_t i = 0; i < n; i++) {
		float fx = xlns32d2fp(a[i]) - xlns32d2fp(maxval);
		c[i] = fp2xlns32d(exp(fx));
	}
}

__device__ inline void xlns32d_softmax(const xlns32 *a, xlns32 *c, size_t n,
				       xlns32 scale)
{
	if (n == 0) return;
	xlns32 maxval = xlns32_neg_inf;
	for (size_t i = 0; i < n; i++) {
		xlns32 v = a[i];
		if (v != xlns32_neg_inf) v = xlns32d_mul(v, scale);
		c[i] = v;
		if (xlns32d_gt(c[i], maxval)) maxval = c[i];
	}
	for (size_t i = 0; i < n; i++)
		c[i] = xlns32d_exp(xlns32d_sub(c[i], maxval));
	xlns32 total = xlns32d_sum(c, n);
	for (size_t i = 0; i < n; i++)
		c[i] = xlns32d_div(c[i], total);
}

__device__ inline void xlns32d_softmax_masked(const xlns32 *a, const xlns32 *mask,
					      xlns32 *c, size_t n, xlns32 scale)
{
	if (n == 0) return;
	xlns32 maxval = xlns32_neg_inf;
	for (size_t i = 0; i < n; i++) {
		xlns32 v = a[i];
		if (v != xlns32_neg_inf) {
			v = xlns32d_mul(v, scale);
			if (mask[i] == xlns32_neg_inf) v = xlns32_neg_inf;
			else if (!xlns32d_is_zero(mask[i])) v = xlns32d_add(v, mask[i]);
		}
		c[i] = v;
		if (xlns32d_gt(c[i], maxval)) maxval = c[i];
	}
	for (size_t i = 0; i < n; i++)
		c[i] = xlns32d_exp(xlns32d_sub(c[i], maxval));
	xlns32 total = xlns32d_sum(c, n);
	for (size_t i = 0; i < n; i++)
		c[i] = xlns32d_div(c[i], total);
}

__device__ inline void xlns32d_layernorm(const xlns32 *x, xlns32 *out,
					 const xlns32 *gamma, const xlns32 *beta,
					 size_t n, float eps)
{
	if (n == 0) return;
	xlns32 mean = xlns32d_sum(x, n);
	mean = xlns32d_div(mean, fp2xlns32d((float)n));

	xlns32 var = xlns32_zero;
	for (size_t i = 0; i < n; i++) {
		xlns32 diff = xlns32d_sub(x[i], mean);
		var = xlns32d_add(var, xlns32d_mul(diff, diff));
	}
	var = xlns32d_div(var, fp2xlns32d((float)n));

	xlns32 inv_std = fp2xlns32d(1.0f / sqrt(xlns32d2fp(var) + eps));
	for (size_t i = 0; i < n; i++) {
		out[i] = xlns32d_mul(xlns32d_sub(x[i], mean), inv_std);
		if (gamma) out[i] = xlns32d_mul(out[i], gamma[i]);
		if (beta) out[i] = xlns32d_add(out[i], beta[i]);
	}
}


#include <iostream>

class xlns32d_float {
    xlns32 x;
 public:
    friend __device__ xlns32d_float operator+(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator+(float, xlns32d_float );
    friend __device__ xlns32d_float operator+(xlns32d_float , float);
    friend __device__ xlns32d_float operator-(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator-(float, xlns32d_float );
    friend __device__ xlns32d_float operator-(xlns32d_float , float);
    friend __device__ xlns32d_float operator*(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator*(float, xlns32d_float );
    friend __device__ xlns32d_float operator*(xlns32d_float , float);
    friend __device__ xlns32d_float operator/(xlns32d_float , xlns32d_float );
    friend __device__ xlns32d_float operator/(float, xlns32d_float );
    friend __device__ xlns32d_float operator/(xlns32d_float , float);
    __device__ xlns32d_float operator=(float); //why not friend?
    friend __device__ xlns32 xlns32d_internal(xlns32d_float );
    friend __device__ float xlns32d_2float(xlns32d_float );
    friend __device__ xlns32d_float float2xlns32d_(float);
    //friend std::ostream& operator<<(std::ostream&, xlns32d_float );
    friend __device__ xlns32d_float operator-(xlns32d_float);
    friend __device__ xlns32d_float operator+=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator+=(xlns32d_float &, float);
    friend __device__ xlns32d_float operator-=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator-=(xlns32d_float &, float);
    friend __device__ xlns32d_float operator*=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator*=(xlns32d_float &, float);
    friend __device__ xlns32d_float operator/=(xlns32d_float &, xlns32d_float);
    friend __device__ xlns32d_float operator/=(xlns32d_float &, float);
    friend __device__ xlns32d_float sin(xlns32d_float);
    friend __device__ xlns32d_float cos(xlns32d_float);
    friend __device__ xlns32d_float exp(xlns32d_float);
    friend __device__ xlns32d_float log(xlns32d_float);
    friend __device__ xlns32d_float atan(xlns32d_float);
    friend __device__ xlns32d_float abs(xlns32d_float);
    friend __device__ xlns32d_float sqrt(xlns32d_float);
//    friend xlns32_float operator-(xlns32_float);
    friend __device__ int operator==(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (arg1.x == arg2.x);
      }
    friend __device__ int operator!=(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (arg1.x != arg2.x);
      }
    friend __device__ int operator<=(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)<=xlns32_canon(arg2.x));
      }
    friend __device__ int operator>=(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)>=xlns32_canon(arg2.x));
      }
    friend __device__ int operator<(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)<xlns32_canon(arg2.x));
      }
    friend __device__ int operator>(xlns32d_float arg1, xlns32d_float arg2)
      {
       return (xlns32_canon(arg1.x)>xlns32_canon(arg2.x));
      }
    friend __device__ int operator==(xlns32d_float arg1, float arg2);
    friend __device__ int operator!=(xlns32d_float arg1, float arg2);
    friend __device__ int operator<=(xlns32d_float arg1, float arg2);
    friend __device__ int operator>=(xlns32d_float arg1, float arg2);
    friend __device__ int operator<(xlns32d_float arg1, float arg2);
    friend __device__ int operator>(xlns32d_float arg1, float arg2);
  };




/*access function for internal representation*/

__device__ xlns32 xlns32d_internal(xlns32d_float y) {
    return y.x;
}

__device__ float xlns32d_2float(xlns32d_float y) {
	return xlns32d2fp(y.x);
}


//no caching
__device__ xlns32d_float float2xlns32d_(float y) {
	xlns32d_float z;
	z.x = fp2xlns32d(y);
	return z;
}


/*overload stream output << operator*/

//#include <ostream>
//std::ostream& operator<< (std::ostream& s, xlns32_float  y) {
//    return s << xlns32_2float(y);
//}

__device__ xlns32d_float operator-(xlns32d_float arg1) {
   xlns32d_float z;
   z.x=xlns32_neg(arg1.x);
   return z;
}



__device__ xlns32d_float operator+(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_add(arg1.x,arg2.x);
   return z;
}

__device__ xlns32d_float operator-(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_sub(arg1.x,arg2.x);
   return z;
}

__device__ xlns32d_float operator*(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_mul(arg1.x,arg2.x);
   return z;
}

__device__ xlns32d_float operator/(xlns32d_float arg1, xlns32d_float arg2) {
   xlns32d_float z;
   z.x=xlns32d_div(arg1.x,arg2.x);
   return z;
}


/*operators with auto type conversion*/

__device__ xlns32d_float operator+(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)+arg2;
}

__device__ xlns32d_float operator+(xlns32d_float arg1, float arg2) {
   return arg1+float2xlns32d_(arg2);
}


__device__ xlns32d_float operator-(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)-arg2;
}

__device__ xlns32d_float operator-(xlns32d_float arg1, float arg2) {
   return arg1-float2xlns32d_(arg2);
}

__device__ xlns32d_float operator*(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)*arg2;
}

__device__ xlns32d_float operator*(xlns32d_float arg1, float arg2) {
   return arg1*float2xlns32d_(arg2);
}


__device__ xlns32d_float operator/(float arg1, xlns32d_float arg2) {
   return float2xlns32d_(arg1)/arg2;
}

__device__ xlns32d_float operator/(xlns32d_float arg1, float arg2) {
   return arg1/float2xlns32d_(arg2);
}

/*comparisons with conversion seems not to inline OK*/

__device__ int operator==(xlns32d_float arg1, float arg2)
      {
       return arg1 == float2xlns32d_(arg2);
      }
__device__ int operator!=(xlns32d_float arg1, float arg2)
      {
       return arg1 != float2xlns32d_(arg2);
      }
__device__ int operator<=(xlns32d_float arg1, float arg2)
      {
       return arg1<=float2xlns32d_(arg2);
      }
__device__ int operator>=(xlns32d_float arg1, float arg2)
      {
       return arg1>=float2xlns32d_(arg2);
      }
__device__ int operator<(xlns32d_float arg1, float arg2)
      {
       return arg1<float2xlns32d_(arg2);
      }
__device__ int operator>(xlns32d_float arg1, float arg2)
      {
       return arg1>float2xlns32d_(arg2);
      }

/*With and without convert:  +=, -=, *=, and /= */

__device__ xlns32d_float operator+=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1+arg2;
   return arg1;
}

__device__ xlns32d_float operator+=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1+float2xlns32d_(arg2);
   return arg1;
}



__device__ xlns32d_float operator-=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1-arg2;
   return arg1;
}

__device__ xlns32d_float operator-=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1-float2xlns32d_(arg2);
   return arg1;
}


__device__ xlns32d_float operator*=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1*arg2;
   return arg1;
}

__device__ xlns32d_float operator*=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1*float2xlns32d_(arg2);
   return arg1;
}


__device__ xlns32d_float operator/=(xlns32d_float & arg1, xlns32d_float arg2) {
   arg1 = arg1/arg2;
   return arg1;
}

__device__ xlns32d_float operator/=(xlns32d_float & arg1, float arg2) {
   arg1 = arg1/float2xlns32d_(arg2);
   return arg1;
}



/*assignment with type conversion*/


__device__ xlns32d_float xlns32d_float::operator=(float rvalue) {
     x = float2xlns32d_(rvalue).x;
   return *this;
}



// functions computed ideally by convert to/from FP


inline __device__ xlns32d_float sin(xlns32d_float x)
{ 
	return float2xlns32d_(sin(xlns32d_2float(x))); 
}

inline __device__ xlns32d_float cos(xlns32d_float x)
{ 
	return float2xlns32d_(cos(xlns32d_2float(x))); 
}

// exp and log can be implemented more efficiently in LNS but 
// this is just cookie cutter ideal implementation at present

inline __device__ xlns32d_float exp(xlns32d_float x)
{ 
	return float2xlns32d_(exp(xlns32d_2float(x))); 
}

inline __device__ xlns32d_float log(xlns32d_float x)
{ 
	return float2xlns32d_(log(xlns32d_2float(x))); 
}

inline __device__ xlns32d_float atan(xlns32d_float x)
{ 
	return float2xlns32d_(atan(xlns32d_2float(x))); 
}

// the following have efficient macro implementations

inline __device__ xlns32d_float sqrt(xlns32d_float x)
{ 
	xlns32d_float result;
	result.x = xlns32_sqrt(x.x); 
	return result; 
}

inline __device__ xlns32d_float abs(xlns32d_float x)
{ 
	xlns32d_float result;
	result.x = xlns32_abs(x.x); 
	return result; 
}

