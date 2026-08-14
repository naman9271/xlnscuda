// 16-bit XLNS for CUDA
// copyright 1999-2025 Mark G. Arnold
// these routines 
//    ran on 16-bit Turbo C/C++ (the file may have CR/LFs from that system) 
//    were used in my PhD research and for several later papers on 32-bit LNS
// they were ported to Linux gcc and g++ around 2015 on 32-bit x86
// they were ported again for 64-bit arch in 2025, 
//    modified for 16-bit similar to bfloat (see xlns32.cpp for original float-like code)
//    with the xlns16_ideal option
// they were copied from xlns16.cpp and modified as __device__ CUDA routines
// they are based on similar math foundation (Gaussian logs, sb and db) as Python xlns,
//    but use different internal storage format:
//    +------+-------------------------+
//    | sign | int(log2) . frac(log2)  |
//    +------+-------------------------+
//    the int(log2) is not twos complement; it is offset (logsignmask XORed)
//    for the 16-bit format in this file, this is roughly similar to bfloat16
//    1 sign bit, 8 int(log2) bits, 7 frac(log2) bits
//    there is an exact representation of 0.0, but no subnormals or NaNs


/* PORTABLE CODE STARTS HERE*/

inline __device__ xlns16 xlns16d_overflow(xlns16 xlns16_x, xlns16 xlns16_y, xlns16 xlns16_temp)
{       
	if (xlns16_logsignmask&xlns16_temp)
		return (xlns16_signmask&(xlns16_x^xlns16_y));
	else
		return (xlns16_signmask&(xlns16_x^xlns16_y))| xlns16_logmask;
}

inline __device__ xlns16 xlns16d_mul(xlns16 x, xlns16 y)
{
   xlns16 xlns16_temp;
   xlns16_temp=(xlns16_logmask&(x))+(xlns16_logmask&(y))-xlns16_logsignmask; 
   return  (xlns16_signmask&(xlns16_temp)) ? xlns16d_overflow(x,y,xlns16_temp) 
                                       :(xlns16_signmask&(x^y))|xlns16_temp;
}

inline __device__ xlns16 xlns16d_div(xlns16 x, xlns16 y)
{
   xlns16 xlns16_temp;
   xlns16_temp=(xlns16_logmask&(x))-(xlns16_logmask&(y))+xlns16_logsignmask; 
   return  (xlns16_signmask&(xlns16_temp)) ? xlns16d_overflow(x,y,xlns16_temp) 
                                       :(xlns16_signmask&(x^y))|xlns16_temp;
}

#ifdef xlns16_ideal
  #define xlns16d_sb xlns16d_sb_ideal
  #define xlns16d_db xlns16d_db_ideal
  #include <math.h>
  inline __device__ xlns16 xlns16d_sb_ideal(xlns16_signed z)
  {
	return ((xlns16) ((log(1+ pow(2.0, ((double) z) / xlns16_scale) )/log(2.0))*xlns16_scale+.5));
  }
  inline __device__ xlns16 xlns16d_db_ideal(xlns16_signed z)
  {
	return ((xlns16_signed) ((log( pow(2.0, ((double) z) / xlns16_scale) - 1 )/log(2.0))*xlns16_scale+.5));
  }
#else
  #define xlns16d_sb xlns16d_sb_premit
  #define xlns16d_db xlns16d_db_premit
  #define xlns16_F 7

  #include <math.h>
  inline __device__ xlns16 xlns16d_db_ideal(xlns16_signed z)  //only for singularity
  {
	return ((xlns16_signed) ((log( pow(2.0, ((double) z) / xlns16_scale) - 1 )/log(2.0))*xlns16_scale+.5));
  }
  inline __device__ xlns16 xlns16d_mitch(xlns16_signed z)
  {
     return (((1<<xlns16_F)+(z&((1<<xlns16_F)-1)))>>(-(z>>xlns16_F)));
  }

  inline __device__ xlns16_signed xlns16d_sb_premit_neg(xlns16_signed zi)   //was called premitchnpi(zi): assumes zi<=0
  {
    xlns16_signed postcond;
    xlns16_signed z;
    postcond = (zi <= -(3<<xlns16_F))? 0: (zi >= -(3<<(xlns16_F-2))? -1: +1);
    //z = ((zi<<3) + (zi^0xffff) + 16)>>3; //for some reason this does not work in CUDA
    z = -((((-zi)<<3) + zi - 8)>>3); //but this produces results == python xlnsconf.lpvip 
    return (zi==0)?1<<xlns16_F: xlns16d_mitch(z) + postcond;
  }

  inline __device__ xlns16 xlns16d_db_premit_neg(xlns16_signed z)   //assumes zi<0 like python xlnsconf.lpvip
  {
    xlns16_signed precond;
    precond = (z < -(2<<xlns16_F))?
                    5<<(xlns16_F-3):                //  0.625
                    (z >> 2) + (9 << (xlns16_F-3));//  .25*zr + 9/8
    return (-z >= 1<<xlns16_F)?-xlns16d_mitch(z+precond): xlns16d_db_ideal(-z)+z; // use ideal for singularity
  }
  inline __device__ xlns16_signed xlns16d_sb_premit(xlns16_signed zi)   //assumes zi>=0 as needed here
  {
    return xlns16d_sb_premit_neg(-zi)+zi;
  }
  inline __device__ xlns16 xlns16d_db_premit(xlns16_signed z)   //assumes zi>0 as needed here
  {
    return xlns16d_db_premit_neg(-z)+z;
  } 
#endif



#ifdef xlns16_alt

__device__ inline xlns16 xlns16d_add(xlns16 x, xlns16 y)
{
    xlns16 minxyl, maxxy, xl, yl, usedb, adjust, adjustez;
    xlns16_signed z;

    xl = x & xlns16_logmask;

    yl = y & xlns16_logmask;
    minxyl = (yl>xl) ? xl : yl;
    maxxy  = (xl>yl) ? x  : y;
    z = minxyl - (maxxy&xlns16_logmask);
    usedb = xlns16_signmask&(x^y);

    #ifdef xlns16_ideal

     float pm1 = usedb ? -1.0 : 1.0;

     adjust = z+((xlns16_signed)(log(pm1+pow(2.0,-((double)z)/xlns16_scale))/log(2.0)*xlns16_scale+.5));

    #else
     
     //adjust = usedb ? xlns16d_db_neg(z) : 
     
     //                 xlns16d_sb_neg(z);
 
      #ifdef xlns16_altopt
    xlns16_signed precond = (usedb==0) ? ((-z)>>3) :          // -.125*z
 
                (z < -(2<<xlns16_F)) ? 5<<(xlns16_F-3):        //  0.625
                                (z >> 2) + (9 << (xlns16_F-3));//  .25*z + 9/8
     xlns16_signed postcond = (z <= -(3<<xlns16_F)) ? 0:
 
                            z >= -(3<<(xlns16_F-2)) ? -(1<<(xlns16_F-6)) :
 
                                                      +(1<<(xlns16_F-6));
     xlns16_signed mitch = (-z >= 1<<xlns16_F)||(usedb==0) ? xlns16d_mitch(z+precond) :
 
                                          -xlns16d_db_ideal(-z)-z; // use ideal for singularity
     adjust = usedb ? -mitch : (z==0) ? 1<<xlns16_F : mitch + postcond;
      #else
       adjust = usedb ? xlns16d_db_premit_neg(z) : 
                        xlns16d_sb_premit_neg(z); 
      #endif
    #endif

    adjustez = (z < -xlns16_esszer) ? 0 : adjust;

    return ((z==0) && usedb) ?
 
                     xlns16_zero :

                      xlns16d_mul(maxxy, xlns16_logsignmask + adjustez);
}


#else



//++++ X-X ERROR fixed

__device__ xlns16 xlns16d_add(xlns16 x, xlns16 y)
{
	xlns16 t;
	xlns16_signed z;

	z = (x&xlns16_logmask) - (y&xlns16_logmask);
	if (z<0)
	{
		z = -z;
		t = x;
		x = y;
		y = t;
	}
	if (xlns16_signmask&(x^y))
	{
		if (z == 0)
			return xlns16_zero;
		if (z < xlns16_esszer)
			return xlns16_neg(y + xlns16d_db(z));
		else
			return xlns16_neg(y+z);
	}
	else
	{
		return y + xlns16d_sb(z);
	}
}

#endif

#define xlns16d_sub(x,y) xlns16d_add(x,xlns16_neg(y))

__device__ inline int xlns16d_is_zero(xlns16 x)
{
	return (xlns16_abs(x) == xlns16_zero);
}

__device__ inline int xlns16d_is_negative(xlns16 x)
{
	return (xlns16_sign(x) != 0) && !xlns16d_is_zero(x);
}

__device__ inline int xlns16d_is_positive(xlns16 x)
{
	return (xlns16_sign(x) == 0) && !xlns16d_is_zero(x);
}

__device__ inline int xlns16d_gt(xlns16 a, xlns16 b)
{
	return xlns16_canon(a) > xlns16_canon(b);
}

__device__ inline int xlns16d_lt(xlns16 a, xlns16 b)
{
	return xlns16_canon(a) < xlns16_canon(b);
}

__device__ inline int xlns16d_eq(xlns16 a, xlns16 b)
{
	return a == b;
}

__device__ inline int xlns16d_ge(xlns16 a, xlns16 b)
{
	return xlns16_canon(a) >= xlns16_canon(b);
}

__device__ inline int xlns16d_le(xlns16 a, xlns16 b)
{
	return xlns16_canon(a) <= xlns16_canon(b);
}

__device__ inline xlns16 xlns16d_max(xlns16 a, xlns16 b)
{
	return xlns16d_gt(a, b) ? a : b;
}

__device__ inline xlns16 xlns16d_min(xlns16 a, xlns16 b)
{
	return xlns16d_lt(a, b) ? a : b;
}

__device__ inline xlns16 xlns16d_copysign(xlns16 x, xlns16 y)
{
	return xlns16_abs(x) | xlns16_sign(y);
}

__device__ inline xlns16 xlns16d_fma(xlns16 a, xlns16 b, xlns16 c)
{
	return xlns16d_add(xlns16d_mul(a, b), c);
}

__device__ inline xlns16 xlns16d_relu(xlns16 x)
{
	return xlns16d_is_negative(x) ? xlns16_zero : x;
}

/*END OF PORTABLE CODE*/

/*START OF PORTABLE CODE THAT DEPENDS ON <math.h>*/

#include <math.h>
#include <stddef.h>

/*END OF PORTABLE CODE THAT DEPENDS ON <math.h>*/


__device__ xlns16 fp2xlns16d(double x)
{
	if ((x>-2.938747e-39)&&(x<2.938747e-39))
		return(xlns16_zero);
        else if (x> 3.40282286e+38)
		return(xlns16_pos_inf);
        else if (x< -3.40282286e+38)
		return(xlns16_neg_inf);
	else if (x > 0.0)
		return xlns16_abs((xlns16_signed) ((log(x)/log(2.0))*xlns16_scale))
		       ^xlns16_logsignmask;
	else
		return (((xlns16_signed) ((log(fabs(x))/log(2.0))*xlns16_scale))
			  |xlns16_signmask)^xlns16_logsignmask;
}

__device__ float xlns16d2fp(xlns16 x)
{
	if (xlns16_abs(x) == xlns16_zero)
		return (0.0);
	else if (xlns16_sign(x))
		return (float) (-pow(2.0,((double) (((xlns16_signed) (xlns16_abs(x)-xlns16_logsignmask))))
					/((float) xlns16_scale)));
	else {
		return (float) (+pow(2.0,((double) (((xlns16_signed) (xlns16_abs(x)-xlns16_logsignmask))))
					/((float) xlns16_scale)));
	}
}

__device__ inline xlns16 xlns16d_from_float(float x)
{
	return fp2xlns16d(x);
}

__device__ inline float xlns16d_to_float(xlns16 x)
{
	return xlns16d2fp(x);
}

__device__ inline xlns16 xlns16d_sum(const xlns16 *a, size_t n)
{
	if (n == 0) return xlns16_zero;
	xlns16 sum = a[0];
	for (size_t i = 1; i < n; i++) {
		sum = xlns16d_add(sum, a[i]);
	}
	return sum;
}

__device__ inline xlns16 xlns16d_vec_dot(const xlns16 *a, const xlns16 *b, size_t n)
{
	if (n == 0) return xlns16_zero;
	xlns16 sum = xlns16d_mul(a[0], b[0]);
	for (size_t i = 1; i < n; i++) {
		sum = xlns16d_add(sum, xlns16d_mul(a[i], b[i]));
	}
	return sum;
}

__device__ inline float xlns16d_vec_dot_f32(const float *a, const float *b, size_t n)
{
	if (n == 0) return 0.0f;
	xlns16 sum = xlns16d_mul(fp2xlns16d(a[0]), fp2xlns16d(b[0]));
	for (size_t i = 1; i < n; i++) {
		xlns16 prod = xlns16d_mul(fp2xlns16d(a[i]), fp2xlns16d(b[i]));
		sum = xlns16d_add(sum, prod);
	}
	return xlns16d2fp(sum);
}

__device__ inline xlns16 xlns16d_max_array(const xlns16 *a, size_t n)
{
	if (n == 0) return xlns16_zero;
	xlns16 maxval = a[0];
	for (size_t i = 1; i < n; i++) {
		if (xlns16d_gt(a[i], maxval)) {
			maxval = a[i];
		}
	}
	return maxval;
}

__device__ inline xlns16 xlns16d_min_array(const xlns16 *a, size_t n)
{
	if (n == 0) return xlns16_zero;
	xlns16 minval = a[0];
	for (size_t i = 1; i < n; i++) {
		if (xlns16d_lt(a[i], minval)) {
			minval = a[i];
		}
	}
	return minval;
}

__device__ inline xlns16 xlns16d_sigmoid(xlns16 x)
{
	float fx = xlns16d2fp(x);
	float z = -fx * 1.4426950408889634f;
	xlns16_signed sz = (xlns16_signed)(z * xlns16_scale + (z >= 0 ? 0.5f : -0.5f));
	return xlns16_logsignmask - xlns16d_sb(sz);
}

__device__ inline xlns16 xlns16d_silu(xlns16 x)
{
	return xlns16d_mul(x, xlns16d_sigmoid(x));
}

__device__ inline xlns16 xlns16d_exp(xlns16 x)
{
	float fx = xlns16d2fp(x);
	return fp2xlns16d(exp(fx));
}

__device__ inline xlns16 xlns16d_log(xlns16 x)
{
	float fx = xlns16d2fp(x);
	if (fx <= 0.0f) return xlns16_zero;
	return fp2xlns16d(log(fx));
}

__device__ inline xlns16 xlns16d_tanh(xlns16 x)
{
	const xlns16 exp2x = xlns16d_exp(xlns16d_mul(x, xlns16_two));
	return xlns16d_div(xlns16d_sub(exp2x, xlns16_one),
			   xlns16d_add(exp2x, xlns16_one));
}

__device__ inline xlns16 xlns16d_gelu(xlns16 x)
{
	const xlns16 sqrt2_over_pi = fp2xlns16d(0.7978845608);
	const xlns16 coeff = fp2xlns16d(0.044715);
	const xlns16 x3 = xlns16d_mul(xlns16d_mul(x, x), x);
	const xlns16 inner = xlns16d_mul(sqrt2_over_pi,
				 xlns16d_add(x, xlns16d_mul(coeff, x3)));
	return xlns16d_mul(xlns16d_mul(xlns16_half, x),
			   xlns16d_add(xlns16_one, xlns16d_tanh(inner)));
}

__device__ inline xlns16 xlns16d_exp2(xlns16 x)
{
	float fx = xlns16d2fp(x);
	return fp2xlns16d(pow(2.0, fx));
}

__device__ inline xlns16 xlns16d_log2(xlns16 x)
{
	float fx = xlns16d2fp(x);
	if (fx <= 0.0f) return xlns16_zero;
	return fp2xlns16d(log(fx) / log(2.0));
}

__device__ inline xlns16 xlns16d_pow(xlns16 base, xlns16 exponent)
{
	float fbase = xlns16d2fp(base);
	float fexp = xlns16d2fp(exponent);
	if (fbase <= 0.0f) return xlns16_zero;
	return fp2xlns16d(powf(fbase, fexp));
}

__device__ inline void xlns16d_softmax_exp(const xlns16 *a, xlns16 *c, size_t n)
{
	xlns16 maxval = xlns16d_max_array(a, n);
	for (size_t i = 0; i < n; i++) {
		float fx = xlns16d2fp(a[i]) - xlns16d2fp(maxval);
		c[i] = fp2xlns16d(exp(fx));
	}
}

__device__ inline void xlns16d_softmax(const xlns16 *a, xlns16 *c, size_t n,
				       xlns16 scale)
{
	if (n == 0) return;
	xlns16 maxval = xlns16_neg_inf;
	for (size_t i = 0; i < n; i++) {
		xlns16 v = a[i];
		if (v != xlns16_neg_inf) v = xlns16d_mul(v, scale);
		c[i] = v;
		if (xlns16d_gt(c[i], maxval)) maxval = c[i];
	}
	for (size_t i = 0; i < n; i++)
		c[i] = xlns16d_exp(xlns16d_sub(c[i], maxval));
	xlns16 total = xlns16d_sum(c, n);
	for (size_t i = 0; i < n; i++)
		c[i] = xlns16d_div(c[i], total);
}

__device__ inline void xlns16d_softmax_masked(const xlns16 *a, const xlns16 *mask,
					      xlns16 *c, size_t n, xlns16 scale)
{
	if (n == 0) return;
	xlns16 maxval = xlns16_neg_inf;
	for (size_t i = 0; i < n; i++) {
		xlns16 v = a[i];
		if (v != xlns16_neg_inf) {
			v = xlns16d_mul(v, scale);
			if (mask[i] == xlns16_neg_inf) v = xlns16_neg_inf;
			else if (!xlns16d_is_zero(mask[i])) v = xlns16d_add(v, mask[i]);
		}
		c[i] = v;
		if (xlns16d_gt(c[i], maxval)) maxval = c[i];
	}
	for (size_t i = 0; i < n; i++)
		c[i] = xlns16d_exp(xlns16d_sub(c[i], maxval));
	xlns16 total = xlns16d_sum(c, n);
	for (size_t i = 0; i < n; i++)
		c[i] = xlns16d_div(c[i], total);
}

__device__ inline void xlns16d_layernorm(const xlns16 *x, xlns16 *out,
					 const xlns16 *gamma, const xlns16 *beta,
					 size_t n, float eps)
{
	if (n == 0) return;
	xlns16 mean = xlns16d_sum(x, n);
	mean = xlns16d_div(mean, fp2xlns16d((float)n));

	xlns16 var = xlns16_zero;
	for (size_t i = 0; i < n; i++) {
		xlns16 diff = xlns16d_sub(x[i], mean);
		var = xlns16d_add(var, xlns16d_mul(diff, diff));
	}
	var = xlns16d_div(var, fp2xlns16d((float)n));

	xlns16 inv_std = fp2xlns16d(1.0f / sqrt(xlns16d2fp(var) + eps));
	for (size_t i = 0; i < n; i++) {
		out[i] = xlns16d_mul(xlns16d_sub(x[i], mean), inv_std);
		if (gamma) out[i] = xlns16d_mul(out[i], gamma[i]);
		if (beta) out[i] = xlns16d_add(out[i], beta[i]);
	}
}


#include <iostream>

class xlns16d_float {
    xlns16 x;
 public:
    friend __device__ xlns16d_float operator+(xlns16d_float , xlns16d_float );
    friend __device__ xlns16d_float operator+(float, xlns16d_float );
    friend __device__ xlns16d_float operator+(xlns16d_float , float);
    friend __device__ xlns16d_float operator-(xlns16d_float , xlns16d_float );
    friend __device__ xlns16d_float operator-(float, xlns16d_float );
    friend __device__ xlns16d_float operator-(xlns16d_float , float);
    friend __device__ xlns16d_float operator*(xlns16d_float , xlns16d_float );
    friend __device__ xlns16d_float operator*(float, xlns16d_float );
    friend __device__ xlns16d_float operator*(xlns16d_float , float);
    friend __device__ xlns16d_float operator/(xlns16d_float , xlns16d_float );
    friend __device__ xlns16d_float operator/(float, xlns16d_float );
    friend __device__ xlns16d_float operator/(xlns16d_float , float);
    __device__ xlns16d_float operator=(float); //why not friend?
    friend __device__ xlns16 xlns16d_internal(xlns16d_float );
    friend __device__ float xlns16d_2float(xlns16d_float );
    friend __device__ xlns16d_float float2xlns16d_(float);
    //friend std::ostream& operator<<(std::ostream&, xlns16d_float );
    friend __device__ xlns16d_float operator-(xlns16d_float);
    friend __device__ xlns16d_float operator+=(xlns16d_float &, xlns16d_float);
    friend __device__ xlns16d_float operator+=(xlns16d_float &, float);
    friend __device__ xlns16d_float operator-=(xlns16d_float &, xlns16d_float);
    friend __device__ xlns16d_float operator-=(xlns16d_float &, float);
    friend __device__ xlns16d_float operator*=(xlns16d_float &, xlns16d_float);
    friend __device__ xlns16d_float operator*=(xlns16d_float &, float);
    friend __device__ xlns16d_float operator/=(xlns16d_float &, xlns16d_float);
    friend __device__ xlns16d_float operator/=(xlns16d_float &, float);
    friend __device__ xlns16d_float sin(xlns16d_float);
    friend __device__ xlns16d_float cos(xlns16d_float);
    friend __device__ xlns16d_float exp(xlns16d_float);
    friend __device__ xlns16d_float log(xlns16d_float);
    friend __device__ xlns16d_float atan(xlns16d_float);
    friend __device__ xlns16d_float abs(xlns16d_float);
    friend __device__ xlns16d_float sqrt(xlns16d_float);
//    friend xlns16_float operator-(xlns16_float);
    friend __device__ int operator==(xlns16d_float arg1, xlns16d_float arg2)
      {
       return (arg1.x == arg2.x);
      }
    friend __device__ int operator!=(xlns16d_float arg1, xlns16d_float arg2)
      {
       return (arg1.x != arg2.x);
      }
    friend __device__ int operator<=(xlns16d_float arg1, xlns16d_float arg2)
      {
       return (xlns16_canon(arg1.x)<=xlns16_canon(arg2.x));
      }
    friend __device__ int operator>=(xlns16d_float arg1, xlns16d_float arg2)
      {
       return (xlns16_canon(arg1.x)>=xlns16_canon(arg2.x));
      }
    friend __device__ int operator<(xlns16d_float arg1, xlns16d_float arg2)
      {
       return (xlns16_canon(arg1.x)<xlns16_canon(arg2.x));
      }
    friend __device__ int operator>(xlns16d_float arg1, xlns16d_float arg2)
      {
       return (xlns16_canon(arg1.x)>xlns16_canon(arg2.x));
      }
    friend __device__ int operator==(xlns16d_float arg1, float arg2);
    friend __device__ int operator!=(xlns16d_float arg1, float arg2);
    friend __device__ int operator<=(xlns16d_float arg1, float arg2);
    friend __device__ int operator>=(xlns16d_float arg1, float arg2);
    friend __device__ int operator<(xlns16d_float arg1, float arg2);
    friend __device__ int operator>(xlns16d_float arg1, float arg2);
  };




/*access function for internal representation*/

__device__ xlns16 xlns16d_internal(xlns16d_float y) {
    return y.x;
}

__device__ float xlns16d_2float(xlns16d_float y) {
	return xlns16d2fp(y.x);
}


//no caching
__device__ xlns16d_float float2xlns16d_(float y) {
	xlns16d_float z;
	z.x = fp2xlns16d(y);
	return z;
}


/*overload stream output << operator*/

//#include <ostream>
//std::ostream& operator<< (std::ostream& s, xlns16_float  y) {
//    return s << xlns16_2float(y);
//}

__device__ xlns16d_float operator-(xlns16d_float arg1) {
   xlns16d_float z;
   z.x=xlns16_neg(arg1.x);
   return z;
}



__device__ xlns16d_float operator+(xlns16d_float arg1, xlns16d_float arg2) {
   xlns16d_float z;
   z.x=xlns16d_add(arg1.x,arg2.x);
   return z;
}

__device__ xlns16d_float operator-(xlns16d_float arg1, xlns16d_float arg2) {
   xlns16d_float z;
   z.x=xlns16d_sub(arg1.x,arg2.x);
   return z;
}

__device__ xlns16d_float operator*(xlns16d_float arg1, xlns16d_float arg2) {
   xlns16d_float z;
   z.x=xlns16d_mul(arg1.x,arg2.x);
   return z;
}

__device__ xlns16d_float operator/(xlns16d_float arg1, xlns16d_float arg2) {
   xlns16d_float z;
   z.x=xlns16d_div(arg1.x,arg2.x);
   return z;
}


/*operators with auto type conversion*/

__device__ xlns16d_float operator+(float arg1, xlns16d_float arg2) {
   return float2xlns16d_(arg1)+arg2;
}

__device__ xlns16d_float operator+(xlns16d_float arg1, float arg2) {
   return arg1+float2xlns16d_(arg2);
}


__device__ xlns16d_float operator-(float arg1, xlns16d_float arg2) {
   return float2xlns16d_(arg1)-arg2;
}

__device__ xlns16d_float operator-(xlns16d_float arg1, float arg2) {
   return arg1-float2xlns16d_(arg2);
}

__device__ xlns16d_float operator*(float arg1, xlns16d_float arg2) {
   return float2xlns16d_(arg1)*arg2;
}

__device__ xlns16d_float operator*(xlns16d_float arg1, float arg2) {
   return arg1*float2xlns16d_(arg2);
}


__device__ xlns16d_float operator/(float arg1, xlns16d_float arg2) {
   return float2xlns16d_(arg1)/arg2;
}

__device__ xlns16d_float operator/(xlns16d_float arg1, float arg2) {
   return arg1/float2xlns16d_(arg2);
}

/*comparisons with conversion seems not to inline OK*/

__device__ int operator==(xlns16d_float arg1, float arg2)
      {
       return arg1 == float2xlns16d_(arg2);
      }
__device__ int operator!=(xlns16d_float arg1, float arg2)
      {
       return arg1 != float2xlns16d_(arg2);
      }
__device__ int operator<=(xlns16d_float arg1, float arg2)
      {
       return arg1<=float2xlns16d_(arg2);
      }
__device__ int operator>=(xlns16d_float arg1, float arg2)
      {
       return arg1>=float2xlns16d_(arg2);
      }
__device__ int operator<(xlns16d_float arg1, float arg2)
      {
       return arg1<float2xlns16d_(arg2);
      }
__device__ int operator>(xlns16d_float arg1, float arg2)
      {
       return arg1>float2xlns16d_(arg2);
      }

/*With and without convert:  +=, -=, *=, and /= */

__device__ xlns16d_float operator+=(xlns16d_float & arg1, xlns16d_float arg2) {
   arg1 = arg1+arg2;
   return arg1;
}

__device__ xlns16d_float operator+=(xlns16d_float & arg1, float arg2) {
   arg1 = arg1+float2xlns16d_(arg2);
   return arg1;
}



__device__ xlns16d_float operator-=(xlns16d_float & arg1, xlns16d_float arg2) {
   arg1 = arg1-arg2;
   return arg1;
}

__device__ xlns16d_float operator-=(xlns16d_float & arg1, float arg2) {
   arg1 = arg1-float2xlns16d_(arg2);
   return arg1;
}


__device__ xlns16d_float operator*=(xlns16d_float & arg1, xlns16d_float arg2) {
   arg1 = arg1*arg2;
   return arg1;
}

__device__ xlns16d_float operator*=(xlns16d_float & arg1, float arg2) {
   arg1 = arg1*float2xlns16d_(arg2);
   return arg1;
}


__device__ xlns16d_float operator/=(xlns16d_float & arg1, xlns16d_float arg2) {
   arg1 = arg1/arg2;
   return arg1;
}

__device__ xlns16d_float operator/=(xlns16d_float & arg1, float arg2) {
   arg1 = arg1/float2xlns16d_(arg2);
   return arg1;
}



/*assignment with type conversion*/


__device__ xlns16d_float xlns16d_float::operator=(float rvalue) {
     x = float2xlns16d_(rvalue).x;
   return *this;
}



// functions computed ideally by convert to/from FP


inline __device__ xlns16d_float sin(xlns16d_float x)
{ 
	return float2xlns16d_(sin(xlns16d_2float(x))); 
}

inline __device__ xlns16d_float cos(xlns16d_float x)
{ 
	return float2xlns16d_(cos(xlns16d_2float(x))); 
}

// exp and log can be implemented more efficiently in LNS but 
// this is just cookie cutter ideal implementation at present

inline __device__ xlns16d_float exp(xlns16d_float x)
{ 
	return float2xlns16d_(exp(xlns16d_2float(x))); 
}

inline __device__ xlns16d_float log(xlns16d_float x)
{ 
	return float2xlns16d_(log(xlns16d_2float(x))); 
}

inline __device__ xlns16d_float atan(xlns16d_float x)
{ 
	return float2xlns16d_(atan(xlns16d_2float(x))); 
}

// the following have efficient macro implementations

inline __device__ xlns16d_float sqrt(xlns16d_float x)
{ 
	xlns16d_float result;
	result.x = xlns16_sqrt(x.x); 
	return result; 
}

inline __device__ xlns16d_float abs(xlns16d_float x)
{ 
	xlns16d_float result;
	result.x = xlns16_abs(x.x); 
	return result; 
}

