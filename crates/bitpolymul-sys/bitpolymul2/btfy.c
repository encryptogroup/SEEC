/*
Copyright (C) 2017 Ming-Shing Chen

This file is part of BitPolyMul.

BitPolyMul is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BitPolyMul is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with BitPolyMul.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <stdint.h>

#include "gfext_aesni.h"

#include "byte_inline_func.h"

#include "config_profile.h"

#include "string.h"



/////////////////////////////////////////////////
///
/// field isomorphism.  Cantor --> GF(2^128).
///
//////////////////////////////////////////////////////


#include "bitmat_prod.h"

#include "gf2128_cantor_iso.h"

static inline
__m128i gf_isomorphism_single_bit( unsigned ith_bit )
{
	return _mm_load_si128( (__m128i*) (&gfCantorto2128[ith_bit]) );
}

static inline
__m128i gf_isomorphism( uint64_t a_in_cantor )
{
	uint8_t a_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( a_iso , gfCantorto2128_8R , a_in_cantor );
	return _mm_load_si128( (__m128i*) a_iso );
}



/////////////////////////////////////////////////
///
/// Butterfly network. pclmulqdq version.
///
//////////////////////////////////////////////////////



///////  one layer /////////////////////

static inline
__m128i butterfly( __m128i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= _gf2ext128_mul_sse( poly[unit_2+i] , a );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+1] , _MM_HINT_T0 );
		poly[unit_2+i] ^= poly[i];
	}
	return a;
}



static inline
__m128i butterfly_avx2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= _gf2ext128_mul_2x1_avx2( poly[unit_2+i] , a );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+1] , _MM_HINT_T0 );
		poly[unit_2+i] ^= poly[i];
	}
	return a;
}


static inline
__m128i butterfly_avx2_b2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );
		_mm_prefetch( &poly[i] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		__m256i ap0 = _gf2ext128_mul_2x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext128_mul_2x1_avx2( p1 , a );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );

		_mm_prefetch( &poly[unit_2+i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+3] , _MM_HINT_T0 );

		q0 ^= ap0;
		q1 ^= ap1;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
		p0 ^= q0;
		p1 ^= q1;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
	}
	return a;
}


static inline
__m128i butterfly_avx2_b4( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=4) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );
		__m256i p2 = _mm256_load_si256( &poly[unit_2+i+2] );
		__m256i p3 = _mm256_load_si256( &poly[unit_2+i+3] );
		_mm_prefetch( &poly[i] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+3] , _MM_HINT_T0 );
		__m256i ap0 = _gf2ext128_mul_2x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext128_mul_2x1_avx2( p1 , a );
		__m256i ap2 = _gf2ext128_mul_2x1_avx2( p2 , a );
		__m256i ap3 = _gf2ext128_mul_2x1_avx2( p3 , a );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );
		__m256i q2 = _mm256_load_si256( &poly[i+2] );
		__m256i q3 = _mm256_load_si256( &poly[i+3] );

		_mm_prefetch( &poly[unit_2+i+4] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+5] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+6] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+7] , _MM_HINT_T0 );

		q0 ^= ap0;
		q1 ^= ap1;
		q2 ^= ap2;
		q3 ^= ap3;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
		_mm256_store_si256( &poly[i+2] , q2 );
		_mm256_store_si256( &poly[i+3] , q3 );
		p0 ^= q0;
		p1 ^= q1;
		p2 ^= q2;
		p3 ^= q3;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
		_mm256_store_si256( &poly[unit_2+i+2] , p2 );
		_mm256_store_si256( &poly[unit_2+i+3] , p3 );
	}
	return a;
}




static inline
__m128i i_butterfly( __m128i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+1] , _MM_HINT_T0 );
		poly[i] ^= _gf2ext128_mul_sse( poly[unit_2+i] , a );
	}
	return a;
}



static inline
__m128i i_butterfly_avx2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+1] , _MM_HINT_T0 );
		poly[i] ^= _gf2ext128_mul_2x1_avx2( poly[unit_2+i] , a );
	}
	return a;
}

static inline
__m128i i_butterfly_avx2_b2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );

		p0 ^= q0;
		p1 ^= q1;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
		_mm_prefetch( &poly[i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+3] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+3] , _MM_HINT_T0 );

		__m256i ap0 = _gf2ext128_mul_2x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext128_mul_2x1_avx2( p1 , a );

		q0 ^= ap0;
		q1 ^= ap1;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
	}
	return a;
}



static inline
__m128i i_butterfly_avx2_b4( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=4) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );
		__m256i p2 = _mm256_load_si256( &poly[unit_2+i+2] );
		__m256i p3 = _mm256_load_si256( &poly[unit_2+i+3] );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );
		__m256i q2 = _mm256_load_si256( &poly[i+2] );
		__m256i q3 = _mm256_load_si256( &poly[i+3] );

		p0 ^= q0;
		p1 ^= q1;
		p2 ^= q2;
		p3 ^= q3;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
		_mm256_store_si256( &poly[unit_2+i+2] , p2 );
		_mm256_store_si256( &poly[unit_2+i+3] , p3 );
		_mm_prefetch( &poly[i+4] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+5] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+6] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+7] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+4] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+5] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+6] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+7] , _MM_HINT_T0 );

		__m256i ap0 = _gf2ext128_mul_2x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext128_mul_2x1_avx2( p1 , a );
		__m256i ap2 = _gf2ext128_mul_2x1_avx2( p2 , a );
		__m256i ap3 = _gf2ext128_mul_2x1_avx2( p3 , a );

		q0 ^= ap0;
		q1 ^= ap1;
		q2 ^= ap2;
		q3 ^= ap3;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
		_mm256_store_si256( &poly[i+2] , q2 );
		_mm256_store_si256( &poly[i+3] , q3 );
	}
	return a;
}


//////////////////////////////////////////////////////////////


static inline
void __btfy( uint64_t * fx , unsigned st_unit_size , unsigned offset , unsigned n_terms , unsigned scalar_a )
{

	unsigned i= st_unit_size;

	__m256i * poly256 = (__m256i*) &fx[0];
#if 1
	for( ; i>3 ; i--) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st = (offset>>1)/unit;
		for(unsigned j= st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_avx2_b4( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}
	for( ; i>2 ; i--) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st = (offset>>1)/unit;
		for(unsigned j= st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_avx2_b2( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}
#endif
	for( ; i>1 ; i--) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st = (offset>>1)/unit;
		for(unsigned j= st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_avx2( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}
	__m128i * poly128 = (__m128i*) &fx[0];
	if( i>0 ) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st=offset/unit;
		for(unsigned j=st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly( poly128 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}
}


static inline
void __i_btfy( uint64_t * fx , unsigned end_unit_size , unsigned offset , unsigned n_terms , unsigned scalar_a )
{
	unsigned i=1;
	__m128i *poly128 = (__m128i*) &fx[0];
	for( ; i < 2; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st=offset/unit;
		for(unsigned j=st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly( poly128 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}

	__m256i *poly256 = (__m256i*) &fx[0];
	for( ; i < 3; i++) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st = (offset>>1)/unit;
		for(unsigned j=st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_avx2( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}
	for( ; i < 4; i++) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st = (offset>>1)/unit;
		for(unsigned j=st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_avx2_b2( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}

	for( ; i <= end_unit_size; i++) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		unsigned st = (offset>>1)/unit;
		for(unsigned j=st;j<st+num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_avx2_b4( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}

}






/////////////////////////////////////////////////////
//
// Public functions.
//
/////////////////////////////////////////////////////




#define _LOG_CACHE_SIZE_ 14


void btfy_128( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{

	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	unsigned i=log_n;

	__m256i * poly256 = (__m256i*) &fx[0];
	for( ; i> _LOG_CACHE_SIZE_ ; i--) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_avx2_b4( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}
	//// i == 15 or less
	unsigned unit = (1<<i);
	unsigned num = n_terms / unit;
	for(unsigned j=0;j<num;j++) {
		__btfy( fx , i , j*unit , unit , scalar_a );
	}
}




void i_btfy_128( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{
	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );

	unsigned n_terms = n_fx;

	{
	unsigned unit = ((1<<_LOG_CACHE_SIZE_) > n_fx)? n_fx : (1<<_LOG_CACHE_SIZE_);
	unsigned num = n_terms/unit;
	for(unsigned j=0;j<num;j++) {  __i_btfy(fx, _LOG_CACHE_SIZE_, j*unit, unit , scalar_a ); };
	}
	__m256i *poly256 = (__m256i*) &fx[0];
	for( unsigned i=_LOG_CACHE_SIZE_+1; i <= log_n; i++) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a > k ) ? gf_isomorphism_single_bit( (scalar_a-k-1)<<1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_avx2_b4( poly256 + j*unit , unit , diff_j<<1 , extra_a );
		}
	}

}





/////////////////////////////////////////////////////////////


#include "gf264_cantor_iso.h"

static inline
__m128i gf264_isomorphism_single_bit( unsigned ith_bit )
{
	return _mm_load_si128( (__m128i*) (&gfCantorto264[ith_bit<<1]) );
}

static inline
__m128i gf264_isomorphism( uint64_t a_in_cantor )
{
	uint8_t a_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( a_iso , gfCantorto264_8R , a_in_cantor );
	return _mm_load_si128( (__m128i*) a_iso );
}

//////////////////////////////////

static inline
__m128i butterfly_64_xmm( __m128i d , __m128i a )
{
	__m128i r0 = d^_gf2ext64_mul_hi_sse( d , a );
	__m128i r1 = r0^_mm_slli_si128( r0 , 8 );
	return r1;
}

static inline
__m128i i_butterfly_64_xmm( __m128i d , __m128i a )
{
	__m128i r0 = d^_mm_slli_si128( d , 8 );
	__m128i r1 = r0^_gf2ext64_mul_hi_sse( r0 , a );
	return r1;
}

static inline
__m128i butterfly_64_u2( __m128i * poly , unsigned ska , __m128i extra_a ) /// unit = 2>>1
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );
	poly[0] = butterfly_64_xmm( poly[0] , a );
	_mm_prefetch( &poly[1] , _MM_HINT_T0 );
	return a;
}

static inline
__m128i i_butterfly_64_u2( __m128i * poly , unsigned ska , __m128i extra_a ) /// unit = 2>>1
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );
	poly[0] = i_butterfly_64_xmm( poly[0] , a );
	_mm_prefetch( &poly[1] , _MM_HINT_T0 );
	return a;
}

////////////////////////////


static inline
__m256i butterfly_64_ymm( __m256i d , __m128i a )
{
	__m128i d0 = _mm256_castsi256_si128( d );
	__m128i d1 = _mm256_extracti128_si256( d , 1 );
	d0 ^= _gf2ext64_mul_2x1_sse( d1 , a );
	d1 ^= d0;
	__m256i r0 = _mm256_castsi128_si256( d0 );
	__m256i r1 = _mm256_inserti128_si256( r0 , d1 , 1 );
	return r1;
}

static inline
__m256i i_butterfly_64_ymm( __m256i d , __m128i a )
{
	__m128i d0 = _mm256_castsi256_si128( d );
	__m128i d1 = _mm256_extracti128_si256( d , 1 );
	d1 ^= d0;
	d0 ^= _gf2ext64_mul_2x1_sse( d1 , a );
	__m256i r0 = _mm256_castsi128_si256( d0 );
	__m256i r1 = _mm256_inserti128_si256( r0 , d1 , 1 );
	return r1;
}

static inline
__m128i butterfly_64_u4( __m256i * poly , unsigned ska , __m128i extra_a ) /// unit = 4>>2
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );
	poly[0] = butterfly_64_ymm( poly[0] , a );
	_mm_prefetch( &poly[1] , _MM_HINT_T0 );
	return a;
}

static inline
__m128i i_butterfly_64_u4( __m256i * poly , unsigned ska , __m128i extra_a ) /// unit = 4>>2
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );
	poly[0] = i_butterfly_64_ymm( poly[0] , a );
	_mm_prefetch( &poly[1] , _MM_HINT_T0 );
	return a;
}


//////////////////////////////////////////////////


static inline
__m128i butterfly_64_avx2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a ) /// unit >= 8>>2
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= _gf2ext64_mul_4x1_avx2( poly[unit_2+i] , a );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+1] , _MM_HINT_T0 );
		poly[unit_2+i] ^= poly[i];
	}
	return a;
}


static inline
__m128i butterfly_64_avx2_b2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );
		_mm_prefetch( &poly[i] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		__m256i ap0 = _gf2ext64_mul_4x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext64_mul_4x1_avx2( p1 , a );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );

		_mm_prefetch( &poly[unit_2+i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+3] , _MM_HINT_T0 );

		q0 ^= ap0;
		q1 ^= ap1;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
		p0 ^= q0;
		p1 ^= q1;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
	}
	return a;
}


static inline
__m128i butterfly_64_avx2_b4( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=4) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );
		__m256i p2 = _mm256_load_si256( &poly[unit_2+i+2] );
		__m256i p3 = _mm256_load_si256( &poly[unit_2+i+3] );
		_mm_prefetch( &poly[i] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+3] , _MM_HINT_T0 );
		__m256i ap0 = _gf2ext64_mul_4x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext64_mul_4x1_avx2( p1 , a );
		__m256i ap2 = _gf2ext64_mul_4x1_avx2( p2 , a );
		__m256i ap3 = _gf2ext64_mul_4x1_avx2( p3 , a );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );
		__m256i q2 = _mm256_load_si256( &poly[i+2] );
		__m256i q3 = _mm256_load_si256( &poly[i+3] );

		_mm_prefetch( &poly[unit_2+i+4] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+5] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+6] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+7] , _MM_HINT_T0 );

		q0 ^= ap0;
		q1 ^= ap1;
		q2 ^= ap2;
		q3 ^= ap3;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
		_mm256_store_si256( &poly[i+2] , q2 );
		_mm256_store_si256( &poly[i+3] , q3 );
		p0 ^= q0;
		p1 ^= q1;
		p2 ^= q2;
		p3 ^= q3;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
		_mm256_store_si256( &poly[unit_2+i+2] , p2 );
		_mm256_store_si256( &poly[unit_2+i+3] , p3 );
	}
	return a;
}





static inline
__m128i i_butterfly_64_avx2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		_mm_prefetch( &poly[i+1] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+1] , _MM_HINT_T0 );
		poly[i] ^= _gf2ext64_mul_4x1_avx2( poly[unit_2+i] , a );
	}
	return a;
}


static inline
__m128i i_butterfly_64_avx2_b2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=2) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );

		p0 ^= q0;
		p1 ^= q1;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
		_mm_prefetch( &poly[i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+3] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+2] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+3] , _MM_HINT_T0 );

		__m256i ap0 = _gf2ext64_mul_4x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext64_mul_4x1_avx2( p1 , a );

		q0 ^= ap0;
		q1 ^= ap1;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
	}
	return a;
}



static inline
__m128i i_butterfly_64_avx2_b4( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	__m128i a = extra_a;
	a ^= gf264_isomorphism( ska );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i+=4) {
		__m256i p0 = _mm256_load_si256( &poly[unit_2+i] );
		__m256i p1 = _mm256_load_si256( &poly[unit_2+i+1] );
		__m256i p2 = _mm256_load_si256( &poly[unit_2+i+2] );
		__m256i p3 = _mm256_load_si256( &poly[unit_2+i+3] );

		__m256i q0 = _mm256_load_si256( &poly[i] );
		__m256i q1 = _mm256_load_si256( &poly[i+1] );
		__m256i q2 = _mm256_load_si256( &poly[i+2] );
		__m256i q3 = _mm256_load_si256( &poly[i+3] );

		p0 ^= q0;
		p1 ^= q1;
		p2 ^= q2;
		p3 ^= q3;
		_mm256_store_si256( &poly[unit_2+i] , p0 );
		_mm256_store_si256( &poly[unit_2+i+1] , p1 );
		_mm256_store_si256( &poly[unit_2+i+2] , p2 );
		_mm256_store_si256( &poly[unit_2+i+3] , p3 );
		_mm_prefetch( &poly[i+4] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+5] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+6] , _MM_HINT_T0 );
		_mm_prefetch( &poly[i+7] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+4] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+5] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+6] , _MM_HINT_T0 );
		_mm_prefetch( &poly[unit_2+i+7] , _MM_HINT_T0 );

		__m256i ap0 = _gf2ext64_mul_4x1_avx2( p0 , a );
		__m256i ap1 = _gf2ext64_mul_4x1_avx2( p1 , a );
		__m256i ap2 = _gf2ext64_mul_4x1_avx2( p2 , a );
		__m256i ap3 = _gf2ext64_mul_4x1_avx2( p3 , a );

		q0 ^= ap0;
		q1 ^= ap1;
		q2 ^= ap2;
		q3 ^= ap3;
		_mm256_store_si256( &poly[i] , q0 );
		_mm256_store_si256( &poly[i+1] , q1 );
		_mm256_store_si256( &poly[i+2] , q2 );
		_mm256_store_si256( &poly[i+3] , q3 );
	}
	return a;
}



///////////////////////////////////////////



void btfy_64( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{

	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	unsigned i=log_n;

	uint64_t * poly = fx;
	for( ; i>4; i-- ) {
		unsigned unit = (1<<i); /// u >= 8
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_64_avx2_b4(  (__m256i*)(poly + j*unit) , unit>>2 , diff_j<<1 , extra_a );
		}
	}
	for( ; i>3; i-- ) {
		unsigned unit = (1<<i); /// u >= 8
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_64_avx2_b2(  (__m256i*)(poly + j*unit) , unit>>2 , diff_j<<1 , extra_a );
		}
	}
	for( ; i>2; i-- ) {
		unsigned unit = (1<<i); /// u >= 8
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_64_avx2(  (__m256i*)(poly + j*unit) , unit>>2 , diff_j<<1 , extra_a );
		}
	}
	for( ; i>1; i-- ) {
		unsigned unit = (1<<i); /// u = 4
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_64_u4( (__m256i*)(poly + j*unit) , diff_j<<1 , extra_a );
		}
	}
	for( ; i>0; i-- ) {
		unsigned unit = (1<<i); /// u = 2
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = butterfly_64_u2( (__m128i*)(poly + j*unit) , diff_j<<1 , extra_a );
		}
	}
}

void i_btfy_64( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{
	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	uint64_t * poly = fx;
	unsigned i=1;
	for( ; i<2; i++ ) {
		unsigned unit = (1<<i); /// u = 2
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_64_u2( (__m128i*)(poly + j*unit) , diff_j<<1 , extra_a );
		}
	}
	for( ; i<3; i++ ) {
		unsigned unit = (1<<i); /// u = 4
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_64_u4( (__m256i*)(poly + j*unit) , diff_j<<1 , extra_a );
		}
	}
	for( ; i<4; i++ ) {
		unsigned unit = (1<<i); /// u >= 8
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_64_avx2(  (__m256i*)(poly + j*unit) , unit>>2 , diff_j<<1 , extra_a );
		}
	}
	for( ; i<5; i++ ) {
		unsigned unit = (1<<i); /// u >= 8
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_64_avx2_b2(  (__m256i*)(poly + j*unit) , unit>>2 , diff_j<<1 , extra_a );
		}
	}
	for( ; i<=log_n; i++ ) {
		unsigned unit = (1<<i); /// u >= 8
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = ( scalar_a > k )? gf264_isomorphism_single_bit( scalar_a - k - 1 ) : _mm_setzero_si128();

		unsigned last_j = 0;
		for(unsigned j=0;j<num;j++) {
			unsigned diff_j = j^last_j;
			last_j = j;
			extra_a = i_butterfly_64_avx2_b4(  (__m256i*)(poly + j*unit) , unit>>2 , diff_j<<1 , extra_a );
		}
	}
}
