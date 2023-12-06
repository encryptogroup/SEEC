/**
 \file 		pk-crypto.h
 \author 	michael.zohner@ec-spride.de
 \copyright	ABY - A Framework for Efficient Mixed-protocol Secure Two-party Computation
			Copyright (C) 2019 ENCRYPTO Group, TU Darmstadt
			This program is free software: you can redistribute it and/or modify
            it under the terms of the GNU Lesser General Public License as published
            by the Free Software Foundation, either version 3 of the License, or
            (at your option) any later version.
            ABY is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
            GNU Lesser General Public License for more details.
            You should have received a copy of the GNU Lesser General Public License
            along with this program. If not, see <http://www.gnu.org/licenses/>.
 \brief		Virtual class for public-key operations
 */

#ifndef PK_CRYPTO_H_
#define PK_CRYPTO_H_

#include "../typedefs.h"
#include "../constants.h"
#include "../utils.h"

class pk_crypto;
class num;
class fe;
class brickexp;

class pk_crypto {
public:
	pk_crypto(seclvl sp) {
		fe_bytelen = 0;
		order = 0;
		secparam = sp;
	}
	;
	virtual ~pk_crypto() {};
	virtual num* get_num() = 0;
	virtual num* get_rnd_num(uint32_t bitlen = 0) = 0;
	virtual fe* get_fe() = 0;
	virtual fe* get_rnd_fe() = 0;
	virtual fe* get_generator() = 0;
	virtual fe* get_rnd_generator() = 0;
	virtual uint32_t num_byte_size() = 0;
	virtual num* get_order() = 0;
	uint32_t fe_byte_size() {
		return fe_bytelen;
	}
	;
	virtual uint32_t get_field_size() = 0;
	virtual brickexp* get_brick(fe* gen) = 0;

protected:
	virtual void init(seclvl secparam, uint8_t* seed) = 0;
	uint32_t fe_bytelen;
	seclvl secparam;
	num* order;
};

//class number
class num {
public:
	num() {

	}
	;
	virtual ~num() {};
	virtual void set(num* src) = 0;
	virtual void set_si(int32_t src) = 0;
	virtual void set_add(num* a, num* b) = 0;
	virtual void set_sub(num* a, num* b) = 0;
	virtual void set_mul(num* a, num* b) = 0;
	virtual void mod(num* modulus) = 0;
	virtual void set_mul_mod(num* a, num* b, num* modulus) = 0;
	virtual void export_to_bytes(uint8_t* buf, uint32_t field_size) = 0;
	virtual void import_from_bytes(uint8_t* buf, uint32_t field_size) = 0;
	virtual void print() = 0;
};

//class field_element
class fe {
public:
	fe() {
	}
	;
	virtual ~fe() {};
	virtual void set(fe* src) = 0;
	virtual void set_mul(fe* a, fe* b) = 0;
	virtual void set_pow(fe* b, num* e) = 0;
	virtual void set_div(fe* a, fe* b) = 0;
	virtual void set_double_pow_mul(fe* b1, num* e1, fe* b2, num* e2) = 0;
	virtual void export_to_bytes(uint8_t* buf) = 0;
	virtual void import_from_bytes(uint8_t* buf) = 0;
	virtual void sample_fe_from_bytes(uint8_t* buf, uint32_t bytelen) = 0;
	virtual void print() = 0;
	virtual bool eq(fe* a) = 0;

protected:
	virtual void init() = 0;
};

class brickexp {
public:
	brickexp() {
	}
	;
	virtual ~brickexp() {};

	virtual void pow(fe* result, num* e) = 0;
};

#endif /* PK_CRYPTO_H_ */
