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

#ifndef _BTFY_H_
#define _BTFY_H_


#include <stdint.h>


#ifdef  __cplusplus
extern  "C" {
#endif



void btfy_128( uint64_t * fx , unsigned n_fx , unsigned scalar_a );

void i_btfy_128( uint64_t * fx , unsigned n_fx , unsigned scalar_a );


void btfy_64( uint64_t * fx , unsigned n_fx , unsigned scalar_a );

void i_btfy_64( uint64_t * fx , unsigned n_fx , unsigned scalar_a );



#ifdef  __cplusplus
}
#endif


#endif
