/**
 \file 		aescircuit.h
 \author 	michael.zohner@ec-spride.de
 \copyright	ABY - A Framework for Efficient Mixed-protocol Secure Two-party Computation
			Copyright (C) 2019 Engineering Cryptographic Protocols Group, TU Darmstadt
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
 \brief		Implementation of AESCiruit
 */

#ifndef __AESCIRCUIT_H_
#define __AESCIRCUIT_H_

#include "../../../abycore/circuit/circuit.h"
#include "../../../abycore/aby/abyparty.h"
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <cassert>

class BooleanCircuit;

// If you change these values and want to test the functionallity with test_aes_circuit,
// you will have to change the AES_BITS, AES_BYTES, AES_KEY_BITS and AES_KEY_BYTES definitions
// on the constant.h definition on the encrypto utils as well.
// WARNING: Currently a correct running of the algorithms cannot be guaranteed if these values are changed.
// There might be some work to do.
#define AES_STATE_KEY_BITS 128
#define AES_STATE_SIZE_BITS 128


//Testing functions
void verify_AES_encryption(uint8_t* input, uint8_t* key, uint32_t nvals, uint8_t* out, crypto* crypt);
/**
 \param		role the role of the user; possible roles: "CLIENT" and "SERVER"
 \param		adress the adress of the server the client connects to
 \param 	port the port of the server the client connects to
 \param		seclvl	the definition of the security level the SFE should be using, see on <ENCRYPTO_utils/crypto/crypto.h>
				to get more information
 \param		nvals the amount of concurrent encryptions to be calculated
 \param		nthreads the amount of threads used
 \param 		mt_alg the Oblivious Extension algorithm to be used; see e_mt_gen_alg in the ABYConstants.h for possible algorithms
 \param		sharing the sharing algorithm to be used; see e_sharing in the ABYConstants.h for possible algorithms
 \param		verbose if true some output values will be suppressed for printing; default is false
 \param		use_vec_ands if true the vector AND optimization for AES circuit for Bool sharing will be usedM default is false
 \param		expand_in_sfe if true the key will be expanded in the SFE, otherwise the key will be expanded before the SFE; default is false
 \param		client_only if true both the key and the values will be inputted by the client; default is false
*/
int32_t test_aes_circuit(e_role role, const std::string& address, uint16_t port, seclvl seclvl,
                         uint32_t nvals, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing,
                         bool verbose = false, bool use_vec_ands = false,
                         bool expand_in_sfe = false, bool client_only = false,
                         uint32_t input_bytes = 16, bool insecure = false);

std::vector<share*> BuildAESCircuit(std::vector<share*> val, share* key, BooleanCircuit* circ);

#endif /* __AESCIRCUIT_H_ */
