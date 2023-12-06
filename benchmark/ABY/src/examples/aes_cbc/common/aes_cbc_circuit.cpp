/**
 \file 		aescircuit.cpp
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
#include "aes_cbc_circuit.h"
#include "../../../abycore/circuit/booleancircuits.h"
#include "../../../abycore/sharing/sharing.h"
#include <ENCRYPTO_utils/cbitvector.h>
#include <ENCRYPTO_utils/timer.h>
#include <vector>

static uint32_t* pos_even;
static uint32_t* pos_odd;


int32_t test_aes_circuit(e_role role, const std::string& address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
		e_mt_gen_alg mt_alg, e_sharing sharing, [[maybe_unused]] bool verbose, bool use_vec_ands, bool expand_in_sfe, bool client_only,
        uint32_t input_blocks, bool insecure) {
	uint32_t bitlen = 32;
	uint32_t aes_key_bits;
	ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 10000);
    assert(sharing == S_BOOL);
	std::vector<Sharing*>& sharings = party->GetSharings();
    if(insecure) {
        sharings[S_BOOL]->SetPreCompPhaseValue(ePreCompInsecure);
    }

	crypto* crypt = new crypto(seclvl.symbits, (uint8_t*) const_seed);
	CBitVector key, verify;

	aes_key_bits = crypt->get_aes_key_bytes() * 8;
    std::vector<CBitVector> inputs(input_blocks);
    for(auto& inp: inputs) {
        inp.Create(AES_BITS, crypt);
    }

//	verify.Create(AES_BITS * nvals);
	key.CreateBytes(AES_KEY_BYTES);

	uint8_t aes_test_key[AES_KEY_BYTES];
	srand(7438);
	for(uint32_t i = 0; i < AES_KEY_BYTES; i++) {
		aes_test_key[i] = (uint8_t) (rand() % 256);
	}
	key.Copy(aes_test_key, 0, AES_KEY_BYTES);

    Circuit* circ = sharings[sharing]->GetCircuitBuildRoutine();
    //Circuit build routine works for Boolean circuits only right now
    assert(circ->GetCircuitType() == C_BOOLEAN);

    share *s_key;

    std::vector<share*> s_in_all(input_blocks);
    for(uint32_t i = 0; i < input_blocks; i++) {
        s_in_all[i] = circ->PutINGate(inputs[i].GetArr(), AES_BITS, CLIENT);
    }

    e_role key_inputter;
    if(client_only) {
        key_inputter = CLIENT;
    } else {
        key_inputter = SERVER;
    }
    s_key = circ->PutINGate(aes_test_key, AES_KEY_BITS, key_inputter);

//    s_key = circ->PutRepeaterGate(input_blocks,s_key);

    auto s_out = BuildAESCircuit(s_in_all, s_key, (BooleanCircuit*) circ);

    party->ExecCircuit();

    CBitVector out(input_blocks * AES_BITS);

    if(role == CLIENT) {
        for(auto block = 0; block < input_blocks; block++) {
            auto out_ptr = s_out[block]->get_clear_value_ptr();
            out.SetBytes(out_ptr, block * AES_BYTES, AES_BYTES);
        }
//        out.PrintHex();
    }


//    output = s_ciphertext->get_clear_value_ptr();
//
//    out.SetBytes(output, 0L, (uint64_t) AES_BYTES * nvals);
////	}
//
//	verify_AES_encryption(input.GetArr(), key.GetArr(), nvals, verify.GetArr(), crypt);



#ifndef BATCH
	std::cout << "Testing AES encryption in " << get_sharing_name(sharing) << " sharing: " << std::endl;
#endif
	for (uint32_t i = 0; i < nvals; i++) {
#ifndef BATCH
		if(!verbose) {
			std::cout << "(" << i << ") Input:\t";
			input.PrintHex(i * AES_BYTES, (i + 1) * AES_BYTES);
			std::cout << "(" << i << ") Key:\t";
			key.PrintHex(0, AES_KEY_BYTES);
			std::cout << "(" << i << ") Circ:\t";
			out.PrintHex(i * AES_BYTES, (i + 1) * AES_BYTES);
			std::cout << "(" << i << ") Verify:\t";
			verify.PrintHex(i * AES_BYTES, (i + 1) * AES_BYTES);
		}
#endif
//		assert(verify.IsEqual(out, i*AES_BITS, (i+1)*AES_BITS));
	}
#ifndef BATCH
	std::cout << "all tests succeeded" << std::endl;
#else

    PrintTimingsJson();
    PrintCommunicationJson();

//	std::cout << party->GetTiming(P_SETUP) << "\t" << party->GetTiming(P_GARBLE) << "\t" << party->GetTiming(P_ONLINE) << "\t" << party->GetTiming(P_TOTAL) <<
//			"\t" << party->GetSentData(P_TOTAL) + party->GetReceivedData(P_TOTAL) << "\t";
//	if(sharing == S_YAO_REV) {
//		std::cout << sharings[S_YAO]->GetNumNonLinearOperations() +sharings[S_YAO_REV]->GetNumNonLinearOperations() << "\t" << sharings[S_YAO]->GetMaxCommunicationRounds()<< std::endl;
//	} else  {
//		std::cout << sharings[sharing]->GetNumNonLinearOperations()	<< "\t" << sharings[sharing]->GetMaxCommunicationRounds()<< std::endl;
//	}
#endif
	delete crypt;
	delete party;

//	free(output);
	return 0;
}

std::vector<share*> BuildAESCircuit(std::vector<share*> in_blocks, share* key, BooleanCircuit* circ) {
    share *chaining_state = circ->PutINGate(uint32_t(0), AES_BITS, CLIENT);
    std::vector<share *> out_shares(in_blocks.size());

    for (auto i = 0; i < in_blocks.size(); i++) {
        auto inp = circ->PutXORGate(in_blocks[i], chaining_state);
        auto inp_ids = inp->get_wires();
        for(auto key_id: key->get_wires()) {
            inp_ids.push_back(key_id);
        }
        auto t = circ->PutGateFromFile(std::string("../../bin/circ/aes_128.aby"), inp_ids);
        delete chaining_state;
        chaining_state = new boolshare(t, circ);
        out_shares[i] = new boolshare(circ->PutOUTGate(t, CLIENT), circ);
    }

    return out_shares;
}

void verify_AES_encryption(uint8_t* input, uint8_t* key, uint32_t nvals, uint8_t* out, crypto* crypt) {
	AES_KEY_CTX* aes_key = (AES_KEY_CTX*) malloc(sizeof(AES_KEY_CTX));
	crypt->init_aes_key(aes_key, key);
	for (uint32_t i = 0; i < nvals; i++) {
		crypt->encrypt(aes_key, out + i * AES_BYTES, input + i * AES_BYTES, AES_BYTES);
	}
	free(aes_key);
}
