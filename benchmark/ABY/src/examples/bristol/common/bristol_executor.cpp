
#include "bristol_executor.h"
#include "../../../abycore/circuit/booleancircuits.h"
#include "../../../abycore/sharing/sharing.h"
#include <ENCRYPTO_utils/cbitvector.h>
#include <ENCRYPTO_utils/timer.h>
#include <vector>


int32_t exec_bristol_circuit(std::string& circuit, uint32_t input_gates, e_role role, const std::string& address, uint16_t port, seclvl seclvl,
                             uint32_t nvals, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing,
                             bool verbose, bool insecure) {
	uint32_t bitlen = 32;
	ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 10000);
    assert(sharing == S_BOOL || sharing == S_YAO);
	std::vector<Sharing*>& sharings = party->GetSharings();
    if(insecure) {
        assert(sharing == S_BOOL);
        sharings[S_BOOL]->SetPreCompPhaseValue(ePreCompInsecure);
    }

	crypto* crypt = new crypto(seclvl.symbits, (uint8_t*) const_seed);
	CBitVector key, verify;

    CBitVector inputs;
    inputs.Create(nvals * input_gates, crypt);

    // lol, that's how it's done in the other examples... ¯\_(ツ)_/¯
    BooleanCircuit* circ = (BooleanCircuit*) sharings[sharing]->GetCircuitBuildRoutine();


    share* s_in;
    s_in = circ->PutSIMDINGate(nvals, inputs.GetArr(), input_gates, CLIENT);

    auto out = circ->PutGateFromFile(circuit, s_in->get_wires(), nvals);

    party->ExecCircuit();



    PrintTimingsJson();
    PrintCommunicationJson();

	delete crypt;
	delete party;

	return 0;
}