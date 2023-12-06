
#ifndef __BRISTOL_EXECUTOR_H_
#define __BRISTOL_EXECUTOR_H_

#include "../../../abycore/circuit/circuit.h"
#include "../../../abycore/aby/abyparty.h"
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <cassert>

class BooleanCircuit;




int32_t exec_bristol_circuit(std::string& circuit, uint32_t input_gates, e_role role, const std::string& address, uint16_t port, seclvl seclvl,
                             uint32_t nvals, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing,
                             bool verbose = false, bool insecure = false);



#endif /* __BRISTOL_EXECUTOR_H_ */
