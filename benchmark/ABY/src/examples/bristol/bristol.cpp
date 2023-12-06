
//Utility libs
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
//ABY Party class
#include "../../abycore/aby/abyparty.h"

#include "common/bristol_executor.h"

int32_t read_test_options(int32_t* argcp, char*** argvp, e_role* role, uint32_t* bitlen, uint32_t* nvals,
		uint32_t* secparam, std::string* address, uint16_t* port, e_sharing* sharing, bool* verbose, uint32_t* nthreads,
		bool* insecure, std::string* circuit, uint32_t * input_gates) {

	uint32_t int_role = 0, int_port = 0, int_sharing = 0;

	parsing_ctx options[] = { { (void*) &int_role, T_NUM, "r", "Role: 0/1", true, false },
			{ (void*) nvals, T_NUM, "n", "SIMD size; Number of parallel operation elements", false, false },
			{ (void*) bitlen, T_NUM, "b", "Bit-length, default 32", false, false },
			{ (void*) secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false },
			{ (void*) address, T_STR, "a", "IP-address, default: localhost", false, false },
			{ (void*) &int_port, T_NUM, "p", "Port, default: 7766", false, false },
			{ (void*) &int_sharing, T_NUM, "g", "Sharing in which the AES circuit should be evaluated [0: BOOL, 1: YAO], default: BOOL", false, false },
			{ (void*) verbose, T_FLAG, "v", "Do not print the result of the evaluation, default: off", false, false },
			{ (void*) nthreads, T_NUM, "t", "Number of threads, default: 1", false, false },
            { (void*) insecure, T_FLAG, "i", "Insecure preprocessing, default: false", false, false },
            { (void*) circuit, T_STR, "c", "Bristol circuit to execute", true, false},
            { (void*) input_gates, T_NUM, "d", "Input gates in circuit", true, false},
    };

	if (!parse_options(argcp, argvp, options, sizeof(options) / sizeof(parsing_ctx))) {
		print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
		std::cout << "Exiting" << std::endl;
		exit(0);
	}

	assert(int_role < 2);
	*role = (e_role) int_role;

	if (int_port != 0) {
		assert(int_port < 1 << (sizeof(uint16_t) * 8));
		*port = (uint16_t) int_port;
	}

	assert(int_sharing < S_LAST);
	assert(int_sharing != S_ARITH);
	*sharing = (e_sharing) int_sharing;

	//delete options;

	return 1;
}

int main(int argc, char** argv) {
	e_role role;
	uint32_t bitlen = 32, nvals = 1, secparam = 128, nthreads = 1;
    uint32_t  input_gates;
	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	bool verbose = false;
	e_mt_gen_alg mt_alg = MT_OT;
    bool insecure = false;
    std::string circuit;

	e_sharing sharing = S_BOOL;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address, &port, &sharing, &verbose,
                      &nthreads, &insecure, &circuit, &input_gates);

	seclvl seclvl = get_sec_lvl(secparam);

	exec_bristol_circuit(circuit, input_gates, role, address, port, seclvl, nvals, nthreads, mt_alg, sharing, verbose, insecure);

	return 0;
}

