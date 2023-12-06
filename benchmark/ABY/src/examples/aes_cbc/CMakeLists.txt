
add_executable(aes_cbc_test aes_cbc_test.cpp common/aes_cbc_circuit.cpp)
target_link_libraries(aes_cbc_test ABY::aby ENCRYPTO_utils::encrypto_utils)
