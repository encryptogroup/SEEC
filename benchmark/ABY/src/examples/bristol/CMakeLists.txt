
add_executable(bristol bristol.cpp common/bristol_executor.cpp)
target_link_libraries(bristol ABY::aby ENCRYPTO_utils::encrypto_utils)
