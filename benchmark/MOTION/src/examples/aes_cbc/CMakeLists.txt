add_executable(aes_cbc aescbc_main.cpp common/aescbc.cpp)

if (NOT MOTION_BUILD_BOOST_FROM_SOURCES)
    find_package(Boost
            COMPONENTS
            program_options
            REQUIRED)
endif ()

target_link_libraries(aes_cbc
        MOTION::motion
        Boost::program_options
        )
