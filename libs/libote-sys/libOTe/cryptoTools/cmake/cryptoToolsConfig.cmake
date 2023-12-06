# these are just pass through config file for the ones that are placed in the build directory.

if(NOT DEFINED OC_CONFIG)

    if(MSVC)
        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set(OC_CONFIG "x64-Debug")
        else()
            set(OC_CONFIG "x64-Release")
        endif()
    elseif(APPLE)
        set(OC_CONFIG "osx")
    else()
        set(OC_CONFIG "linux")
    endif()
endif()
if(NOT DEFINED OC_THIRDPARTY_HINT)
    set(OC_THIRDPARTY_HINT "${CMAKE_CURRENT_LIST_DIR}/../out/install/${OC_CONFIG}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cryptoToolsFindBuildDir.cmake")
include("${CRYPTOTOOLS_BUILD_DIR}/cryptoToolsConfig.cmake")

