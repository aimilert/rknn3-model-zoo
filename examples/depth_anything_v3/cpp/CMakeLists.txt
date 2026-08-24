cmake_minimum_required(VERSION 3.15)
project(rknn_da3_base_demo LANGUAGES CXX)

set(TARGET_SOC rk3588 CACHE STRING "Use rknn3 API library for RK1820 proxy/RK3588 host")

if(TARGET_SOC STREQUAL "x86")
    set(DISABLE_RGA ON CACHE BOOL "Disable RGA for x86 host builds" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty rknn3_3rdparty)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils rknn3_utils)

add_executable(rknn_da3_base_demo main.cpp)
target_include_directories(rknn_da3_base_demo PRIVATE
    ${LIBRKNNAPI_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils
)
target_link_libraries(rknn_da3_base_demo PRIVATE imageutils fileutils ${LIBRKNNAPI} dl m)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(rknn_da3_base_demo PRIVATE log)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(rknn_da3_base_demo PRIVATE Threads::Threads)
endif()

set_target_properties(rknn_da3_base_demo PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
)

install(TARGETS rknn_da3_base_demo DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../model DESTINATION .)
