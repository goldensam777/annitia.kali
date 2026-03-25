cmake_minimum_required(VERSION 3.18)
project(annitia C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra")

# Options
option(ANNITIA_BUILD_CUDA "Build with CUDA backend (from k-mamba)" OFF)
option(ANNITIA_BUILD_TESTS "Build tests" ON)

# k-mamba submodule
set(KMAMBA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/k-mamba)

# Configure k-mamba build flags to match ours
set(KMAMBA_BUILD_CPU ON CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_CUDA ${ANNITIA_BUILD_CUDA} CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${KMAMBA_DIR} kmamba_build)

# Dependencies
find_package(OpenBLAS REQUIRED)

# annitia sources
set(ANNITIA_SOURCES
    src/annitia.c
    src/survival_loss.c
    src/masld_data.c
    src/mask_utils.c
)

add_library(annitia STATIC ${ANNITIA_SOURCES})

target_include_directories(annitia PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KMAMBA_DIR}/include
    ${KMAMBA_DIR}/src
    ${OpenBLAS_INCLUDE_DIRS}
)

target_link_libraries(annitia PUBLIC
    k-mamba
    ${OpenBLAS_LIBRARIES}
)

if(ANNITIA_BUILD_CUDA)
    target_compile_definitions(annitia PUBLIC KMAMBA_BUILD_CUDA=1)
endif()

# Shared library for Python ctypes wrapper
add_library(annitia_shared SHARED ${ANNITIA_SOURCES})
set_target_properties(annitia_shared PROPERTIES OUTPUT_NAME "annitia")

target_include_directories(annitia_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KMAMBA_DIR}/include
    ${KMAMBA_DIR}/src
    ${OpenBLAS_INCLUDE_DIRS}
)

target_link_libraries(annitia_shared PUBLIC
    k-mamba
    ${OpenBLAS_LIBRARIES}
)

if(ANNITIA_BUILD_CUDA)
    target_compile_definitions(annitia_shared PUBLIC KMAMBA_BUILD_CUDA=1)
endif()

# Preprocess binary (CSV → binary)
add_executable(annitia_preprocess src/csv_parser.c)
target_include_directories(annitia_preprocess PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KMAMBA_DIR}/include
)
target_link_libraries(annitia_preprocess PRIVATE m)

# Train binary
add_executable(annitia_train src/train_main.c)
target_link_libraries(annitia_train PRIVATE annitia)

# Predict binary
add_executable(annitia_predict src/predict_main.c)
target_link_libraries(annitia_predict PRIVATE annitia)

# Tests
if(ANNITIA_BUILD_TESTS)
    enable_testing()
    add_executable(test_survival_loss tests/test_survival_loss.c)
    target_link_libraries(test_survival_loss PRIVATE annitia)
    add_test(NAME test_survival_loss COMMAND test_survival_loss)

    add_executable(test_data_loader tests/test_data_loader.c)
    target_link_libraries(test_data_loader PRIVATE annitia)
    add_test(NAME test_data_loader COMMAND test_data_loader)
endif()
