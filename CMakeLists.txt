#
# Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
#
# This file is part of DistButterfly and is under the GNU General Public 
# License, which can be found in the LICENSE file in the root directory, or at
# <http://www.gnu.org/licenses/>.
#
cmake_minimum_required(VERSION 2.8.5) # for the new FindMPI module
project(DistButterfly)

set(DBF_VERSION_MAJOR 0)
set(DBF_VERSION_MINOR 5)

option(RELEASE "Avoid unnecessary assertions for faster runs." ON)
option(TIMING "Measure and print basic timing info." OFF)
option(BUILD_TESTS "Build the test drivers" ON)
option(AVOID_COMPLEX_MPI "Avoid complex MPI routines for robustness" ON)
mark_as_advanced(AVOID_COMPLEX_MPI)

find_package(MPI)
if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "A C++ MPI compiler is required but not found")
endif()
include_directories(${MPI_CXX_INCLUDE_PATH})

# Initialize CXXFLAGS.
# NOTE: Adapted from Matthias Vallentin's StackExchange post
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 ${MPI_CXX_COMPILE_FLAGS}")
if(RELEASE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast")
endif()
# Compiler-specific C++11 activation.
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  if(NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
    message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else()
  message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif()

if(MATH_LIBS)
  set(NEED_MATH FALSE)
else(MATH_LIBS)
  set(NEED_MATH TRUE)
endif()

if(NEED_MATH)
  set(MATH_DESC "BLAS/LAPACK link flags")
  # Look for default BLAS/LAPACK
  set(REFERENCE_REQUIRED LAPACK BLAS)
  find_library(BLAS_LIB
               NAMES blas blas.a blas.lib
               PATHS ${REFERENCE_ROOT})
  find_library(LAPACK_LIB
               NAMES lapack lapack.a lapack.lib
                     reflapack reflapack.a reflapack.lib
               PATHS ${REFERENCE_ROOT})
  set(REFERENCE_FOUND ON)
  set(MATH_LIBS "")
  foreach(NAME ${REFERENCE_REQUIRED})
    if( ${NAME}_LIB )
      message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
      list(APPEND MATH_LIBS ${${NAME}_LIB})
    else()
      message(STATUS "Could not find ${NAME}_LIB")
      set(REFERENCE_FOUND OFF)
    endif()
  endforeach()
  message(STATUS "REFERENCE_FOUND=${REFERENCE_FOUND}")
  if(REFERENCE_FOUND)
    message(STATUS "WARNING: Using reference BLAS/LAPACK.")
    message(STATUS "MATH_LIBS=${MATH_LIBS}")
  else()
    set(MATH_LIBS "" CACHE STRING ${MATH_DESC})
    message(FATAL_ERROR 
      "Could not find BLAS/LAPACK/BLACS/ScaLAPACK libs. Please provide the root directory of MKL with -DMKL_ROOT, the directory of reference implementations with -DREFERENCE_ROOT, or manually specify all math libraries with -DMATH_LIBS. There are numerous idiosyncratic library dependencies for BLAS/LAPACK/BLACS/ScaLAPACK, so you will almost certainly need to manually specify -DMATH_LIBS.")
  endif()
  # Append the standard math libraries to the link list.
  list(APPEND MATH_LIBS m)
endif()

# Attempt to detect the BLAS/LAPACK underscore conventions. 
# We currently only handle whether or not there is an underscore appended.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS})
check_function_exists(daxpy HAVE_DAXPY)
if(HAVE_DAXPY)
  set(BLAS_POST FALSE)
  set(BLAS_DEFS "")
else()
  check_function_exists(daxpy_ HAVE_DAXPY_POST)
  if(HAVE_DAXPY_POST)
    set(BLAS_POST TRUE)
    set(BLAS_DEFS "-DBLAS_POST")
  else()
    message(FATAL_ERROR "Could not determine BLAS format.")
  endif()
endif()
check_function_exists(dpotrf HAVE_DPOTRF)
if(HAVE_DPOTRF)
  set(LAPACK_POST FALSE)
  set(LAPACK_DEFS "")
else()
  check_function_exists(dpotrf_ HAVE_DPOTRF_POST)
  if(HAVE_DPOTRF_POST)
    set(LAPACK_POST TRUE)
    set(LAPACK_DEFS "-DLAPACK_POST")
  else()
    message(FATAL_ERROR "Could not determine LAPACK format.")
  endif()
endif()

# Look for MKL and MASS vectorization routines
check_function_exists(vdSin MKL)
check_function_exists(vsin MASS)

# Look for restrict support
include(CheckCXXSourceCompiles)
set(RESTRICT_CODE
    "int main(void)
     {
         int* RESTRICT a;
         return 0;
     }
    ")
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict__")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict__)
if(HAVE___restrict__)
  set(RESTRICT "__restrict__")
  message(STATUS "Using __restrict__ keyword.")
else()
  set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict")
  check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict)
  if(HAVE___restrict)
    set(RESTRICT "__restrict")
    message(STATUS "Using __restrict keyword.")
  else()
    set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=restrict")
    check_cxx_source_compiles("${RESTRICT_CODE}" HAVE_restrict)
    if(HAVE_restrict)
      set(RESTRICT "restrict")
      message(STATUS "Using restrict keyword.")
    else()
      set(RESTRICT "")
      message(STATUS "Could not find a restrict keyword.")
    endif()
  endif()
endif()

configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.cmake
               ${CMAKE_BINARY_DIR}/include/dist-butterfly/config.h)
install(FILES ${CMAKE_BINARY_DIR}/include/dist-butterfly/config.h
        DESTINATION include/dist-butterfly)

# We only have header files
file(GLOB_RECURSE DBF_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
     "include/*.h" "include/*.hpp")
set(DBF_SRC "${DBF_HEADERS}")

# Copy the headers into the build directory
set(COPIED_HEADERS "")
foreach(HEADER ${DBF_HEADERS})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}
            ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
  )
  list(APPEND COPIED_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/${HEADER}")
  get_filename_component(HEADER_PATH ${HEADER} PATH)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${HEADER} DESTINATION ${HEADER_PATH})
endforeach()

# Make sure the DBF headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

if(BUILD_TESTS)
  set(HTREE_TEST_DIR ${PROJECT_SOURCE_DIR}/test/htree)
  set(HTREE_TESTS ConstrainedHTreeWalker HTreeWalker)

  set(TRANSFORM_TEST_DIR ${PROJECT_SOURCE_DIR}/test/transform)
  set(TRANSFORM_TESTS Backproj-2d GenRadon-2d GenRadon-3d HypRadon-2d 
                      NonUniformFT-2d NonUniformFT-3d 
                      Random3DWaves UpWave-3d VariableUpWave-2d)
endif()

# Create a dummy library in order to be able to force the math libraries
# to be linked last
add_library(cmake-dummy-lib STATIC cmake/CMakeDummyFunction.cpp)
target_link_libraries(cmake-dummy-lib ${MATH_LIBS} ${MPI_CXX_LIBRARIES})

# Build the test drivers if necessary
set(MPI_LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
if(BUILD_TESTS)
  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/htree")
  foreach(TEST ${HTREE_TESTS})
    add_executable(${TEST} ${HTREE_TEST_DIR}/${TEST}.cpp ${COPIED_HEADERS})
    target_link_libraries(${TEST} cmake-dummy-lib)
    set_target_properties(${TEST} 
                          PROPERTIES OUTPUT_NAME ${TEST}
                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    if(MPI_LINK_FLAGS)
        set_target_properties(${TEST} PROPERTIES LINK_FLAGS ${MPI_LINK_FLAGS})
    endif()
    install(TARGETS ${TEST} DESTINATION bin/htree)
  endforeach()

  set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/transform")
  foreach(TEST ${TRANSFORM_TESTS})
    add_executable(${TEST} ${TRANSFORM_TEST_DIR}/${TEST}.cpp ${COPIED_HEADERS})
    target_link_libraries(${TEST} cmake-dummy-lib)
    set_target_properties(${TEST} 
                          PROPERTIES OUTPUT_NAME ${TEST}
                          RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    if(MPI_LINK_FLAGS)
      set_target_properties(${TEST} PROPERTIES LINK_FLAGS ${MPI_LINK_FLAGS})
    endif()
    install(TARGETS ${TEST} DESTINATION bin/transform)
  endforeach()
endif()
