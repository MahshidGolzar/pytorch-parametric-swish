cmake_minimum_required(VERSION 3.18)
project(parametric_swish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED Python3_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# Ask the already-installed `torch` package where its CMake config lives,
# rather than requiring a separate libtorch download.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch, os; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not import torch via ${Python3_EXECUTABLE}; activate the project venv first.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- Op library, loaded from Python via torch.ops.load_library() ----------
add_library(parametric_swish_ops MODULE
  csrc/parametric_swish.cpp
  csrc/register_ops.cpp
)
target_link_libraries(parametric_swish_ops PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(parametric_swish_ops PRIVATE csrc)
set_target_properties(parametric_swish_ops PROPERTIES PREFIX "")
if(WIN32)
  set_target_properties(parametric_swish_ops PROPERTIES SUFFIX ".pyd")
endif()

# --- Standalone C++ smoke test, no Python involved, for GDB/debugger use --
add_executable(smoke_test cpp_tests/smoke_test.cpp csrc/parametric_swish.cpp)
target_link_libraries(smoke_test PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(smoke_test PRIVATE csrc)

# libtorch ships as a set of DLLs; on Windows both targets need them next to
# the binary at run time. Copy them into the build output directory.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(target parametric_swish_ops smoke_test)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:${target}>
    )
  endforeach()
endif()
