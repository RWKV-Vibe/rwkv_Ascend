# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

set(ALL_OP_LIST)
set(PUBLIC_DIRECTORY)
set(ALL_OP_LIST_TARGET)
file(GLOB subdirectories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(subdirectory ${subdirectories})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory})
    if(NOT subdirectory IN_LIST PUBLIC_DIRECTORY)
      list(APPEND ALL_OP_LIST ${subdirectory})
      add_subdirectory(${subdirectory})
    endif()
  endif() 
endforeach()