if(NOT DEFINED DYNCALL_SRC OR NOT EXISTS "${DYNCALL_SRC}/CMakeLists.txt")
  message(FATAL_ERROR "DYNCALL_SRC must name a dyncall source tree")
endif()

macro(replace_required variable old_text new_text description)
  string(FIND "${${variable}}" "${old_text}" old_position)
  string(FIND "${${variable}}" "${new_text}" new_position)
  if(old_position EQUAL -1 AND new_position EQUAL -1)
    message(FATAL_ERROR
      "Pinned dyncall layout changed while applying ${description}")
  endif()
  string(REPLACE "${old_text}" "${new_text}" ${variable} "${${variable}}")
endmacro()

set(patch_target "${DYNCALL_SRC}")
set(old "if(MSVC)")
set(new "if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL \"Clang\")")
foreach(rel_file CMakeLists.txt dyncall/CMakeLists.txt dyncallback/CMakeLists.txt)
  set(f "${patch_target}/${rel_file}")
  file(READ "${f}" content)
  replace_required(content "${old}" "${new}" "MSVC condition in ${rel_file}")
  file(WRITE "${f}" "${content}")
endforeach()

set(root_cmake "${patch_target}/CMakeLists.txt")
file(READ "${root_cmake}" content)
replace_required(
  content
  "cmake_minimum_required (VERSION 2.6)"
  "cmake_minimum_required(VERSION 3.10...3.31)"
  "minimum CMake version")
set(unpatched_content "${content}")
string(REGEX REPLACE
  "COMMENT \"Assembling \\$\\{ASM_FILE\\} ---> \\\\\"\\$\\{CMAKE_ASM_COMPILER\\}\\\\\" \\$\\{ASM_INCLUDE_DIRECTORIES\\} -o \\$\\{OBJ_FILE\\} \\$\\{ASM_FILE\\}\"\\+[\t ]*COMMENT "
  "COMMENT "
  content "${content}")
if(content STREQUAL unpatched_content)
  message(FATAL_ERROR
    "Pinned dyncall layout changed while applying duplicated assembler diagnostic")
endif()
replace_required(
  content
  "elseif(CMAKE_COMPILER_IS_CLANG)\nelseif"
  "elseif(CMAKE_COMPILER_IS_CLANG)\n  enable_language(ASM)\nelseif"
  "Clang assembler setup")
file(WRITE "${root_cmake}" "${content}")

set(dyncall_cmake "${patch_target}/dyncall/CMakeLists.txt")
file(READ "${dyncall_cmake}" content)
set(old_asm_language
  [=[  set_source_files_properties(${ASM_SRC} PROPERTIES LANGUAGE "C")
]=])
string(FIND "${content}" "${old_asm_language}" old_asm_position)
if(NOT old_asm_position EQUAL -1)
  string(REPLACE "${old_asm_language}" "" content "${content}")
elseif(content MATCHES "LANGUAGE \"C\"")
  message(FATAL_ERROR
    "Pinned dyncall layout changed while applying assembly source language")
endif()
file(WRITE "${dyncall_cmake}" "${content}")

set(dynload_test_cmake "${patch_target}/test/dynload_plain/CMakeLists.txt")
file(READ "${dynload_test_cmake}" content)
set(old_dynload_source
  [=[file(WRITE x.c "int dynload_plain_testfunc() { return 5; }")
add_library(x SHARED x.c)]=])
set(new_dynload_source
  [=[set(dynload_test_source "${CMAKE_CURRENT_BINARY_DIR}/x.c")
file(WRITE "${dynload_test_source}" "int dynload_plain_testfunc() { return 5; }")
add_library(x SHARED "${dynload_test_source}")]=])
string(FIND "${content}" "${old_dynload_source}" old_dynload_position)
string(FIND "${content}" "${new_dynload_source}" new_dynload_position)
if(old_dynload_position EQUAL -1 AND new_dynload_position EQUAL -1)
  message(FATAL_ERROR
    "Pinned dyncall layout changed while applying generated test source")
endif()
string(REPLACE "${old_dynload_source}" "${new_dynload_source}" content "${content}")
replace_required(
  content "exec_program(" "execute_process(COMMAND sh -c "
  "deprecated dynload test command")
replace_required(
  content
  " OUTPUT_VARIABLE DEF_C_DYLIB)"
  " OUTPUT_VARIABLE DEF_C_DYLIB OUTPUT_STRIP_TRAILING_WHITESPACE)"
  "dynload test command output handling")
file(WRITE "${dynload_test_cmake}" "${content}")
