set(patch_target "${DYNCALL_SRC}")
set(old "if(MSVC)")
set(new "if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL \"Clang\")")
foreach(rel_file CMakeLists.txt dyncall/CMakeLists.txt dyncallback/CMakeLists.txt)
  set(f "${patch_target}/${rel_file}")
  file(READ "${f}" content)
  string(REPLACE "${old}" "${new}" content "${content}")
  file(WRITE "${f}" "${content}")
endforeach()

set(root_cmake "${patch_target}/CMakeLists.txt")
file(READ "${root_cmake}" content)
string(REPLACE
  "cmake_minimum_required (VERSION 2.6)"
  "cmake_minimum_required(VERSION 3.10...3.31)"
  content "${content}")
string(REGEX REPLACE
  "COMMENT \"Assembling \\$\\{ASM_FILE\\} ---> \\\\\"\\$\\{CMAKE_ASM_COMPILER\\}\\\\\" \\$\\{ASM_INCLUDE_DIRECTORIES\\} -o \\$\\{OBJ_FILE\\} \\$\\{ASM_FILE\\}\"\\+[\t ]*COMMENT "
  "COMMENT "
  content "${content}")
string(REPLACE
  "elseif(CMAKE_COMPILER_IS_CLANG)\nelseif"
  "elseif(CMAKE_COMPILER_IS_CLANG)\n  enable_language(ASM)\nelseif"
  content "${content}")
file(WRITE "${root_cmake}" "${content}")

set(dyncall_cmake "${patch_target}/dyncall/CMakeLists.txt")
file(READ "${dyncall_cmake}" content)
string(REPLACE
  "  set_source_files_properties(\${ASM_SRC} PROPERTIES LANGUAGE \"C\")\n"
  ""
  content "${content}")
file(WRITE "${dyncall_cmake}" "${content}")

set(dynload_test_cmake "${patch_target}/test/dynload_plain/CMakeLists.txt")
file(READ "${dynload_test_cmake}" content)
string(REPLACE
  "file(WRITE x.c \"int dynload_plain_testfunc() { return 5; }\")\nadd_library(x SHARED x.c)"
  "set(dynload_test_source \"\${CMAKE_CURRENT_BINARY_DIR}/x.c\")\nfile(WRITE \"\${dynload_test_source}\" \"int dynload_plain_testfunc() { return 5; }\")\nadd_library(x SHARED \"\${dynload_test_source}\")"
  content "${content}")
file(WRITE "${dynload_test_cmake}" "${content}")
