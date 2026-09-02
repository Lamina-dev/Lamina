if(NOT DEFINED LMMP_SRC)
  message(FATAL_ERROR "LMMP_SRC is required")
endif()

set(div_source "${LMMP_SRC}/src/lmmp/lmmpn/generic/div.c")
file(READ "${div_source}" content)
string(REPLACE
  "    // q: assigned for macro reuse, unused in this logic (known warning)\n    mp_limb_t t = numa[na - 2], q = 0, r = 0;"
  "    mp_limb_t t = numa[na - 2], q, r;"
  content "${content}")
string(REPLACE
  "        _udiv_qrnnd_preinv(q, r, ah, al, x, inv);\n        return r >> shift;"
  "        _udiv_qrnnd_preinv(q, r, ah, al, x, inv);\n        (void)q;\n        return r >> shift;"
  content "${content}")
string(REPLACE
  "        _udiv_qrnnd_preinv(q, r, ah, al, x, inv);\n        return r;"
  "        _udiv_qrnnd_preinv(q, r, ah, al, x, inv);\n        (void)q;\n        return r;"
  content "${content}")
file(WRITE "${div_source}" "${content}")

set(lmmp_cmake "${LMMP_SRC}/CMakeLists.txt")
file(READ "${lmmp_cmake}" content)
string(REPLACE
  "\${CMAKE_SOURCE_DIR}/src/lmmp/lmmpn/asm/"
  "\${CMAKE_CURRENT_SOURCE_DIR}/src/lmmp/lmmpn/asm/"
  content "${content}")
file(WRITE "${lmmp_cmake}" "${content}")

get_filename_component(lmmc_src "${LMMP_SRC}/.." ABSOLUTE)
set(lmmc_cmake "${lmmc_src}/CMakeLists.txt")
file(READ "${lmmc_cmake}" content)
string(REPLACE
  "set(LMMC_LMMP_ASM \"GENERIC\" CACHE STRING \"LMMP assembly mode\")"
  "set(LMMC_LMMP_ASM \"AUTO\" CACHE STRING \"LMMP assembly mode\")"
  content "${content}")
file(WRITE "${lmmc_cmake}" "${content}")

get_filename_component(lmcas_src "${lmmc_src}/.." ABSOLUTE)
set(lmcas_cmake "${lmcas_src}/CMakeLists.txt")
file(READ "${lmcas_cmake}" content)
string(REPLACE
  "set(LMCAS_LMMP_ASM \"GENERIC\" CACHE STRING \"LMMP assembly mode\")"
  "set(LMCAS_LMMP_ASM \"AUTO\" CACHE STRING \"LMMP assembly mode\")"
  content "${content}")
file(WRITE "${lmcas_cmake}" "${content}")
