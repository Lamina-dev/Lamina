if(NOT DEFINED LAMINA OR NOT DEFINED SOURCE OR NOT DEFINED DIAGNOSTIC)
    message(FATAL_ERROR "LAMINA, SOURCE, and DIAGNOSTIC are required")
endif()

execute_process(
    COMMAND "${LAMINA}" "${SOURCE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

set(output "${stdout}${stderr}")
if(result EQUAL 0)
    message(FATAL_ERROR "expected compilation failure, but command succeeded\n${output}")
endif()
if(NOT output MATCHES "${DIAGNOSTIC}")
    message(FATAL_ERROR "expected diagnostic '${DIAGNOSTIC}'\n${output}")
endif()
