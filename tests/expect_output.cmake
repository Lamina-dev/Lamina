if(NOT DEFINED LAMINA OR NOT DEFINED SOURCE OR NOT DEFINED EXPECTED)
    message(FATAL_ERROR "LAMINA, SOURCE, and EXPECTED are required")
endif()

execute_process(
    COMMAND "${LAMINA}" "${SOURCE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "expected successful execution, got ${result}\n${stdout}${stderr}")
endif()

string(STRIP "${stdout}" actual)
string(STRIP "${EXPECTED}" expected)
if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "unexpected output\nexpected: '${expected}'\nactual:   '${actual}'\nstderr:   '${stderr}'")
endif()
