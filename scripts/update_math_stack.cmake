cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED ROOT)
    message(FATAL_ERROR "ROOT is required")
endif()

find_package(Git REQUIRED)

function(run_git directory)
    foreach(attempt RANGE 1 3)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" ${ARGN}
            WORKING_DIRECTORY "${directory}"
            RESULT_VARIABLE status
            COMMAND_ECHO STDOUT
        )
        if(status EQUAL 0)
            return()
        endif()
        if(attempt LESS 3)
            execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 2)
        endif()
    endforeach()
    message(FATAL_ERROR "git command failed in ${directory}")
endfunction()

run_git("${ROOT}" submodule sync --recursive)
run_git("${ROOT}" submodule update --init --recursive --checkout)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule status --recursive
    WORKING_DIRECTORY "${ROOT}"
    RESULT_VARIABLE status_result
    OUTPUT_VARIABLE status_output
    ERROR_VARIABLE status_error
)
if(NOT status_result EQUAL 0)
    message(FATAL_ERROR "failed to inspect pinned submodules: ${status_error}")
endif()
if(status_output MATCHES "(^|\n)[+U-]")
    message(FATAL_ERROR
        "submodule checkout differs from a recorded gitlink:\n${status_output}")
endif()
