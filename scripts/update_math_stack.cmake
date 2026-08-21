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

set(lmcas "${ROOT}/external/LMCAS")
set(lmmc "${lmcas}/LMMC")
set(lammp "${lmmc}/LAMMP")

run_git("${ROOT}" submodule sync --recursive)
if(NOT EXISTS "${lammp}/CMakeLists.txt")
    run_git("${ROOT}" submodule update --init --recursive -- external/LMCAS)
endif()

run_git("${lmcas}" fetch origin lmcas-2-current)
run_git("${lmcas}" merge --ff-only origin/lmcas-2-current)
run_git("${lmmc}" fetch origin main)
run_git("${lmmc}" merge --ff-only origin/main)
run_git("${lammp}" fetch origin main)
run_git("${lammp}" merge --ff-only origin/main)
