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

function(update_branch directory branch)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --is-shallow-repository
        WORKING_DIRECTORY "${directory}"
        OUTPUT_VARIABLE shallow
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE shallow_status
    )
    if(shallow_status EQUAL 0 AND shallow STREQUAL "true")
        run_git("${directory}" fetch --unshallow --no-recurse-submodules
                origin "${branch}")
    else()
        run_git("${directory}" fetch --no-recurse-submodules origin "${branch}")
    endif()
    run_git("${directory}" merge --ff-only FETCH_HEAD)
endfunction()

set(lmcas "${ROOT}/external/LMCAS")
set(lmmc "${lmcas}/LMMC")
set(lammp "${lmmc}/LAMMP")

run_git("${ROOT}" submodule sync --recursive)
if(NOT EXISTS "${lammp}/CMakeLists.txt")
    run_git("${ROOT}" submodule update --init --recursive -- external/LMCAS)
endif()

update_branch("${lmcas}" main)
update_branch("${lmmc}" main)
update_branch("${lammp}" main)
