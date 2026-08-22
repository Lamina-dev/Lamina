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
    # CI checks out submodules at a detached commit and may not create the
    # corresponding remote-tracking branch.  Merge the explicitly fetched
    # FETCH_HEAD instead, and avoid recursively fetching stale nested pointers.
    run_git("${directory}" fetch --no-recurse-submodules origin "${branch}")
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
