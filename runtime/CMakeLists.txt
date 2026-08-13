add_library(lamina_runtime OBJECT
    binary.cpp
    vm.cpp
    object/fraction.cpp
        object/StringObj.cpp
    object/object.cpp
    object/value.cpp
    object/code_module.cpp
    gc.cpp
        object/array.cpp
        object/array.hpp
        object/tuple.cpp
        object/tuple.hpp
)



#find_package(GTest)
#
#add_executable(test_main tests/test_main.cpp)
#target_link_libraries(test_main PRIVATE gtest gtest_main lmx_objsys)
#enable_testing()
#add_test(NAME TestMain COMMAND test_main)
#
#add_executable(test_string tests/test_string.cpp)
#target_link_libraries(test_string PRIVATE gtest gtest_main lmx_objsys)
#enable_testing()
#add_test(NAME TestString COMMAND test_string)
