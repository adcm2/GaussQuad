cmake_minimum_required (VERSION 3.5)


project(GSHTrans
        VERSION 1.0
        LANGUAGES CXX
        )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
