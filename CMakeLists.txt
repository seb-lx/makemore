cmake_minimum_required(VERSION 3.28)
project(makemore CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



file(GLOB_RECURSE SRC_FILES "src/*.cpp")

add_executable(makemore main.cpp ${SRC_FILES})

target_include_directories(makemore PRIVATE include)


# Compile flags
target_compile_options(makemore PRIVATE
    # common compile options
    -pedantic-errors
)
target_compile_options(makemore PRIVATE
    # debug specific compile options
    $<$<CONFIG:Debug>:
        -ggdb -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror
    >
    # release specific compile options
    $<$<CONFIG:Release>:
        #-O2
        -O3 -march=native
    >
)
# preprocessor definition
target_compile_definitions(makemore PRIVATE $<$<CONFIG:Release>:NDEBUG>)

# copy data folder to build
add_custom_command(TARGET makemore POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    $<TARGET_FILE_DIR:makemore>/data
    COMMENT "Copying data folder to build..."
)


find_package(OpenMP REQUIRED)
target_link_libraries(makemore OpenMP::OpenMP_CXX)


add_executable(tests tests/test_tensor.cpp)
target_include_directories(tests PRIVATE include)



