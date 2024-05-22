cmake_minimum_required(VERSION 3.21)

project(burst LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

set(isClang $<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>)
set(isGcc $<C_COMPILER_ID:GNU>)
set(isDebug $<CONFIG:Debug>)

# If you want to add, remove, or disable any warnings, you can do that here.
# Remove a warning by adding -Wno-xxx where xxx is the name of the warning. For
# example: -Wno-cast-align.
set(clangWarnings
    -Wno-declaration-after-statement
    -Wno-padded
    -Wno-unknown-pragmas
    -Wno-c++98-compat-pedantic
    -Wno-c++11-compat-pedantic
    -Weverything)

set(gccWarnings
    -Wno-padded
    -Wno-unknown-pragmas
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wcast-align
    -Wunused
    -Wconversion
    -Wsign-conversion
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2)

set(sanitizers -fsanitize=address,undefined)

#include(CTest)

# Any source files added to the project have to be added to this list.
add_executable(server
  server.cpp
)
add_executable(client
  client.cpp
)

#target_include_directories(server PRIVATE ./)
#target_include_directories(client PRIVATE ./)

target_compile_features(server PRIVATE c_std_${CMAKE_C_STANDARD})
target_compile_features(client PRIVATE c_std_${CMAKE_C_STANDARD})

# Use compiler specific extensions (strdup is used).
#set_target_properties(mips_pipeline PROPERTIES C_EXTENSIONS ON)

target_compile_options(server PRIVATE
    $<${isClang}:${clangWarnings}>
    $<${isGcc}:${gccWarnings}>
    $<${isDebug}:${debugOptions}>
)
target_compile_options(client PRIVATE
    $<${isClang}:${clangWarnings}>
    $<${isGcc}:${gccWarnings}>
    $<${isDebug}:${debugOptions}>
)
