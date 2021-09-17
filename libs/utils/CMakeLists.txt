cmake_minimum_required(VERSION 3.1)

project(libutils)

set(HEADERS
        libutils/fast_random.h
        libutils/string_utils.h
        libutils/timer.h
        )

set(SOURCES
        libutils/string_utils.cpp
        )

set(CMAKE_CXX_STANDARD 11)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
