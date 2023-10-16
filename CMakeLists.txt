cmake_minimum_required(VERSION 3.5)

project(MathExpression
  VERSION 0.1
  DESCRIPTION "c++ MathExpression"
  LANGUAGES CXX
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# For QT.
#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)
#set(CMAKE_AUTOUIC ON)

set(TARGET ${CMAKE_PROJECT_NAME})
set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIB_DIR ${ROOT_DIR}/lib)
set(MAIN_DIR ${ROOT_DIR})

set(INCLUDE_DIR ${ROOT_DIR} ${ROOT_DIR}/include)
set(INCLUDE_PATH ${INCLUDE_DIR} ${MISC_DIR})
set(SOURCE_DIR ${ROOT_DIR} ${ROOT_DIR}/src)
set(RESOURCE_DIR ${ROOT_DIR}/resources)

# Quoting a list variable displays the values with dilimeters(';').
#message("Include Dirs: ${INCLUDE_DIR}")

set(SOURCES
  ${MAIN_DIR}/main.cpp
)

set(HEADERS
  ${INCLUDE_DIR}/MathExpression.h
)

set(RESOURCES
#  ${RESOURCE_DIR}/resource.qrc
)

set(PROJECT_SOURCES ${SOURCES} ${HEADERS})

add_executable(${TARGET} ${PROJECT_SOURCES} ${RESOURCES})
target_include_directories(${TARGET} PRIVATE ${INCLUDE_PATH})
target_link_libraries(${TARGET} PRIVATE ${LIBRARIES})
