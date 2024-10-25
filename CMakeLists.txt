cmake_minimum_required(VERSION 3.17)

project(focus-stack)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/tools/opencv/mingw_build/install")

find_package(OpenCV REQUIRED)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(PROJECT_LIB ${PROJECT_NAME}_lib)
set(PROJECT_TEST ${PROJECT_NAME}_test)

option(BUILD_TESTS "build tests" OFF) # set to ON if you want to build tests

file(GLOB_RECURSE PROJECT_LIB_SOURCES LIST_DIRECTORIES false CONFIGURE_DEPENDS
    ${SRC_DIR}/*.cc ${SRC_DIR}/*.hh ${SRC_DIR}/*cl
)
list(FILTER PROJECT_LIB_SOURCES EXCLUDE REGEX .*_tests.cc|.*main.cc)

add_library(${PROJECT_LIB}
    SHARED # set to SHARED if you want dll, STATIC if you want lib
    ${PROJECT_LIB_SOURCES}
)
target_link_libraries(${PROJECT_LIB}
    PUBLIC ${OpenCV_LIBS}
)
if(MSVC)
    target_compile_definitions(${PROJECT_LIB} PRIVATE _USE_MATH_DEFINES)
endif()

#add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cc)
#target_link_libraries(${PROJECT_NAME}
#    PRIVATE ${PROJECT_LIB}
#)

if (BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    file(GLOB_RECURSE PROJECT_TEST_SOURCES LIST_DIRECTORIES false CONFIGURE_DEPENDS
        ${SRC_DIR}/*_tests.cc
    )
    add_executable(${PROJECT_TEST} ${PROJECT_TEST_SOURCES} ${SRC_DIR}/gtest_main.cc)
    target_link_libraries(${PROJECT_TEST}
        PRIVATE ${PROJECT_LIB}
        PRIVATE GTest::gtest_main
    )
    add_test(NAME ${PROJECT_TEST} COMMAND ${PROJECT_TEST})
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_LIB})
target_include_directories(${PROJECT_LIB}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # 包含生成的导出头文件
        $<INSTALL_INTERFACE:include>
)

# 安装库文件
install(TARGETS ${PROJECT_LIB}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# 安装头文件
file(GLOB_RECURSE HEADER_FILES "${SRC_DIR}/*.h" "${SRC_DIR}/*.hpp" "${SRC_DIR}/*.hh")
install(FILES ${HEADER_FILES} DESTINATION include/focusstack)

# 安装生成的导出头文件
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB}_export.h" DESTINATION include/focusstack)

# 导出目标供 find_package 使用
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Config.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})
