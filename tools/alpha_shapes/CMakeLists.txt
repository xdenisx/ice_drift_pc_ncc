cmake_minimum_required(VERSION 2.8)


project(alpha_shapes)
set(CMAKE_CXX_STANDARD 14)


#if( CGAL_CORE_FOUND )
#  message( INFO "Can not find cgal core" )
#  target_link_libraries (${PROJECT_NAME} ${CGAL_CORE_LIBRARY})
#elseif( SOME_CRITICAL_COND )
#  message( FATAL_ERROR "Can not find cgal core" )
#endif()

find_package(Boost 1.71 REQUIRED COMPONENTS
             program_options)

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} gmp Boost::program_options)
