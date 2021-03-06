cmake_minimum_required(VERSION 3.6)

project(Solver)

option(USE_PPROF OFF)
option(USE_ASAN OFF)

add_definitions(-DHOME)
if (USE_PPROF)
  add_definitions(-DUSE_PPROF)
endif()

add_compile_options(
  "-std=c++1z"
  "-Wall"
)

if (USE_ASAN)
  add_compile_options(
    "-fno-omit-frame-pointer"
    "-fsanitize=address"
  )
endif()

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

if (USE_PPROF)
  target_link_libraries(${PROJECT_NAME} "-lprofiler")
endif()

if (USE_ASAN)
  target_link_libraries(${PROJECT_NAME} "-fsanitize=address")
endif()
