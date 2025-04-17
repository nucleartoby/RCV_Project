cmake_minimum_required(VERSION 3.10)
project(BlackScholesEngine)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(bs_engine
    main.cpp
    black_scholes.cpp
)