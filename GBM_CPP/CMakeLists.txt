cmake_minimum_required(VERSION 3.10)
project(MyLightGBMProject)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Assuming LightGBM is installed in /usr/local, set the include directory
include_directories(/usr/local/include/LightGBM)

# Specify the path to the LightGBM library
link_directories(/usr/local/lib)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Define the executable
add_executable(my_executable main.cpp)

# Link the LightGBM library
target_link_libraries(my_executable lib_lightgbm.so)
target_link_libraries(my_executable Eigen3::Eigen)
