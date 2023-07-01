if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required(VERSION 3.8)
project(sycl_fpga_kernel)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes /fp:precise")
else()
    set(AC_TYPES_FLAG "-qactypes -fp-model=precise")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tutorial_interfaces REQUIRED)
#find_package(oneDPL REQUIRED)
#set(IntelDPCPP_DIR "/opt/intel/oneapi/compiler/2023.1.0/linux/IntelDPCPP")
# set(IntelSYCL_DIR "/opt/intel/oneapi/compiler/2023.1.0/linux/IntelSYCL")

# #find_package(IntelDPCPP REQUIRED)
# find_package(IntelSYCL REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set(SOURCE_FILE src/subscriber_member_function.cpp)
set(TARGET_NAME fpga_kernel)
#set(AC_TYPES_FLAG "-qactypes -fp-model=precise")

# FPGA device selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex")
    message(STATUS "FPGA_DEVICE was not specified.\
                \nConfiguring the design to target the default FPGA family (Intel Agilex®). \
                \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA device ${FPGA_DEVICE}")
endif()

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)

set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Wall ${WIN_FLAG} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga")

add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs tutorial_interfaces)
# add_executable(listener src/subscriber_member_function.cpp)
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
# ament_target_dependencies(listener rclcpp std_msgs tutorial_interfaces)
ament_target_dependencies(${EMULATOR_TARGET} rclcpp std_msgs tutorial_interfaces)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# install(TARGETS
#   talker
#   DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
install(TARGETS
  talker
  #listener
  ${EMULATOR_TARGET}
  DESTINATION lib/${PROJECT_NAME})
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
