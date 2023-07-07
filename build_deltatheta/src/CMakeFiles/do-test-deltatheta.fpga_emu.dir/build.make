# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bobxu/sycl_stomp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bobxu/sycl_stomp/build_deltatheta

# Include any dependencies generated for this target.
include src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/flags.make

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/flags.make
src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o: ../src/test_delta_theta_block.cpp
src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bobxu/sycl_stomp/build_deltatheta/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o -MF CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o.d -o CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o -c /home/bobxu/sycl_stomp/src/test_delta_theta_block.cpp

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.i"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bobxu/sycl_stomp/src/test_delta_theta_block.cpp > CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.i

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.s"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bobxu/sycl_stomp/src/test_delta_theta_block.cpp -o CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.s

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/flags.make
src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o: ../src/Adder_Tree.cpp
src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bobxu/sycl_stomp/build_deltatheta/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o -MF CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o.d -o CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o -c /home/bobxu/sycl_stomp/src/Adder_Tree.cpp

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.i"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bobxu/sycl_stomp/src/Adder_Tree.cpp > CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.i

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.s"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bobxu/sycl_stomp/src/Adder_Tree.cpp -o CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.s

# Object files for target do-test-deltatheta.fpga_emu
do__test__deltatheta_fpga_emu_OBJECTS = \
"CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o" \
"CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o"

# External object files for target do-test-deltatheta.fpga_emu
do__test__deltatheta_fpga_emu_EXTERNAL_OBJECTS =

do-test-deltatheta.fpga_emu: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/test_delta_theta_block.cpp.o
do-test-deltatheta.fpga_emu: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/Adder_Tree.cpp.o
do-test-deltatheta.fpga_emu: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/build.make
do-test-deltatheta.fpga_emu: src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bobxu/sycl_stomp/build_deltatheta/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../do-test-deltatheta.fpga_emu"
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/do-test-deltatheta.fpga_emu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/build: do-test-deltatheta.fpga_emu
.PHONY : src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/build

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/clean:
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && $(CMAKE_COMMAND) -P CMakeFiles/do-test-deltatheta.fpga_emu.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/clean

src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/depend:
	cd /home/bobxu/sycl_stomp/build_deltatheta && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bobxu/sycl_stomp /home/bobxu/sycl_stomp/src /home/bobxu/sycl_stomp/build_deltatheta /home/bobxu/sycl_stomp/build_deltatheta/src /home/bobxu/sycl_stomp/build_deltatheta/src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/do-test-deltatheta.fpga_emu.dir/depend

