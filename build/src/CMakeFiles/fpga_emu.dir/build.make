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
CMAKE_BINARY_DIR = /home/bobxu/sycl_stomp/build

# Utility rule file for fpga_emu.

# Include any custom commands dependencies for this target.
include src/CMakeFiles/fpga_emu.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/fpga_emu.dir/progress.make

src/CMakeFiles/fpga_emu: stomp.fpga_emu

fpga_emu: src/CMakeFiles/fpga_emu
fpga_emu: src/CMakeFiles/fpga_emu.dir/build.make
.PHONY : fpga_emu

# Rule to build all files generated by this target.
src/CMakeFiles/fpga_emu.dir/build: fpga_emu
.PHONY : src/CMakeFiles/fpga_emu.dir/build

src/CMakeFiles/fpga_emu.dir/clean:
	cd /home/bobxu/sycl_stomp/build/src && $(CMAKE_COMMAND) -P CMakeFiles/fpga_emu.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/fpga_emu.dir/clean

src/CMakeFiles/fpga_emu.dir/depend:
	cd /home/bobxu/sycl_stomp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bobxu/sycl_stomp /home/bobxu/sycl_stomp/src /home/bobxu/sycl_stomp/build /home/bobxu/sycl_stomp/build/src /home/bobxu/sycl_stomp/build/src/CMakeFiles/fpga_emu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/fpga_emu.dir/depend

