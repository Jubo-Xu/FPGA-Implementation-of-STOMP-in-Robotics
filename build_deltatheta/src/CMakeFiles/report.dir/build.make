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

# Utility rule file for report.

# Include any custom commands dependencies for this target.
include src/CMakeFiles/report.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/report.dir/progress.make

src/CMakeFiles/report: do-test-deltatheta_report.a

report: src/CMakeFiles/report
report: src/CMakeFiles/report.dir/build.make
.PHONY : report

# Rule to build all files generated by this target.
src/CMakeFiles/report.dir/build: report
.PHONY : src/CMakeFiles/report.dir/build

src/CMakeFiles/report.dir/clean:
	cd /home/bobxu/sycl_stomp/build_deltatheta/src && $(CMAKE_COMMAND) -P CMakeFiles/report.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/report.dir/clean

src/CMakeFiles/report.dir/depend:
	cd /home/bobxu/sycl_stomp/build_deltatheta && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bobxu/sycl_stomp /home/bobxu/sycl_stomp/src /home/bobxu/sycl_stomp/build_deltatheta /home/bobxu/sycl_stomp/build_deltatheta/src /home/bobxu/sycl_stomp/build_deltatheta/src/CMakeFiles/report.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/report.dir/depend

