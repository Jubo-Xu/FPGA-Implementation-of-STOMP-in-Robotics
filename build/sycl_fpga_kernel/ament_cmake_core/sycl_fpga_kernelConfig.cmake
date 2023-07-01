# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_sycl_fpga_kernel_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED sycl_fpga_kernel_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(sycl_fpga_kernel_FOUND FALSE)
  elseif(NOT sycl_fpga_kernel_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(sycl_fpga_kernel_FOUND FALSE)
  endif()
  return()
endif()
set(_sycl_fpga_kernel_CONFIG_INCLUDED TRUE)

# output package information
if(NOT sycl_fpga_kernel_FIND_QUIETLY)
  message(STATUS "Found sycl_fpga_kernel: 0.0.0 (${sycl_fpga_kernel_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'sycl_fpga_kernel' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${sycl_fpga_kernel_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(sycl_fpga_kernel_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${sycl_fpga_kernel_DIR}/${_extra}")
endforeach()
