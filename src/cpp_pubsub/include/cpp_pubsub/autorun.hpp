#ifndef __AUTORUN_HPP__
#define __AUTORUN_HPP__

#include <sycl/sycl.hpp>
#include <type_traits>

namespace fpga_tools {

namespace detail {
// Autorun implementation
template <bool run_forever, typename KernelID>
struct Autorun_impl {
  // Constructor with a kernel name
  template <typename DeviceSelector, typename KernelFunctor>
  Autorun_impl(DeviceSelector device_selector, KernelFunctor kernel) {
    // static asserts to ensure KernelFunctor is callable
    static_assert(std::is_invocable_r_v<void, KernelFunctor>,
                  "KernelFunctor must be callable with no arguments");

    // create the device queue
    sycl::queue q{device_selector};

    // submit the user's kernel
    if constexpr (run_forever) {
      if constexpr (std::is_same_v<KernelID, void>) {
        // AutorunForever, kernel name not given
        q.single_task([=] {
          while (1) {
            kernel();
          }
        });
      } else {
        // AutorunForever, kernel name given
        q.single_task<KernelID>([=] {
          while (1) {
            kernel();
          }
        });
      }
    } else {
      // run the kernel as-is, if the user wanted it to run forever they
      // will write their own explicit while-loop
      if constexpr (std::is_same_v<KernelID, void>) {
        // Autorun, kernel name not given
        q.single_task(kernel);
      } else {
        // Autorun, kernel name given
        q.single_task<KernelID>(kernel);
        // q.submit([&] (sycl::handler &h) {
        //   kernel;
        // });
      }
    }
  }
};
}  // namespace detail

// Autorun
template <typename KernelID = void>
using Autorun = detail::Autorun_impl<false, KernelID>;

// AutorunForever
template <typename KernelID = void>
using AutorunForever = detail::Autorun_impl<true, KernelID>;
}  // namespace fpga_tools

#endif /* __AUTORUN_HPP__ */