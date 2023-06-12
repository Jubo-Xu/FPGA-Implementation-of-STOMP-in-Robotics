//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data shared between CPU and offload device.
// •	A device queue and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
//#include <sycl/ext/intel/fpga_extensions.hpp>
#include "Adder_Tree.h"
#include "CostFunction.h"
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include "Delta_Theta_Block.hpp"


constexpr int N_main = 8;
constexpr int LOG_N_main = 3;


Adder_Tree adder_tree;


// declare the kernel names globally to reduce name mangling
class ARProducerID;
class ARKernel_inID;
class ARKernel_outID;
class ARConsumerID;
class ARKernel_selfID;
class AR_4_1_kernel_ID; //mainly for testing

// declare the pipe names globally to reduce name mangling
class ARProducePipeID;
class ARConsumePipeID;
class ARKernel_self_inID;
class ARKernel_self_outID;
class PipeArray_1_ID;
class PipeArray_2_ID;

// pipes
using ARProducePipe = sycl::ext::intel::pipe<ARProducePipeID, float>;
using ARConsumePipe = sycl::ext::intel::pipe<ARConsumePipeID, float>;
using ARKernel_self_inPipe = sycl::ext::intel::pipe<ARKernel_self_inID, float, 1>;
using ARKernel_self_outPipe = sycl::ext::intel::pipe<ARKernel_self_outID, float>;
using ARKernel_PipeArray =  fpga_tools::PipeArray<PipeArray_1_ID, float, 1, 4>;
using ARKernel_PipeArray_2 =  fpga_tools::PipeArray<PipeArray_2_ID, float, 1, 4>;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

// Define the selector
#if FPGA_EMULATOR
  // Intel extension: FPGA emulator selector on systems without FPGA card.
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
  // Intel extension: FPGA simulator selector on systems without FPGA card.
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  // Intel extension: FPGA selector on systems with FPGA card.
  auto selector = sycl::ext::intel::fpga_selector_v;
#else
  // The default device selector will select the most performant device.
  auto selector = default_selector_v;
#endif
//std::array<std::array<float, N>, LOG_N> &datain;


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
////////////////// DEFINE THE AUTORUN BLOCK //////////////////////////

struct MyAutorun_in {
  void operator()() const {
    // notice that in this version, we explicitly add the while(1)-loop
    
    while (1) {
    //  [[intel::fpga_register]] float Coef[4];
    //   auto d = ARProducePipe::read();
    //   float out = 0.0; 
    //   #pragma unroll
    //   for(int i=0; i<4; i++){
    //     Coef[i] = d;
    //   }
    //   #pragma unroll
    //   for(int i=0; i<4; i++){
    //     //out = sycl::ext::intel::fpga_reg(out) + sycl::ext::intel::fpga_reg(Coef[i]);
    //     out = sycl::ext::intel::fpga_reg(out)  + Coef[i];
    //   }
    //   ARConsumePipe::write(out);
      //////////////////////////////////////////////////////////////////
      //////////// TEST FOR SERIAL TO PARALLEL SIMPLE //////////////////
      [[intel::fpga_register]] float Coef[4];
      float out = 0;
      for(int i=0; i<4; i++){
          Coef[i] = ARProducePipe::read();
      }
      size_t index = 0;
      fpga_tools::UnrolledLoop<4>([&index, &Coef](auto i) {
      ARKernel_PipeArray::PipeAt<i>::write(Coef[index++]);
      });
      // for(int j=0; j<2; j++){
      //    out += Coef[j];
      // }
      //ARConsumePipe::write(out);
      #pragma unroll
      for(int k=0; k<4; k++){
        Coef[k] = 0;
      }

    }
  }
}; 

struct MyAutorun_out_new {
  void operator()() const {
    // notice that in this version, we explicitly add the while(1)-loop
    
    while (1) {
      //////////// TEST FOR SERIAL TO PARALLEL SIMPLE //////////////////
      [[intel::fpga_register]] float Coef[4];

      size_t index = 0;
      fpga_tools::UnrolledLoop<4>([&index, &Coef](auto i) {
          Coef[index++] = ARKernel_PipeArray_2::PipeAt<i>::read();
      });
      for(int i=0; i<4; i++){
        ARConsumePipe::write(Coef[i]);
      }

    }
  }
}; 

template <typename Pipein, typename Pipeout>
struct Autoout{
  void operator()() const {
    // notice that in this version, we explicitly add the while(1)-loop
    
    while (1) {
      //[[intel::fpga_register]] float Coef[2];  
      Delta_Theta_Block::Array_Num_k Coef;
      Delta_Theta_Block::Array_Num_k OUT;
      size_t index = 0;
      fpga_tools::UnrolledLoop<4>([&index, &Coef](auto i) {
      Coef.Array[index++] = Pipein::template PipeAt<i>::read();
      });

      OUT = Delta_Theta_Block::Find_MAX_and_MIN(Coef);
      
      for(int i=0; i<4; i++){
        Pipeout::write(OUT.Array[i]);
      }
    }
  }
};

struct MyAutorun_out {
  void operator()() const {
    // notice that in this version, we explicitly add the while(1)-loop
    [[intel::fpga_register]] float Sphere_center[3] = {1.0f, 1.0f, 1.0f};
    [[intel::fpga_register]] float last[3][2] = {
      {0.0f, 0.0f},
      {0.0f, 0.0f},
      {0.0f, 0.0f}
    };
    Obstacle_CostFunction::Array_DIM_NB Cartesian_last;
    #pragma unroll
    for(int i=0; i<3; i++){
      #pragma unroll
      for(int j=0; j<2; j++){
        Cartesian_last.Array[i][j] = 0.0f;
      }
    }
    
    while (1) {
      [[intel::fpga_register]] float Coef[2];  
      [[intel::fpga_register]] float Cartesian_Pos[3][2]; 
      [[intel::fpga_register]] float Distance[2]; 
      [[intel::fpga_register]] float Velocity[2];
      Obstacle_CostFunction::Array_NB velocity_out;
      Obstacle_CostFunction::Array_DIM_NB Cartesian_in;
      float Sphere_radius = 1.0f;
      float out = 0;
      size_t index = 0;
      fpga_tools::UnrolledLoop<2>([&index, &Coef](auto i) {
      Coef[index++] = ARKernel_PipeArray::PipeAt<i>::read();
      });
      
      #pragma unroll
      for(int i=0; i<2; i++){
        Cartesian_Pos[0][i] = Coef[i];
        Cartesian_Pos[1][i] = Coef[i];
        Cartesian_Pos[2][i] = Coef[i];
      }

      #pragma unroll
      for(int i=0; i<3; i++){
        #pragma unroll
        for(int j=0; j<2; j++){
          Cartesian_in.Array[i][j] = Cartesian_Pos[i][j];
        }
      }
      //Test for distance
      // #pragma unroll
      // for(int i=0; i<2; i++){
      //   [[intel::fpga_register]] float square_term[3];
      //   #pragma unroll
      //   for(int j=0; j<3; j++){
      //       square_term[j] = (Cartesian_Pos[j][i]-Sphere_center[j])*(Cartesian_Pos[j][i]-Sphere_center[j]);
      //   }
      //   float square_add = square_term[0] + square_term[1] + square_term[2];
      //   Distance[i] = sycl::sqrt(square_add) - Sphere_radius;
      // }
      ///////////////////////
      /// Test for velocity
    //    #pragma unroll
    //   for(int i=0; i<2; i++){
    //     [[intel::fpga_register]] float Diff_1st[3];
    //     #pragma unroll
    //     for(int j=0; j<3; j++){
    //         Diff_1st[j] = Cartesian_Pos[j][i] - last[j][i];
    //         last[j][i] = Cartesian_Pos[j][i];
            
    //     }
    //     Velocity[i] = sycl::sqrt(Diff_1st[0]*Diff_1st[0] + Diff_1st[1]*Diff_1st[1] + Diff_1st[2]*Diff_1st[2]);
    //  }
    velocity_out = Obstacle_CostFunction::Calc_Velocity_AR(Cartesian_in, Cartesian_last);
      ///////////////////////
      // for(int i=0; i<5; i++){
      //   out += Coef[i];
      // }
      //ARConsumePipe::write(out);
      for(int i=0; i<2; i++){
        ARConsumePipe::write(velocity_out.Array[i]);
      }
    }
  }
};

// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
fpga_tools::Autorun<ARKernel_inID> ar_kernel_in{selector, MyAutorun_in{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, MyAutorun_out{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Autoout<ARKernel_PipeArray, ARConsumePipe>{}};
fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Delta_Theta_Block::Theta_Calc_Kernel2<ARKernel_PipeArray, ARKernel_PipeArray_2>{}};
fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, MyAutorun_out_new{}};

////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////

// Submit a kernel to read data from global memory and write to a pipe
//
template <typename KernelID, typename Pipe>
sycl::event SubmitProducerKernel(sycl::queue& q, sycl::buffer<float, 1>& in_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor in(in_buf, h, read_only);
    int size = in_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
        Pipe::write(in[i]);
      }
    });
  });
}

// Submit a kernel to read data from a pipe and write to global memory
//
template <typename KernelID, typename Pipe>
sycl::event SubmitConsumerKernel(sycl::queue& q, sycl::buffer<float, 1>& out_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out(out_buf, h, write_only, no_init);
    //int size = out_buf.size();
    int size = 12;
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
          out[i] = Pipe::read();
      }
    });
  });
}

// template <typename KernelID, typename Pipe1, typename Pipe2>
// sycl::event ExecuteKernel(sycl::queue& q) {
//   return q.single_task(MyAutorun_in<Pipe1, Pipe2>{});
// }



int main(){

std::cout<<"un peu d'espoir\n";


// #if FPGA_EMULATOR
//   // Intel extension: FPGA emulator selector on systems without FPGA card.
//   auto selector = sycl::ext::intel::fpga_emulator_selector_v;
// #elif FPGA_SIMULATOR
//   // Intel extension: FPGA simulator selector on systems without FPGA card.
//   auto selector = sycl::ext::intel::fpga_simulator_selector_v;
// #elif FPGA_HARDWARE
//   // Intel extension: FPGA selector on systems with FPGA card.
//   auto selector = sycl::ext::intel::fpga_selector_v;
// #else
//   // The default device selector will select the most performant device.
//   auto selector = default_selector_v;
// #endif
// Instantiate Class

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//////////////// TEST FOR ADDER TREE ////////////////////////////////////
// // Initialize vector
// std::array<std::array<float, N_main>, LOG_N_main> my_data = {{
//   {2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0},
//   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
//   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
// }};

// std::array<std::array<float, N_main>, LOG_N_main> my_data_out;

// //float my_data_out[LOG_N][N];
// int size_t = sizeof(my_data)/sizeof(my_data[0][0]);
// int size_r = sizeof(my_data[0])/sizeof(my_data[0][0]);
// int size_c = size_t/size_r;

// std::cout<<"row of array: "<< size_r<<"\n";
// std::cout<<"column of array: "<< size_c <<"\n";

// try{
//   sycl::queue q;
//   sycl::buffer<float, 2> my_buffer(sycl::range<2>{LOG_N_main, N_main});
//   sycl::buffer my_buffer_out(my_data_out);
//   sycl::buffer<float, 1> buffer_single{sycl::range{1}};

//   // Initialization for buffer input to adder tree in host
//   {sycl::host_accessor host_accessor_in(my_buffer);
//   for(int i=0; i<LOG_N_main; i++){
//     for(int j=0; j<N_main; j++){
//       host_accessor_in[i][j] = my_data[i][j];
//     }
//   }}
  
//   /*q.submit([&](sycl::handler &h) {
//       sycl::accessor my_accessor(my_buffer, h);
//       sycl::accessor my_accessor_out(my_buffer_out, h);
//       h.parallel_for(sycl::range<2>(LOG_N, N), [=](sycl::id<2> i) {
//         int r = i[0];
//         int c = i[1];
//         my_accessor_out[r][c] = my_accessor[r][c];
//       });
//   });*/
//   buffer_single = adder_tree.Adder_Tree_Execute(q, my_buffer, N_main, LOG_N_main);
//   //buffer_single = Adder_Tree_Execute_new(q, my_buffer);
  
//   q.wait();

//   std::cout<< "submit finished\n"<< std::endl;

//   //host out for adder_tree
//   sycl::host_accessor host_accessor_out(my_buffer_out);
//   sycl::host_accessor out_accessor(buffer_single);
//   std::cout<< "added value: \n"<<std::endl;
//   std::cout<< out_accessor[0] << "\n";

//   }catch (sycl::exception const &e) {
//     std::cout << "An exception is caught for vector add.\n";
//     std::terminate();
// }  

  

// for(int i=0; i<LOG_N_main; i++){
//     for(int j=0; j<N_main; j++){
//         std::cout<< my_data[i][j] << " ";
//     }
    
//   }
// std::cout<< "\n";
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////// TEST FOR COST FUNCTION //////////////////////////////////

// std::vector<float> VEC = {1.0, 2.0, 0.0, 0.0, 0.0};
// std::vector<float> OUT(5, 0);
// sycl::queue q;
// sycl::buffer<float> my_buffer{sycl::range{5}};
// {
//     sycl::host_accessor host_accessor{my_buffer};
//     for(int i=0; i<5; i++){
//       host_accessor[i] = VEC[i];
//     }
//     std::cout<< "Buffer initial: \n";
//     for(int i=0; i<5; i++){
//       std::cout<< host_accessor[i];
//     }
//     std::cout<<"\n";
// }

// {
//   q.submit([&] (sycl::handler &h) {
//     sycl::accessor a{my_buffer, h};
//     h.single_task([=] {
//       #pragma unroll
//       for(int i=0; i<4; i++){
//         a[4-i] = sycl::ext::intel::fpga_reg(a[4-i-1]);
//         //a[i] = 3.0;
//       }
//     });
//   });
//   std::cout<<"submit finished" <<std::endl;
//   sycl::host_accessor out_accessor{my_buffer};
//   std::cout<< "Buffer after shifting: \n";
//   for(int i=0; i<5; i++){
//     std::cout<< out_accessor[i];
//   }
//   std::cout<< "\n";
// }


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// TEST FOR AUTORUN //////////////////////////////////

// Number of total numbers of input
int count = 12;
//bool passed = true;

std::vector<float> in_data(count), out_data(count);
// Clear the output buffer
std::fill(out_data.begin(), out_data.end(), -1);
// Initialize the input buffer
// for(int i=0; i<count; i++){
//   in_data[i] = i+1;
// }
//in_data = {1.0f, 2.0f, 2.0f, 3.0f, 2.5f, 4.0f, 5.5f, 2.0f, 8.0f, 3.5f};
in_data = {4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 3.0f};
std::cout<<"input data: \n";
for(int i=0; i<count; i++){
  std::cout<<in_data[i];
}
std::cout<<"\n";

 try {
    // create the queue
    sycl::queue q(selector, exception_handler);

    sycl::device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // stream data through the Autorun kernel
    std::cout << "Running the Autorun kernel test\n";
    {
      // Create input and output buffers
      sycl::buffer in_buf(in_data);
      sycl::buffer out_buf(out_data);
      sycl::buffer<float, 1> mid_buf{sycl::range{5}};
      SubmitProducerKernel<ARProducerID, ARProducePipe>(q, in_buf);
      // q.submit([&] (sycl::handler& h) {
      //    //sycl::accessor a{mid_buf, h};
      //    h.single_task<ARKernel_selfID>([=] {
      //      /*float out =0.0;
      //      #pragma unroll
      //      for(int i=0; i<5; i++){
      //        a[i] = ARKernel_self_inPipe::read();
      //      }
      //      for(int i=0; i<5; i++){
      //        out += a[i];
      //      }*/
      //      auto out = ARKernel_self_inPipe::read();
      //      //ARKernel_self_outPipe::write(out);
      //      ARConsumePipe::write(out);
      //    });

      // });
      SubmitConsumerKernel<ARConsumerID, ARConsumePipe>(q, out_buf);
    }

    std::cout<<"submit finished\n";

    // validate the results
    // operator== for a vector checks sizes, then checks per-element
    //passed &= (out_data == in_data);
    
    // To show the output buffer
    std::cout<<"output is: \n";
    for(int i=0; i<count; i++){
      std::cout<<out_data[i]<<"\n";
    }
    std::cout<<"\n";

  } catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  /*if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }*/
  std::cout<<"finished \n";

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


return 0;

}
