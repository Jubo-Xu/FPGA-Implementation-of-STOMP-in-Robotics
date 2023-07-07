
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
#include "CostSmooth.hpp"
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>


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
class PipeArray_p_ID;
class PipeArray_epsilon_ID;
class PipeArray_theta_out_ID;

//declare for kernal1
class ARProduceKernel_kernal1_ID;
class ARConsumeKernel_kernal1_ID;
class PipeArray_in_kernal1_ID;
class PipeArray_out_kernal1_ID;

//declare for kernal2
class ARProduceKernel_kernal2_ID;
class ARConsumeKernel_kernal2_ID;
class PipeArray_in_kernal2_ID;
class PipeArray_out_kernal2_ID;

//declare for kernal3
class ARProduceKernel_kernal3_ID;
class ARConsumeKernel_kernal3_ID;
class PipeArray_in1_kernal3_ID;
class PipeArray_out1_kernal3_ID;
class PipeArray_in2_kernal3_ID;
class PipeArray_out2_kernal3_ID;
//declare for the whole
class PipeArray_Delta_theta_kernel1_to_2_ID;
class PipeArray_Delta_theta_kernel2_to_3_ID;
class PipeArray_Delta_theta_kernel_out_2_MMUL_ID;
class PipeArray_Delta_theta_kernel_out_2_smooth_ID;
class ARDelta_theta_kernel1_ID;
class ARDelta_theta_kernel2_ID;
class ARDelta_theta_kernel3_ID;

// pipes
using ARProducePipe = sycl::ext::intel::pipe<ARProducePipeID, float>;
using ARConsumePipe = sycl::ext::intel::pipe<ARConsumePipeID, float>;
using ARKernel_self_inPipe = sycl::ext::intel::pipe<ARKernel_self_inID, float, 1>;
using ARKernel_self_outPipe = sycl::ext::intel::pipe<ARKernel_self_outID, float>;
using ARKernel_PipeArray =  fpga_tools::PipeArray<PipeArray_1_ID, float, 1, 4>;
using ARKernel_PipeArray_2 =  fpga_tools::PipeArray<PipeArray_2_ID, float, 1, 4>;
using pipe_array_p = fpga_tools::PipeArray<PipeArray_p_ID, float, 1, 4>;
using pipe_array_epsilon = fpga_tools::PipeArray<PipeArray_epsilon_ID, float, 1, 12>;
using pipe_array_theta_out = fpga_tools::PipeArray<PipeArray_theta_out_ID, float, 1, 3>;

//Pipes for kernal1
using ARKernel1_in_PipeArray =  fpga_tools::PipeArray<PipeArray_in_kernal1_ID, float, 1, 4>;
using ARKernel1_out_PipeArray =  fpga_tools::PipeArray<PipeArray_out_kernal1_ID, float, 1, 4>;
//Pipes for kernal 2
using ARKernel2_in_PipeArray =  fpga_tools::PipeArray<PipeArray_in_kernal2_ID, float, 1, 4>;
using ARKernel2_out_PipeArray =  fpga_tools::PipeArray<PipeArray_out_kernal2_ID, float, 1, 4>;
//Pipes for kernal 3
using ARKernel3_in1_PipeArray =  fpga_tools::PipeArray<PipeArray_in1_kernal3_ID, float, 1, 4>;
using ARKernel3_out1_PipeArray =  fpga_tools::PipeArray<PipeArray_out1_kernal3_ID, float, 1, 3>;
using ARKernel3_in2_PipeArray =  fpga_tools::PipeArray<PipeArray_in2_kernal3_ID, float, 1, 12>;
using ARKernel3_out2_PipeArray =  fpga_tools::PipeArray<PipeArray_out2_kernal3_ID, float, 1, 3>;
//Pipes for the whole
using pipearray_DT_1_2 = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel1_to_2_ID, float, 1, 4>;
using pipearray_DT_2_3 = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel2_to_3_ID, float, 1, 4>;
using pipearray_DT_out_2_MMUL = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel_out_2_MMUL_ID, float, 1, 3>;
using pipearray_DT_out_2_smooth = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel_out_2_smooth_ID, float, 1, 3>;


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
//fpga_tools::Autorun<ARKernel_inID> ar_kernel_in{selector, MyAutorun_in{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, MyAutorun_out{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Autoout<ARKernel_PipeArray, ARConsumePipe>{}};
//fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, MyAutorun_out_new{}};
//fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipe_array_p, pipe_array_epsilon, pipe_array_theta_out>{}};

//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Delta_Theta_Block::Theta_Calc_Kernel1<ARKernel1_in_PipeArray, ARKernel1_out_PipeArray>{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Delta_Theta_Block::Theta_Calc_Kernel2<ARKernel2_in_PipeArray, ARKernel2_out_PipeArray>{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Delta_Theta_Block::Theta_Calc_Kernel3<ARKernel3_in1_PipeArray, ARKernel3_in2_PipeArray,ARKernel3_out1_PipeArray,ARKernel3_out2_PipeArray>{}};
// Define Delta theta kernel 1
fpga_tools::Autorun<ARDelta_theta_kernel1_ID> ar_kernel_DT1(selector, Delta_Theta_Block::Theta_Calc_Kernel1<ARKernel1_in_PipeArray, pipearray_DT_1_2>{});
// Define Delta theta kernel 2
fpga_tools::Autorun<ARDelta_theta_kernel2_ID> ar_kernel_DT2(selector, Delta_Theta_Block::Theta_Calc_Kernel2<pipearray_DT_1_2, pipearray_DT_2_3>{});
// Define Delta theta kernel 3
fpga_tools::Autorun<ARDelta_theta_kernel3_ID> ar_kernel_DT3(selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipearray_DT_2_3, ARKernel3_in2_PipeArray, ARKernel3_out1_PipeArray,ARKernel3_out2_PipeArray>{});

/////////////////////////////////////////////////////////////////////// 
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

/////////////////////Test case for kernal 1/////////////////////////////
// Submit
template <typename KernelID, typename Pipe_in>
sycl::event SubmitProducer_test_kernal1(sycl::queue& q, sycl::buffer<float, 1>& in_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor in(in_buf, h, read_only);
    //int size = in_buf.size();
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //   Pipe::write(in[i]);
      // }
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<4>([&index_1, &in, &i](auto j) {
              Pipe_in::template PipeAt<j>::write(in[i*4 + index_1]);
              index_1++;
        });
      }
    });
  });
}

// consumer
template <typename KernelID, typename Pipe_out>
sycl::event SubmitConsumer_test_kernal1(sycl::queue& q, sycl::buffer<float, 1>& out_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out(out_buf, h, write_only, no_init);
    //int size = out_buf.size();
    //int size = 12;
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //     out[i] = Pipe::read();
      // }

        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &out](auto i) {
          out[index++] = Pipe_out::template PipeAt<i>::read();
        });
    });
  });
}

///////////// Test for kernal 2//////////////////////

//Submit for testing of Delta theta calc
template <typename KernelID, typename Pipe_1>
sycl::event SubmitProducer_test_kernal2(sycl::queue& q, sycl::buffer<float, 1>& in_buf_2) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor in_2(in_buf_2, h, read_only);
    //int size = in_buf.size();
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //   Pipe::write(in[i]);
      // }
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<4>([&index_1, &in_2, &i](auto j) {
              Pipe_1::template PipeAt<j>::write(in_2[i*4 + index_1]);
              index_1++;
        });
      }

    });
  });
}

// consumer kernel for testing of delta theta calc
template <typename KernelID, typename Pipe_out2>
sycl::event SubmitConsumer_test_kernal2(sycl::queue& q, sycl::buffer<float, 1>& out_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out(out_buf, h, write_only, no_init);
    //int size = out_buf.size();
    //int size = 12;
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //     out[i] = Pipe::read();
      // }
       size_t index = 0;
        fpga_tools::UnrolledLoop<4>([&index, &out](auto i) {
          out[index++] = Pipe_out2::template PipeAt<i>::read();
        });
    });
  });
}

///////////// Test for kernal 3//////////////////////

//Submit for testing of Delta theta calc
template <typename KernelID, typename Pipe_1, typename Pipe_2>
sycl::event SubmitProducer_test_kernal3(sycl::queue& q, sycl::buffer<float, 1>& in_buf_p, sycl::buffer<float, 1>& in_buf_epsilon) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor in_p(in_buf_p, h, read_only);
    sycl::accessor in_epsilon(in_buf_epsilon, h, read_only);
    //int size = in_buf.size();
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //   Pipe::write(in[i]);
      // }
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        size_t index_2 = 0;
        fpga_tools::UnrolledLoop<4>([&index_1, &in_p, &i](auto j) {
              Pipe_1::template PipeAt<j>::write(in_p[i*4 + index_1]);
              index_1++;
        });
        fpga_tools::UnrolledLoop<12>([&index_2, &in_epsilon, &i](auto k) {
              Pipe_2::template PipeAt<k>::write(in_epsilon[i*12 + index_2]);
              index_2++;
        });
      }

    });
  });
}

// consumer kernel for testing of delta theta calc
template <typename KernelID, typename Pipe_out1, typename Pipe_out2>
sycl::event SubmitConsumer_test_kernal3(sycl::queue& q, sycl::buffer<float, 1>& out1_buf,sycl::buffer<float, 1>& out2_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out1(out1_buf, h, write_only, no_init);
    sycl::accessor out2(out2_buf, h, write_only, no_init);
    //int size = out_buf.size();
    //int size = 12;
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //     out[i] = Pipe::read();
      // }
       for(int b=0; b<2; b++){
       size_t index1 = 0;
       size_t index2 = 0;
        fpga_tools::UnrolledLoop<3>([&index1, &out1,&b](auto i) {
          out1[b*3+index1++] = Pipe_out1::template PipeAt<i>::read();
        });
        fpga_tools::UnrolledLoop<3>([&index2, &out2,&b](auto k) {
          out2[b*3+index2++] = Pipe_out2::template PipeAt<k>::read();
        });
       }
    });
  });
}

//////////////////Test for the whole////////////////////////////////
// Submit
template <typename KernelID, typename Pipe_in,typename Pipe_ep>
sycl::event SubmitProducer_test_kernalwhole(sycl::queue& q, sycl::buffer<float, 1>& in_buf,sycl::buffer<float, 1>& in_ep_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor in(in_buf, h, read_only);
    sycl::accessor in_ep(in_ep_buf, h, read_only);
    //int size = in_buf.size();
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //   Pipe::write(in[i]);
      // }
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        size_t index_2 = 0;
        fpga_tools::UnrolledLoop<4>([&index_1, &in, &i](auto j) {
              Pipe_in::template PipeAt<j>::write(in[i*4 + index_1]);
              index_1++;
        });
        fpga_tools::UnrolledLoop<12>([&index_2, &in_ep, &i](auto k) {
              Pipe_ep::template PipeAt<k>::write(in_ep[i*12 + index_2]);
              index_2++;
        });
      }
    });
  });
}

// consumer kernel for testing of delta theta calc
template <typename KernelID, typename Pipe_out1, typename Pipe_out2>
sycl::event SubmitConsumer_test_kernalwhole(sycl::queue& q, sycl::buffer<float, 1>& out1_buf,sycl::buffer<float, 1>& out2_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out1(out1_buf, h, write_only, no_init);
    sycl::accessor out2(out2_buf, h, write_only, no_init);
    //int size = out_buf.size();
    //int size = 12;
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //     out[i] = Pipe::read();
      // }
       for(int b=0; b<2; b++){
       size_t index1 = 0;
       size_t index2 = 0;
        fpga_tools::UnrolledLoop<3>([&index1, &out1,&b](auto i) {
          out1[b*3+index1++] = Pipe_out1::template PipeAt<i>::read();
        });
        fpga_tools::UnrolledLoop<3>([&index2, &out2,&b](auto k) {
          out2[b*3+index2++] = Pipe_out2::template PipeAt<k>::read();
        });
       }
    });
  });
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

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
// std::cout<<"input data: \n";
// for(int i=0; i<count; i++){
//   std::cout<<in_data[i];
// }
// std::cout<<"\n";
std::vector<float> in_p(2*4), in_epsilon(2*12), out_value(2*3);
std::fill(out_value.begin(), out_value.end(), -1);
in_p = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
in_epsilon = {
  1.0f, 2.0f, 3.0f, 4.0f,
  2.0f, 3.0f, 3.0f, 4.0f,
  5.0f, 10.0f, 1.5f, 3.25f,
  1.0f, 2.0f, 3.0f, 4.0f,
  5.0f, 10.0f, 1.5f, 3.25f,
  1.0f, 2.0f, 3.0f, 4.0f
};

/////////// DATA FOR TEST OF KERNAL 1 ////////////////
std::vector<float> In = 
{4.0f, 3.0f, 2.0f, 1.0f};
// std::vector<float> Out(3);
// std::fill(Out.begin(), Out.end(), -1);

/////////// DATA FOR TEST OF KERNAL 2 ////////////////
// std::vector<float> In2 = 
// {4.0f, 3.0f, 2.0f, 1.0f};
// std::vector<float> Out2(4);
// std::fill(Out2.begin(), Out2.end(), -1);

/////////// DATA FOR TEST OF KERNAL 3 ////////////////
//std::vector<float> In31(2*4), In32(2*12), Out31(2*3), Out32(2*3);
std::vector<float> In32(2*12), Out31(2*3), Out32(2*3);
std::fill(Out31.begin(), Out31.end(), -1);
std::fill(Out32.begin(), Out32.end(), -1);
//In31 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
In32 = {
  1.0f, 2.0f, 3.0f, 4.0f,
  2.0f, 3.0f, 3.0f, 4.0f,
  5.0f, 10.0f, 1.5f, 3.25f,
  1.0f, 2.0f, 3.0f, 4.0f,
  5.0f, 10.0f, 1.5f, 3.25f,
  1.0f, 2.0f, 3.0f, 4.0f
};




///////////////////////////////////////////////////

// //int number = 832;
// ac_fixed<16, 8, true> check = 3.5f;
// ac_fixed<16, 8, true> divde[2] = {0.5f, 1.5f};
// ac_fixed<16, 8, true> test = check & divde[0];
// std::cout << test <<std::endl;

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
      // // Create input and output buffers
      // sycl::buffer in_buf_p(in_p);
      // sycl::buffer in_buf_epsilon(in_epsilon);
      // sycl::buffer out_buf(out_value);
      // sycl::buffer<float, 1> mid_buf{sycl::range{5}};
      //SubmitProducerKernel_test<ARProducerID, pipe_array_p, pipe_array_epsilon>(q, in_buf_p, in_buf_epsilon);
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
      //SubmitConsumerKernel_test<ARConsumerID, pipe_array_theta_out>(q, out_buf);

      ////////////////////// TEST FOR KERNAL 1 ////////////////////////////////
      //sycl::buffer In_buf_kernal1(In);
      // sycl::buffer Out_buf_kernal1(Out);
      // SubmitProducer_test_kernal1<ARProduceKernel_kernal1_ID, ARKernel1_in_PipeArray>(q, In_buf_kernal1);
      // SubmitConsumer_test_kernal1<ARConsumeKernel_kernal1_ID, ARKernel1_out_PipeArray>(q, Out_buf_kernal1);

      ////////////////////// TEST FOR KERNAL 2 ////////////////////////////////
      // sycl::buffer In_buf_kernal2(In2);
      // sycl::buffer Out_buf_kernal2(Out2);
      // SubmitProducer_test_kernal2<ARProduceKernel_kernal2_ID, ARKernel2_in_PipeArray>(q, In_buf_kernal2);
      // SubmitConsumer_test_kernal2<ARConsumeKernel_kernal2_ID, ARKernel2_out_PipeArray>(q, Out_buf_kernal2);

      ////////////////////// TEST FOR KERNAL 3 ////////////////////////////////
      // sycl::buffer In_buf_kernal31(In31);
      // sycl::buffer Out_buf_kernal31(Out31);
      // sycl::buffer In_buf_kernal32(In32);
      // sycl::buffer Out_buf_kernal32(Out32);
      // SubmitProducer_test_kernal3<ARProduceKernel_kernal3_ID, ARKernel3_in1_PipeArray,ARKernel3_in2_PipeArray>(q, In_buf_kernal31,In_buf_kernal32);
      // SubmitConsumer_test_kernal3<ARConsumeKernel_kernal3_ID, ARKernel3_out1_PipeArray,ARKernel3_out2_PipeArray>(q, Out_buf_kernal31,Out_buf_kernal32);

    ////////////////////// TEST FOR KERNAL whole ////////////////////////////////
      sycl::buffer In_buf_kernal1(In);
      sycl::buffer In_buf_kernal32(In32);
      sycl::buffer Out_buf_kernal31(Out31);
      sycl::buffer Out_buf_kernal32(Out32);
      SubmitProducer_test_kernalwhole<ARProduceKernel_kernal1_ID, ARKernel1_in_PipeArray,ARKernel3_in2_PipeArray>(q, In_buf_kernal1,In_buf_kernal32);
      SubmitConsumer_test_kernalwhole<ARConsumeKernel_kernal3_ID, ARKernel3_out1_PipeArray,ARKernel3_out2_PipeArray>(q, Out_buf_kernal31,Out_buf_kernal32);

    }
    std::cout<<"submit finished\n";

    // validate the results
    // operator== for a vector checks sizes, then checks per-element
    //passed &= (out_data == in_data);
    
    // To show the output buffer
    // std::cout<<"output is: \n";
    // for(int i=0; i<count; i++){
    //   std::cout<<out_data[i]<<"\n";
    // }
    // std::cout<<"\n";
    // std::cout<<"output is: \n";
    // for(int i=0; i<2*3; i++){
    //   std::cout<<out_value[i]<<"\n";
    // }
    // std::cout<<"\n";

   ///////////// Print the out for kernal 1 //////////////
    // std::cout<<"output is: \n";
    // std::cout<<"matrix out1: \n";
    // for(int i=0; i<4; i++){
    //   std::cout<< Out[i]<<std::endl;
    // }
    // std::cout<<"\n";

       ///////////// Print the out for kernal 2 //////////////
    // std::cout<<"output is: \n";
    // std::cout<<"matrix out1: \n";
    // for(int i=0; i<4; i++){
    //   std::cout<< Out2[i]<<std::endl;
    // }
    // std::cout<<"\n";

       ///////////// Print the out for kernal 3 //////////////
    std::cout<<"output1 is: \n";
    for(int i=0; i<2*3; i++){
      std::cout<<Out31[i]<<"\n";
    }
    std::cout<<"output2 is: \n";
    for(int i=0; i<2*3; i++){
      std::cout<<Out32[i]<<"\n";
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
