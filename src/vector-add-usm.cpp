
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
// declare the ID for test of MMUL
class ARProduceKernel_MMUL_ID;
class ARConsumeKernel_MMUL_ID;
class ARMMUL_kernel_ID;
class PipeArray_in_MMUL_ID;
class PipeArray_out1_MMUL_ID;
class PipeArray_out2_MMUL_ID;
// declare the ID for test of ARTupdate
class ARProduceKernel_ARTupdate_ID;
class ARConsumeKernel_ARTupdate_ID;
class ARTupdate_kernel_ID;
class PipeArray_traj_ARTupdate_ID;
class PipeArray_theta_ARTupdate_ID;
class PipeArray_state_ARTupdate_ID;
class PipeArray_out1_ARTupdate_ID;
class PipeArray_out2_ARTupdate_ID;
// declare the ID for test SparseMMUL
class ARProduceKernel_SparseMMUL_ID;
class ARConsumeKernel_SparseMMUL_ID;
class SparseMMUL_kernel_ID;
class PipeArray_state_SparseMMUL_ID;
class PipeArray_in_SparseMMUL_ID;
class PipeArray_out_SparseMMUL_ID;
// declare the ID for test smooth
class ARProduceKernel_smooth_ID;
class ARConsumeKernel_smooth_ID;
class Smooth_Kernel_ID;
class PipeArray_theta_initial_smooth_ID;
class PipeArray_delta_theta_smooth_ID;
class PipeArray_mul_smooth_ID;
class PipeArray_sparse_smooth_ID;
class PipeArray_out_smooth_ID;
class PipeArray_state_smooth_ID;

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
// Define the pipes for test of MMUL
using pipe_array_in_MMUL = fpga_tools::PipeArray<PipeArray_in_MMUL_ID, float, 5, 3>;
using pipe_array_out1_MMUL = fpga_tools::PipeArray<PipeArray_out1_MMUL_ID, float, 5, 3>;
using pipe_array_out2_MMUL = fpga_tools::PipeArray<PipeArray_out2_MMUL_ID, float, 5, 3>;
// Define the pipes for test of ARTupdate
using pipe_array_traj_ARTupdate = fpga_tools::PipeArray<PipeArray_traj_ARTupdate_ID, float, 5, 3>;
using pipe_array_theta_ARTupdate = fpga_tools::PipeArray<PipeArray_theta_ARTupdate_ID, float, 5, 3>;
using pipe_array_state_ARTupdate = fpga_tools::PipeArray<PipeArray_state_ARTupdate_ID, float, 1, 3>;
using pipe_array_out1_ARTupdate = fpga_tools::PipeArray<PipeArray_out1_ARTupdate_ID, float, 5, 3>;
using pipe_array_out2_ARTupdate = fpga_tools::PipeArray<PipeArray_out2_ARTupdate_ID, float, 5, 3>;
// Define the pipes for test of SparseMMUL
using pipe_array_state_SparseMMUL = fpga_tools::PipeArray<PipeArray_state_SparseMMUL_ID, float, 1, 3>;
using pipe_array_in_SparseMMUL = fpga_tools::PipeArray<PipeArray_in_SparseMMUL_ID, float, 5, 3>;
using pipe_array_out_SparseMMUL = fpga_tools::PipeArray<PipeArray_out_SparseMMUL_ID, float, 5, 3>;
// Define the pipes for test of smooth
using pipe_array_theta_initial_smooth = fpga_tools::PipeArray<PipeArray_theta_initial_smooth_ID, float, 5, 3>;
using pipe_array_delta_theta_smooth = fpga_tools::PipeArray<PipeArray_delta_theta_smooth_ID, float, 5, 3>;
using pipe_array_mul_smooth = fpga_tools::PipeArray<PipeArray_mul_smooth_ID, float, 1, 3>;
using pipe_array_sparse_smooth = fpga_tools::PipeArray<PipeArray_sparse_smooth_ID, float, 1, 3>;
using pipe_array_out_smooth = sycl::ext::intel::pipe<PipeArray_out_smooth_ID, float, 1>;
using pipe_array_state_smooth = sycl::ext::intel::pipe<PipeArray_state_smooth_ID, float, 1>;

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
//fpga_tools::Autorun<ARKernel_inID> ar_kernel_in{selector, MyAutorun_in{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, MyAutorun_out{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Autoout<ARKernel_PipeArray, ARConsumePipe>{}};
//fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Delta_Theta_Block::Theta_Calc_Kernel2<ARKernel_PipeArray, ARKernel_PipeArray_2>{}};
//fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, MyAutorun_out_new{}};
//fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipe_array_p, pipe_array_epsilon, pipe_array_theta_out>{}};

//Declare the kernel for MMUL
fpga_tools::Autorun<ARMMUL_kernel_ID> ar_kernel_test_MMUL(selector, smooth::ARMMul<pipe_array_in_MMUL, pipe_array_out1_MMUL, pipe_array_out2_MMUL>{});

//Declare the kernel for ARTupdate
//fpga_tools::Autorun<ARTupdate_kernel_ID> ar_kernel_test_ARTupdate(selector, smooth::ARTUpdate<pipe_array_state_ARTupdate, pipe_array_traj_ARTupdate, pipe_array_theta_ARTupdate, pipe_array_out1_ARTupdate, pipe_array_out2_ARTupdate>{});

//Declare the kernel for SparseMMUL
//fpga_tools::Autorun<SparseMMUL_kernel_ID> ar_kernel_test_SparseMMUL(selector, smooth::ARSparseMul<pipe_array_state_SparseMMUL, pipe_array_in_SparseMMUL, pipe_array_out_SparseMMUL>{});

//Declare the kernel for Smooth
//fpga_tools::Autorun<Smooth_Kernel_ID> ar_kernel_test_Smooth(selector, smooth::ARSmooth<pipe_array_state_smooth, pipe_array_mul_smooth, pipe_array_sparse_smooth, pipe_array_theta_initial_smooth, pipe_array_delta_theta_smooth, pipe_array_out_smooth>{});
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

// Submit for testing of Delta theta calc
// template <typename KernelID, typename Pipe_1, typename Pipe_2>
// sycl::event SubmitProducerKernel_test(sycl::queue& q, sycl::buffer<float, 1>& in_buf_p, sycl::buffer<float, 1>& in_buf_epsilon) {
//   return q.submit([&](sycl::handler& h) {
//     sycl::accessor in_p(in_buf_p, h, read_only);
//     sycl::accessor in_epsilon(in_buf_epsilon, h, read_only);
//     //int size = in_buf.size();
//     h.single_task<KernelID>([=] {
//       // for (int i = 0; i < size; i++) {
//       //   Pipe::write(in[i]);
//       // }
//       for(int i=0; i<2; i++){
//         size_t index_1 = 0;
//         size_t index_2 = 0;
//         fpga_tools::UnrolledLoop<4>([&index_1, &in_p, &i](auto j) {
//               Pipe_1::template PipeAt<j>::write(in_p[i*4 + index_1]);
//               index_1++;
//         });
//         fpga_tools::UnrolledLoop<12>([&index_2, &in_epsilon, &i](auto k) {
//               Pipe_2::template PipeAt<k>::write(in_epsilon[i*12 + index_2]);
//               index_2++;
//         });
//       }

//     });
//   });
// }

//////////////////////////////////////////////////////////////////////////
///////////////////////// TEST FOR MMUL //////////////////////////////////
// Submit
template <typename KernelID, typename Pipe>
sycl::event SubmitProduce_test_MMUL(sycl::queue& q, sycl::buffer<float, 1>& buf_in) {
  return q.submit([&] (sycl::handler &h){
    sycl::accessor IN(buf_in, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<5; i++){
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &IN, &i](auto idx_1) {
                    //val[index_in_val++] = PipeIn::template PipeAt<idx_1>::read();
              Pipe::template PipeAt<idx_1>::write(IN[index*5 + i]);
              index++;
        });
      }
    });
  });
}
// Get
template <typename KernelID, typename Pipeout1, typename Pipeout2>
sycl::event SubmitConsume_test_MMUL(sycl::queue& q, sycl::buffer<float, 1>& buf_out1, sycl::buffer<float, 1>& buf_out2) {
  return q.submit([&] (sycl::handler &h) {
    sycl::accessor OUT1(buf_out1, h, write_only, no_init);
    sycl::accessor OUT2(buf_out2, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<5; i++){
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &OUT1, &i](auto idx_1) {
              OUT1[index*5+i] = Pipeout1::template PipeAt<idx_1>::read();
              index++;
              //Pipe::template PipeAt<idx_1>::write(IN[index++][i]);
        });
        // fpga_tools::UnrolledLoop<3>([&index_2, &OUT2, &i](auto idx_2) {
        //       OUT2[index_2*5+i] = Pipeout2::template PipeAt<idx_2>::read();
        //       index_2++;
        //       //Pipe::template PipeAt<idx_1>::write(IN[index++][i]);
        // });
      }
      size_t index_2 = 0;
      fpga_tools::UnrolledLoop<3>([&index_2, &OUT2](auto idx_2) {
        OUT2[index_2++] = Pipeout2::template PipeAt<idx_2>::read();
      });
    });
  });
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////////////////////////// TEST FOR ARTUpdate ////////////////////////////
// produce
template <typename KernelID, typename PipeInState, typename PipeInInit, typename PipeInMMul>
sycl::event SubmitProduce_test_ARTUpdate(sycl::queue& q, sycl::buffer<float, 1>& buf_in_traj, sycl::buffer<float, 1>& buf_in_mul){
  return q.submit([&] (sycl::handler &h) {
    sycl::accessor IN_TRAJ(buf_in_traj, h, read_only);
    sycl::accessor IN_MUL(buf_in_mul, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        if(i==0){
        PipeInState::template PipeAt<0>::write(1);
        for(int j=0; j<5; j++){
            size_t index_1 = 0;
            size_t index_2 = 0;
            
            fpga_tools::UnrolledLoop<3>([&index_1, &IN_TRAJ, &j](auto idx_1) {
              PipeInInit::template PipeAt<idx_1>::write(IN_TRAJ[index_1*5 + j]);
              index_1++;
              //Pipe::template PipeAt<idx_1>::write(IN[index++][i]);
            });
            fpga_tools::UnrolledLoop<3>([&index_2, &IN_MUL, &i, &j](auto idx_2){
              PipeInMMul::template PipeAt<idx_2>::write(IN_MUL[i*3*5 + index_2*5 + j]);
              index_2++;
            });}}
          else{
            PipeInState::template PipeAt<0>::write(0);
            for(int j=0; j<5; j++){
            size_t index_1 = 0;
            
            fpga_tools::UnrolledLoop<3>([&index_1, &IN_MUL, &i, &j](auto idx_2){
              PipeInMMul::template PipeAt<idx_2>::write(IN_MUL[i*3*5 + index_1*5 + j]);
              index_1++;
            });}
          }
        }
    });
  });
}
// consume
template <typename KernelID, typename Pipe_out1, typename Pipe_out2>
sycl::event SubmitConsume_ARTUpdate(sycl::queue& q, sycl::buffer<float, 1>& out_buf1, sycl::buffer<float, 1>& out_buf2){
  return q.submit([&](sycl::handler& h){
    sycl::accessor out1(out_buf1, h, write_only, no_init);
    sycl::accessor out2(out_buf2, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        for(int j=0; j<5; j++){
          size_t index_1 = 0;
          //size_t index_2 = 0;
          fpga_tools::UnrolledLoop<3>([&index_1, &out1, &out2, &i, &j](auto idx_2){
              // PipeInMMul::template PipeAt<idx_2>::write(IN_MUL[i*3*5 + index_2*5 + j]);
              // index_2++;
              out1[i*3*5+index_1*5+j] = Pipe_out1::template PipeAt<idx_2>::read();
              out2[i*3*5+index_1*5+j] = Pipe_out2::template PipeAt<idx_2>::read();
              index_1++;
            });
        }
      }
    });
  });
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//////////////////// TEST FOR SPARTSEMMUL ////////////////////////////////
//produce
template <typename KernelID, typename Pipe_in, typename Pipe_state>
sycl::event SubmitProduce_SparseMMUL(sycl::queue& q, sycl::buffer<float, 1>& buf_in){
  return q.submit([&](sycl::handler& h){
    sycl::accessor IN(buf_in, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        if(!i){
          Pipe_state::template PipeAt<2>::write(1);
          for(int j=0; j<5; j++){
           
            size_t index_1 =0;
            fpga_tools::UnrolledLoop<3>([&index_1, &IN, &j, &i](auto idx_1){
              Pipe_in::template PipeAt<idx_1>::write(IN[i*3*5 + index_1*5 + j]);
              index_1++;
            });
          }
        }
        else{
          Pipe_state::template PipeAt<2>::write(0);
          // for(int j == 0; j<5; j++){
          //   size_t index_2 = 0;
          //   fpga_tools::UnrolledLoop<3>([&index_2, &IN, &j, &i](auto idx_2){
          //     Pipe_in::template PipeA
          //   })
          // }
        }
      }
    });
  });
}
// Consume
template <typename KernelID, typename Pipe_out>
sycl::event SubmitConsume_SparseMMUL(sycl::queue& q, sycl::buffer<float, 1>& out_buf){
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &out, &i](auto idx){
            out[i*3+index] = Pipe_out::template PipeAt<idx>::read();
            index++;
        });
      }
    });
  });
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//////////////////////// TEST FOR SMOOTH CALC ////////////////////////////
// produce
template <typename KernelID, typename Pipe_in_Mul, typename Pipe_in_sparse, typename Pipe_theta_initial, typename Pipe_delta_theta>
sycl::event SubmitProduce_Smooth(sycl::queue& q, sycl::buffer<float, 1>& buf_mul, sycl::buffer<float, 1>& buf_sparse, sycl::buffer<float, 1>& buf_theta_initial, sycl::buffer<float, 1>& buf_delta_theta){
  return q.submit([&](sycl::handler& h){
    sycl::accessor IN_mul(buf_mul, h, read_only);
    sycl::accessor IN_sparse(buf_sparse, h, read_only);
    sycl::accessor IN_theta_initial(buf_theta_initial, h, read_only);
    sycl::accessor IN_delta_theta(buf_delta_theta, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        for(int j=0; j<5; j++){
          size_t index_1 = 0;
          fpga_tools::UnrolledLoop<3>([&index_1, &IN_theta_initial, &IN_delta_theta, &i, &j](auto idx_1){
            Pipe_theta_initial::template PipeAt<idx_1>::write(IN_theta_initial[i*3*5+index_1*5+j]);
            Pipe_delta_theta::template PipeAt<idx_1>::write(IN_delta_theta[i*3*5+index_1*5+j]);
            index_1++;
          });
        }
        size_t index_2 = 0;
        fpga_tools::UnrolledLoop<3>([&index_2, &IN_mul, &IN_sparse, &i](auto idx_2){
          Pipe_in_Mul::template PipeAt<idx_2>::write(IN_mul[i*3 + index_2]);
          Pipe_in_sparse::template PipeAt<idx_2>::write(IN_sparse[i*3 + index_2]);
          index_2++;
        });
      }
    });
  });
}

// Consume
template <typename KernelID, typename Pipe_out>
sycl::event SubmitConsume_Smooth(sycl::queue& q, sycl::buffer<float, 1>& out_buf){
  return q.submit([&](sycl::handler& h){
    sycl::accessor OUT(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        OUT[i] = Pipe_out::read();
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

// consumer kernel for testing of delta theta calc
template <typename KernelID, typename Pipe>
sycl::event SubmitConsumerKernel_test(sycl::queue& q, sycl::buffer<float, 1>& out_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out(out_buf, h, write_only, no_init);
    //int size = out_buf.size();
    //int size = 12;
    h.single_task<KernelID>([=] {
      // for (int i = 0; i < size; i++) {
      //     out[i] = Pipe::read();
      // }
      for(int j=0; j<2; j++){
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &out, &j](auto i) {
          out[j*3 + index] = Pipe::template PipeAt<i>::read();
          index++;
        });
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


/////////// DATA FOR TEST OF MMUL ////////////////
// float In_MMUL[3][5] = {
//   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
//   {1.5f, 2.5f, 3.5f, 4.5f, 5.5f},
//   {2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
// };
// float Out_MMUL[3][5] = {
//   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
//   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
//   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
// };
std::vector<float> In_MMUL = {
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
  2.0f, 3.0f, 4.0f, 5.0f, 6.0f
};
std::vector<float> Out1_MMUL(3*5);
std::fill(Out1_MMUL.begin(), Out1_MMUL.end(), -1);
std::vector<float> Out2_MMUL(3);
std::fill(Out2_MMUL.begin(), Out2_MMUL.end(), -1);
//////////////////////////////////////////////////

//////////// DATA FOR TEST OF ARTUPDATE ///////////
std::vector<float> In_Traj_ARTupdate = {
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
  2.0f, 3.0f, 4.0f, 5.0f, 6.0f
};
std::vector<float> In_Theta_ARTupdate = {
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,

  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f
};
std::vector<float> Out_ARTupdate_1(2*3*5);
std::fill(Out_ARTupdate_1.begin(), Out_ARTupdate_1.end(), -1);
std::vector<float> Out_ARTupdate_2(2*3*5);
std::fill(Out_ARTupdate_2.begin(), Out_ARTupdate_2.end(), -1);
///////////////////////////////////////////////////

////////// DATA FOR TEST OF SPARSEMMUL ////////////
std::vector<float> In_SparseMMUL = {
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,
  0.5f, 4.6f, 3.28f, 4.35f, 6.78f,
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,

  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f
};
std::vector<float> Out_SparseMMUL(2*3);
std::fill(Out_SparseMMUL.begin(), Out_SparseMMUL.end(), -1);
///////////////////////////////////////////////////

////////// DATA FOR TEST OF SMOOTH ////////////////
std::vector<float> In_theta_initial_smooth = {
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,
  0.5f, 4.6f, 3.28f, 4.35f, 6.78f,
  1.5f, 2.5f, 1.5f, 2.5f, 3.5f,

  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f
};
std::vector<float> In_delta_theta_smooth = {
  1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
  2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
  4.3f, 4.4f, 2.1f, 4.2f, 5.5f,

  1.0f, 4.3f, 3.9f, 4.4f, 3.3f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f
};
std::vector<float> In_mul_smooth = {
  2.0f, 3.0f, 4.0f,

  1.0f, 2.0f, 3.0f
};
std::vector<float> In_sparse_smooth = {
  1.0f, 2.0f, 3.0f,

  2.0f, 3.0f, 4.0f
};
std::vector<float> out_smooth = {-1, -1};
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
      // Create input and output buffers
      sycl::buffer in_buf_p(in_p);
      sycl::buffer in_buf_epsilon(in_epsilon);
      sycl::buffer out_buf(out_value);
      sycl::buffer<float, 1> mid_buf{sycl::range{5}};
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

      ////////////////////// TEST FOR MMUL ////////////////////////////////
      sycl::buffer In_buf_MMUL(In_MMUL);
      sycl::buffer Out1_buf_MMUL(Out1_MMUL);
      sycl::buffer Out2_buf_MMUL(Out2_MMUL);
      SubmitProduce_test_MMUL<ARProduceKernel_MMUL_ID, pipe_array_in_MMUL>(q, In_buf_MMUL);
      SubmitConsume_test_MMUL<ARConsumeKernel_MMUL_ID, pipe_array_out1_MMUL, pipe_array_out2_MMUL>(q, Out1_buf_MMUL, Out2_buf_MMUL);

      /////////////////////////////////////////////////////////////////////
      ////////////////////// TEST FOR ARTUPDATE ///////////////////////////
      // sycl::buffer In_buf_traj_ARTupdate(In_Traj_ARTupdate);
      // sycl::buffer In_buf_theta_ARTupdate(In_Theta_ARTupdate);
      // sycl::buffer Out_buf_1_ARTupdate(Out_ARTupdate_1);
      // sycl::buffer Out_buf_2_ARTupdate(Out_ARTupdate_2);
      // SubmitProduce_test_ARTUpdate<ARProduceKernel_ARTupdate_ID, pipe_array_state_ARTupdate, pipe_array_traj_ARTupdate, pipe_array_theta_ARTupdate>(q, In_buf_traj_ARTupdate, In_buf_theta_ARTupdate);
      // SubmitConsume_ARTUpdate<ARConsumeKernel_ARTupdate_ID, pipe_array_out1_ARTupdate, pipe_array_out2_ARTupdate>(q, Out_buf_1_ARTupdate, Out_buf_2_ARTupdate);

      /////////////////////////////////////////////////////////////////////
      ////////////////////// TEST FOR SPARSEMMUL //////////////////////////
      // sycl::buffer In_buf_SparseMMUL(In_SparseMMUL);
      // sycl::buffer out_buf_SparseMMUL(Out_SparseMMUL);
      // SubmitProduce_SparseMMUL<ARProduceKernel_SparseMMUL_ID, pipe_array_in_SparseMMUL, pipe_array_state_SparseMMUL>(q, In_buf_SparseMMUL);
      // SubmitConsume_SparseMMUL<ARConsumeKernel_SparseMMUL_ID, pipe_array_out_SparseMMUL>(q, out_buf_SparseMMUL);

      /////////////////////////////////////////////////////////////////////
      ////////////////////// TEST FOR SMOOTH //////////////////////////////
      // sycl::buffer In_buf_theta_initial_smooth(In_theta_initial_smooth);
      // sycl::buffer In_buf_delta_theta_smooth(In_delta_theta_smooth);
      // sycl::buffer In_buf_mul_smooth(In_mul_smooth);
      // sycl::buffer In_buf_sparse_smooth(In_sparse_smooth);
      // sycl::buffer out_buf_smooth(out_smooth);
      // SubmitProduce_Smooth<ARProduceKernel_smooth_ID, pipe_array_mul_smooth, pipe_array_sparse_smooth, pipe_array_theta_initial_smooth, pipe_array_delta_theta_smooth>(q, In_buf_mul_smooth, In_buf_sparse_smooth, In_buf_theta_initial_smooth, In_buf_delta_theta_smooth);
      // SubmitConsume_Smooth<ARConsumeKernel_smooth_ID, pipe_array_out_smooth>(q, out_buf_smooth);

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

    ///////////// Print the out for MMUL //////////////
    std::cout<<"output is: \n";
    std::cout<<"matrix out1: \n";
    for(int i=0; i<3; i++){
      for(int j=0; j<5; j++){
        std::cout<<Out1_MMUL[i*5+j]<<' ';
      }
      std::cout << "\n";
    }
    std::cout<<"\n";
    std::cout<<"matrix out2: \n";
    for(int i=0; i<3; i++){
      std::cout<< Out2_MMUL[i]<<std::endl;
    }
    std::cout<<"\n";

    ///////////// Print the out for ARTupdate //////////
    // std::cout<<"output is: \n";
    // for(int i=0; i<2; i++){
    //   std::cout<<"The iteration "<<i<<"\n";
    //   std::cout<<"The first output: \n";
    //   for(int j=0; j<3; j++){
    //     for(int k=0; k<5; k++){
    //       std::cout<<Out_ARTupdate_1[i*15+j*5+k]<<' ';
    //     }
    //     std::cout<<"\n";
    //   }
    //   std::cout<<"The second output: \n";
    //   for(int j=0; j<3; j++){
    //     for(int k=0; k<5; k++){
    //       std::cout<<Out_ARTupdate_2[i*15+j*5+k]<<' ';
    //     }
    //     std::cout<<"\n";
    //   }
    //   std::cout<<"\n";
    // }
    // std::cout<<"\n";

    ///////////// Print the out for SparseMMUL //////////
    // std::cout<<"output is: \n";
    // for(int i=0; i<2; i++){
    //   std::cout<<"iteration: "<<i<<"\n";
    //   for(int j=0; j<3; j++){
    //     std::cout<<Out_SparseMMUL[i*3+j]<<"\n";
    //   }
    // }
    // std::cout<<"\n";

    ///////////// Print the out for Smooth //////////////
    // std::cout<<"output is: \n";
    // for(int i=0; i<2; i++){
    //   std::cout<<"iteration: "<<i<<"\n";
    //   std::cout<<out_smooth[i]<<std::endl;
    // }
    // std::cout<<"\n";

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
