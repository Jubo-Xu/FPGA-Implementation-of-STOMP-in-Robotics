
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
// #include <sycl/ext/intel/fpga_extensions.hpp>
#include "Adder_Tree.h"
#include "CostFunction.h"
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include "Delta_Theta_Block.hpp"
#include "CostSmooth.hpp"
#include "Determination.hpp"
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
class AR_4_1_kernel_ID; // mainly for testing

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

// declare AR_Determine_Block
class ARProduceKernel_deteermine_ID;
class ARConsumeKernel_deteermine_ID;
class Deteermine_Kernel_ID;
class PipeArray_in_cost_obstacle_deteermine_ID;
class PipeArray_in_cost_smooth_deteermine_ID;
class PipeArray_in_theta_deteermine_ID;
class PipeArray_out_new_theta_deteermine_ID;
class PipeArray_out_final_theta_deteermine_ID;
class PipeArray_out_end_deteermine_ID;

// declare AR_input_Block
class ARProduceKernel_input_ID;
class ARConsumeKernel_input_ID;
class Input_Kernel_ID;
class PipeArray_in_theta_initial_input_ID;
class PipeArray_in_theta_loop_input_ID;
class PipeArray_in_end_input_ID;
class PipeArray_out1_input_ID;
class PipeArray_out2_input_ID;
class PipeArray_out3_input_ID;
class PipeArray_out4_input_ID;

// declare AR_Input_2_kNoisy
class ARProduceKernel_knoisy_ID;
class ARConsumeKernel_knoisy_ID;
class Knoisy_Kernel_ID;
class PipeArray_in_theta_knoisy_ID;
class PipeArray_in_RNG_knoisy_ID;
class PipeArray_out_knoisy_ID;

// declare AR_input_Block + AR_Input_2_kNoisy
class Input_whole_Kernel_ID;
class PipeArray_input_2_noisy_ID;


// pipes
using ARProducePipe = sycl::ext::intel::pipe<ARProducePipeID, float>;
using ARConsumePipe = sycl::ext::intel::pipe<ARConsumePipeID, float>;
using ARKernel_self_inPipe = sycl::ext::intel::pipe<ARKernel_self_inID, float, 1>;
using ARKernel_self_outPipe = sycl::ext::intel::pipe<ARKernel_self_outID, float>;
using ARKernel_PipeArray = fpga_tools::PipeArray<PipeArray_1_ID, float, 1, 4>;
using ARKernel_PipeArray_2 = fpga_tools::PipeArray<PipeArray_2_ID, float, 1, 4>;
using pipe_array_p = fpga_tools::PipeArray<PipeArray_p_ID, float, 1, 4>;
using pipe_array_epsilon = fpga_tools::PipeArray<PipeArray_epsilon_ID, float, 1, 12>;
using pipe_array_theta_out = fpga_tools::PipeArray<PipeArray_theta_out_ID, float, 1, 3>;

// pipes for AR_Determine_Block
using ARDetermine_in_cost_obstacle_PipeArray = sycl::ext::intel::pipe<PipeArray_in_cost_obstacle_deteermine_ID, float>;
using ARDetermine_in_cost_smooth_PipeArray = sycl::ext::intel::pipe<PipeArray_in_cost_smooth_deteermine_ID, float>;
using ARDetermine_in_theta_PipeArray = fpga_tools::PipeArray<PipeArray_in_theta_deteermine_ID, float, 5, 3>;
using ARDetermine_out_new_theta_PipeArray = fpga_tools::PipeArray<PipeArray_out_new_theta_deteermine_ID, float, 5, 3>;
using ARDetermine_out_final_theta_PipeArray = fpga_tools::PipeArray<PipeArray_out_final_theta_deteermine_ID, float, 5, 3>;
using ARDetermine_out_end_PipeArray = fpga_tools::PipeArray<PipeArray_out_end_deteermine_ID, float, 1, 2>;

// pipes for AR_Input_Block
using ARInput_in_theta_initial_PipeArray = sycl::ext::intel::pipe<PipeArray_in_theta_initial_input_ID, float>;
using ARInput_in_theta_loop_PipeArray = fpga_tools::PipeArray<PipeArray_in_theta_loop_input_ID, float,5,3>;
using ARInput_in_end_PipeArray = fpga_tools::PipeArray<PipeArray_in_end_input_ID, float,1,2>;
using ARInput_out1_PipeArray = fpga_tools::PipeArray<PipeArray_out1_input_ID, float, 5, 3>;
using ARInput_out2_PipeArray = fpga_tools::PipeArray<PipeArray_out2_input_ID, float, 5, 3>;
using ARInput_out3_PipeArray = fpga_tools::PipeArray<PipeArray_out3_input_ID, float, 5,3>;
using ARInput_out4_PipeArray = fpga_tools::PipeArray<PipeArray_out4_input_ID, float, 5,3>;

// pipes for AR_Input_2_kNoisy
using ARKnoisy_in_theta_PipeArray = fpga_tools::PipeArray<PipeArray_in_theta_knoisy_ID, float,1,3>;
using ARKnoisy_in_RNG_PipeArray = fpga_tools::PipeArray<PipeArray_in_RNG_knoisy_ID, float,1,12>;
using ARKnoisy_out_PipeArray = fpga_tools::PipeArray<PipeArray_out_knoisy_ID, float, 5, 12>;

//pipes for input+noisy
using ARInput_in_theta_initial_PipeArray = sycl::ext::intel::pipe<PipeArray_in_theta_initial_input_ID, float>;
using ARInput_in_theta_loop_PipeArray = fpga_tools::PipeArray<PipeArray_in_theta_loop_input_ID, float,5,3>;
using ARInput_in_end_PipeArray = fpga_tools::PipeArray<PipeArray_in_end_input_ID, float,1,2>;
using ARInput_out1_PipeArray = fpga_tools::PipeArray<PipeArray_out1_input_ID, float, 5, 3>;
using ARInput_out2_PipeArray = fpga_tools::PipeArray<PipeArray_out2_input_ID, float, 5, 3>;
using ARInput_out3_PipeArray = fpga_tools::PipeArray<PipeArray_out3_input_ID, float, 5,3>;
using ARInput_input_2_noisy_PipeArray = fpga_tools::PipeArray<PipeArray_input_2_noisy_ID, float, 5, 3>
using ARKnoisy_in_RNG_PipeArray = fpga_tools::PipeArray<PipeArray_in_RNG_knoisy_ID, float,5,12>;
using ARKnoisy_out_PipeArray = fpga_tools::PipeArray<PipeArray_out_knoisy_ID, float, 5, 12>;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list)
{
  for (std::exception_ptr const &e : e_list)
  {
    try
    {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e)
    {
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
// std::array<std::array<float, N>, LOG_N> &datain;

// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
// fpga_tools::Autorun<ARKernel_inID> ar_kernel_in{selector, MyAutorun_in{}};
// fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, MyAutorun_out{}};
// fpga_tools::Autorun<ARKernel_outID> ar_kernel_out{selector, Autoout<ARKernel_PipeArray, ARConsumePipe>{}};
// fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, MyAutorun_out_new{}};
// fpga_tools::Autorun<AR_4_1_kernel_ID> ar_kernel_parallel_to_serial{selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipe_array_p, pipe_array_epsilon, pipe_array_theta_out>{}};


// fpga_tools::Autorun<Input_Kernel_ID> ar_kernel_parallel_to_serial{selector, Determine_and_Connections::AR_input_block<ARInput_in_theta_initial_PipeArray, ARInput_in_theta_loop_PipeArray, ARInput_in_end_PipeArray,ARInput_out1_PipeArray, ARInput_out2_PipeArray,ARInput_out3_PipeArray,ARInput_out4_PipeArray>{}};
// fpga_tools::Autorun<Knoisy_Kernel_ID> ar_kernel_parallel_to_serial{selector, Determine_and_Connections::AR_Input_2_kNoisy<ARKnoisy_in_theta_PipeArray, ARKnoisy_in_RNG_PipeArray, ARKnoisy_out_PipeArray>{}};

//declare for input+noisy
fpga_tools::Autorun<Input_Kernel_ID> ar_kernel_parallel_to_serial{selector, Determine_and_Connections::AR_input_block<ARInput_in_theta_initial_PipeArray, ARInput_in_theta_loop_PipeArray, ARInput_in_end_PipeArray,ARInput_out1_PipeArray, ARInput_out2_PipeArray,ARInput_out3_PipeArray,ARInput_input_2_noisy_PipeArray>{}};
fpga_tools::Autorun<Knoisy_Kernel_ID> ar_kernel_knoisy{selector, Determine_and_Connections::AR_Input_2_kNoisy<ARInput_input_2_noisy_PipeArray, ARKnoisy_in_RNG_PipeArray, ARKnoisy_out_PipeArray>{}};

///////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// Submit a kernel to read data from global memory and write to a pipe
//
// template <typename KernelID, typename Pipe>
// sycl::event SubmitProducerKernel(sycl::queue &q, sycl::buffer<float, 1> &in_buf)
// {
//   return q.submit([&](sycl::handler &h)
//                   {
//     sycl::accessor in(in_buf, h, read_only);
//     int size = in_buf.size();
//     h.single_task<KernelID>([=] {
//       for (int i = 0; i < size; i++) {
//         Pipe::write(in[i]);
//       }
//     }); });
// }

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// Submit a kernel to read data from a pipe and write to global memory
//
// template <typename KernelID, typename Pipe>
// sycl::event SubmitConsumerKernel(sycl::queue &q, sycl::buffer<float, 1> &out_buf)
// {
//   return q.submit([&](sycl::handler &h)
//                   {
//     sycl::accessor out(out_buf, h, write_only, no_init);
//     //int size = out_buf.size();
//     int size = 12;
//     h.single_task<KernelID>([=] {
//       for (int i = 0; i < size; i++) {
//           out[i] = Pipe::read();
//       }
//     }); });
// }

// template <typename KernelID, typename Pipe1, typename Pipe2>
// sycl::event ExecuteKernel(sycl::queue& q) {
//   return q.single_task(MyAutorun_in<Pipe1, Pipe2>{});
// }


  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////// TEST FOR Determine //////////////////////////////////
  // Submit
  template <typename KernelID, typename Pipe_cost_obstacle, typename Pipe_cost_smooth, typename Pipe_theta>
  sycl::event SubmitProduce_test_determine(sycl::queue & q,  sycl::buffer<float, 1> & buf_in_cost_obstacle, sycl::buffer<float, 1> & buf_in_cost_smooth,sycl::buffer<float, 1> & buf_in_theta)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor IN_theta(buf_in_theta, h, read_only);
    sycl::accessor IN_cost_smooth(buf_in_cost_smooth, h, read_only);
    sycl::accessor IN_cost_obstacle(buf_in_cost_obstacle, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        //if(i==0){
        //PipeInState::template PipeAt<0>::write(1);
        for(int j=0; j<5; j++){
        Pipe_cost_obstacle::write(IN_cost_obstacle[i]);
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &IN_theta, &i,&j](auto idx_1) {
                    //val[index_in_val++] = PipeIn::template PipeAt<idx_1>::read();
              Pipe_theta::template PipeAt<idx_1>::write(IN_theta[i*3*5 + index*5 + j]);
              index++;
        });
        }
        Pipe_cost_smooth::write(IN_cost_smooth[0]);
      }
    }); });
  }

  // Consume
  template <typename KernelID, typename Pipe_out_new, typename Pipe_out_final,typename Pipe_out_end>
  sycl::event SubmitConsume_test_determine(sycl::queue & q, sycl::buffer<float, 1> & out_new_buf, sycl::buffer<float, 1> & out_final_buf, sycl::buffer<float, 1> & out_end_buf)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor OUT_new(out_new_buf, h, write_only, no_init);
    sycl::accessor OUT_final(out_final_buf, h, write_only, no_init);
    sycl::accessor OUT_end(out_end_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        OUT_end[i] = Pipe_out_end::template PipeAt<0>::read();
      // //consume for out_new only
      for(int j=0; j<5; j++){
          size_t index_1 = 0;
          fpga_tools::UnrolledLoop<3>([&index_1, &OUT_new, &i, &j](auto idx_1){
              OUT_new[i*3*5+index_1*5+j] = Pipe_out_new::template PipeAt<idx_1>::read();
              index_1++;
            });
        }
        //Consume for out_final only
        // for(int j=0; j<15; j++){
        //   size_t index_1 = 0;
        //   fpga_tools::UnrolledLoop<3>([&index_1, &OUT_final, &i, &j](auto idx_1){
        //       OUT_final[i*3*5+(j%3)*5+(j/3)] = Pipe_out_final::template PipeAt<idx_1>::read();
        //       index_1++;
        //     });
        // }
      }
    }); });
  }

  ///////////////////////// TEST FOR Input_block //////////////////////////////////
  // Submit
  template <typename KernelID, typename Pipe_theta_initial, typename Pipe_theta_loop, typename Pipe_end>
  sycl::event SubmitProduce_test_input(sycl::queue & q,  sycl::buffer<float, 1> & buf_in_theta_initial, sycl::buffer<float, 1> & buf_in_theta_loop,sycl::buffer<float, 1> & buf_in_end)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor IN_theta_initial(buf_in_theta_initial, h, read_only);
    sycl::accessor IN_theta_loop(buf_in_theta_loop, h, read_only);
    sycl::accessor IN_end(buf_in_end, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        //if(i==0){
        //PipeInState::template PipeAt<0>::write(1);
        for(int j=0; j<5; j++){
          for (int k=0; k<3; k++){
            Pipe_theta_initial::write(IN_theta_initial[0]);
          }
        // size_t index = 0;
        // fpga_tools::UnrolledLoop<3>([&index, &IN_theta_loop, &i,&j](auto idx_1) {
        //             //val[index_in_val++] = PipeIn::template PipeAt<idx_1>::read();
        //       Pipe_theta_loop::template PipeAt<idx_1>::write(IN_theta_loop[i*3*5 + index*5 + j]);
        //       index++;
        // });
        }
        Pipe_end::template PipeAt<0>::write(IN_end[i]);
      }
    }); });
  }

  // Consume
  template <typename KernelID, typename Pipe_out1, typename Pipe_out2,typename Pipe_out3, typename Pipe_out4>
  sycl::event SubmitConsume_test_input(sycl::queue & q, sycl::buffer<float, 1> & out1_buf, sycl::buffer<float, 1> & out2_buf, sycl::buffer<float, 1> & out3_buf,sycl::buffer<float, 1> & out4_buf)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor OUT1(out1_buf, h, write_only, no_init);
    sycl::accessor OUT2(out2_buf, h, write_only, no_init);
    sycl::accessor OUT3(out3_buf, h, write_only, no_init);
    sycl::accessor OUT4(out4_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
      // //consume for out_new only
      for(int j=0; j<5; j++){
          size_t index_1 = 0;
          fpga_tools::UnrolledLoop<3>([&index_1, &OUT1, &i, &j, &OUT2, &OUT3, &OUT4](auto idx_1){
              OUT1[i*3*5+index_1*5+j] = Pipe_out1::template PipeAt<idx_1>::read();
              OUT2[i*3*5+index_1*5+j] = Pipe_out2::template PipeAt<idx_1>::read();
              OUT3[i*3*5+index_1*5+j] = Pipe_out3::template PipeAt<idx_1>::read();
              OUT4[i*3*5+index_1*5+j] = Pipe_out4::template PipeAt<idx_1>::read();
              index_1++;
            });
      }
      }
    }); });
  }

   ///////////////////////// TEST FOR AR_Input_2_kNoisy//////////////////////////////////
  // Submit
  template <typename KernelID, typename Pipe_theta, typename Pipe_RNG>
  sycl::event SubmitProduce_test_knoisy(sycl::queue & q,  sycl::buffer<float, 1> & buf_in_theta, sycl::buffer<float, 1> & buf_in_RNG)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor IN_theta(buf_in_theta, h, read_only);
    sycl::accessor IN_RNG(buf_in_RNG, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<12>([&index_1,  &IN_RNG, &i](auto idx_1){
          Pipe_RNG::template PipeAt<idx_1>::write(IN_RNG[i*3 + index_1]);
          index_1++;
        });
        size_t index_2 = 0;
        fpga_tools::UnrolledLoop<3>([&index_2, &IN_theta, &i](auto idx_2){
          Pipe_theta::template PipeAt<idx_2>::write(IN_theta[i*3 + index_2]);
          index_2++;
        });
      }
    }); });
  }

  // Consume
  template <typename KernelID, typename Pipe_out>
  sycl::event SubmitConsume_test_knoisy(sycl::queue & q, sycl::buffer<float, 1> & out_buf)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor OUT(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        size_t index = 0;
        fpga_tools::UnrolledLoop<12>([&index, &OUT, &i](auto idx){
            OUT[i*3+index] = Pipe_out::template PipeAt<idx>::read();
            index++;
        });
      }
    }); });
  }


  ///////////////////////// TEST FOR Input + noisy//////////////////////////////////
  // Submit
  template <typename KernelID, typename Pipe_theta_initial, typename Pipe_theta_loop, typename Pipe_end, typename Pipe_RNG>
  sycl::event SubmitProduce_test_input_noisy(sycl::queue & q,  sycl::buffer<float, 1> & buf_in_theta_initial, sycl::buffer<float, 1> & buf_in_theta_loop,sycl::buffer<float, 1> & buf_in_end, sycl::buffer<float, 1> & buf_in_RNG)
  {
    return q.submit([&](sycl::handler &h) {
    sycl::accessor IN_theta_initial(buf_in_theta_initial, h, read_only);
    sycl::accessor IN_theta_loop(buf_in_theta_loop, h, read_only);
    sycl::accessor IN_end(buf_in_end, h, read_only);
    sycl::accessor IN_RNG(buf_in_RNG, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        //if(i==0){
        //PipeInState::template PipeAt<0>::write(1);
        for(int j=0; j<5; j++){
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<12>([&index_1,  &IN_RNG, &i,&j](auto idx_1){
          Pipe_RNG::template PipeAt<idx_1>::write(IN_RNG[i*12*5 + index_1*5+j]);
          index_1++;
        });
          for (int k=0; k<3; k++){
            Pipe_theta_initial::write(IN_theta_initial[0]);
          }
        // size_t index = 0;
        // fpga_tools::UnrolledLoop<3>([&index, &IN_theta_loop, &i,&j](auto idx_1) {
        //             //val[index_in_val++] = PipeIn::template PipeAt<idx_1>::read();
        //       Pipe_theta_loop::template PipeAt<idx_1>::write(IN_theta_loop[i*3*5 + index*5 + j]);
        //       index++;
        // });
        }
        Pipe_end::template PipeAt<0>::write(IN_end[i]);
      }
    }); });
  }

   // Consume
  template <typename KernelID, typename Pipe_out1, typename Pipe_out2,typename Pipe_out3, typename Pipe_out>
  sycl::event SubmitConsume_test_input_noisy(sycl::queue & q, sycl::buffer<float, 1> & out1_buf, sycl::buffer<float, 1> & out2_buf, sycl::buffer<float, 1> & out3_buf, sycl::buffer<float, 1> & out_buf)
  {
    return q.submit([&](sycl::handler &h){
    sycl::accessor OUT1(out1_buf, h, write_only, no_init);
    sycl::accessor OUT2(out2_buf, h, write_only, no_init);
    sycl::accessor OUT3(out3_buf, h, write_only, no_init);
    sycl::accessor OUT(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        for(int j=0; j<5; j++){
          size_t index_1 = 0;
          fpga_tools::UnrolledLoop<3>([&index_1, &OUT1, &i, &j, &OUT2, &OUT3](auto idx_1){
              OUT1[i*3*5+index_1*5+j] = Pipe_out1::template PipeAt<idx_1>::read();
              OUT2[i*3*5+index_1*5+j] = Pipe_out2::template PipeAt<idx_1>::read();
              OUT3[i*3*5+index_1*5+j] = Pipe_out3::template PipeAt<idx_1>::read();
              index_1++;
            });
            size_t index = 0;
           fpga_tools::UnrolledLoop<12>([&index, &OUT, &i,&j](auto idx){
            OUT[i*12*5+index*5+j] = Pipe_out::template PipeAt<idx>::read();
            index++;
        });
      }
      }
    }); });
  }

int main()
{

  std::cout << "un peu d'espoir\n";
  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// TEST FOR AUTORUN //////////////////////////////////

  // // Number of total numbers of input
  // int count = 12;
  // //bool passed = true;

  // std::vector<float> in_data(count), out_data(count);
  // // Clear the output buffer
  // std::fill(out_data.begin(), out_data.end(), -1);
  // // Initialize the input buffer
  // // for(int i=0; i<count; i++){
  // //   in_data[i] = i+1;
  // // }
  // //in_data = {1.0f, 2.0f, 2.0f, 3.0f, 2.5f, 4.0f, 5.5f, 2.0f, 8.0f, 3.5f};
  // in_data = {4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 3.0f};
  // // std::cout<<"input data: \n";
  // // for(int i=0; i<count; i++){
  // //   std::cout<<in_data[i];
  // // }
  // // std::cout<<"\n";
  // std::vector<float> in_p(2*4), in_epsilon(2*12), out_value(2*3);
  // std::fill(out_value.begin(), out_value.end(), -1);
  // in_p = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  // in_epsilon = {
  //   1.0f, 2.0f, 3.0f, 4.0f,
  //   2.0f, 3.0f, 3.0f, 4.0f,
  //   5.0f, 10.0f, 1.5f, 3.25f,
  //   1.0f, 2.0f, 3.0f, 4.0f,
  //   5.0f, 10.0f, 1.5f, 3.25f,
  //   1.0f, 2.0f, 3.0f, 4.0f
  // };
  ////////////Test number for determine////////////////
  // std::vector<float> In_theta = {
  //     1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  //     1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
  //     2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

  //     1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
  //     1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
  //     2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // std::vector<float> In_theta = {
  //     0, 0.1f, 0, 0, 0.02f,
  //     0, 0, 0, 0, 0,
  //     0.2f, 0.003f, 0.05f, 0.012f, 0.014f,

  //     0.2f, 0.003f, 0.05f, 0.012f, 0.014f,
  //     0, 0.1f, 0, 0, 0.02f,
  //     0, 0, 0, 0, 0};
  // std::vector<float> In_cost_obstacle = {2.0f, 3.0f};
  // std::vector<float> In_cost_smooth = {3.0f, 4.0f};
  // std::vector<float> Out_final(2 * 3 * 5);
  // std::fill(Out_final.begin(), Out_final.end(), -1);
  // std::vector<float> Out_new(2 * 3 * 5);
  // std::fill(Out_new.begin(), Out_new.end(), -1);
  // std::vector<float> Out_end = {-1, -1};

    ////////////Test number for input////////////////
//   std::vector<float> In_theta_loop = {
//   1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
//   2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
//   4.3f, 4.4f, 2.1f, 4.2f, 5.5f,

//   1.0f, 4.3f, 3.9f, 4.4f, 3.3f,
//   1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
//   1.0f, 2.0f, 3.0f, 4.0f, 5.0f
// };

//   std::vector<float> In_theta_initial = {
//   1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
//   2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
//   4.3f, 4.4f, 2.1f, 4.2f, 5.5f};
//   std::vector<float> In_end = {0, 0};
//   std::vector<float> Out1(2 * 3 * 5);
//   std::fill(Out1.begin(), Out1.end(), -1);
//   std::vector<float> Out2(2 * 3 * 5);
//   std::fill(Out2.begin(), Out2.end(), -1);
//   std::vector<float> Out3(2 * 3 * 5);
//   std::fill(Out3.begin(), Out3.end(), -1);
//   std::vector<float> Out4(2 * 3 * 5);
//   std::fill(Out4.begin(), Out4.end(), -1);

// ////////////Test number for knoisy////////////////
//   std::vector<float> In_theta = {
//   2.0f, 3.0f, 4.0f,

//   1.0f, 2.0f, 3.0f
// };
// std::vector<float> In_RNG = {
//   1.0f, 2.0f, 3.0f,

//   2.0f, 3.0f, 4.0f,
//   1.0f, 2.0f, 3.0f,

//   2.0f, 3.0f, 4.0f
// };
// std::vector<float> Out(2*12);
// std::fill(Out.begin(), Out.end(), -1);

///////DATA for input+noisy//////////////
   ////////////Test number for input////////////////
  std::vector<float> In_theta_loop = {
  1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
  2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
  4.3f, 4.4f, 2.1f, 4.2f, 5.5f,

  1.0f, 4.3f, 3.9f, 4.4f, 3.3f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f
};

  std::vector<float> In_theta_initial = {
  1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
  2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
  4.3f, 4.4f, 2.1f, 4.2f, 5.5f};
  std::vector<float> In_end = {0, 0};
  std::vector<float> Out1(2 * 3 * 5);
  std::fill(Out1.begin(), Out1.end(), -1);
  std::vector<float> Out2(2 * 3 * 5);
  std::fill(Out2.begin(), Out2.end(), -1);
  std::vector<float> Out3(2 * 3 * 5);
  std::fill(Out3.begin(), Out3.end(), -1);
  std::vector<float> Out4(2 * 3 * 5);
  std::fill(Out4.begin(), Out4.end(), -1);

////////////Test number for knoisy////////////////
  std::vector<float> In_theta = {
  2.0f, 3.0f, 4.0f,

  1.0f, 2.0f, 3.0f
};
std::vector<float> In_RNG = {
  1.0f, 2.0f, 3.0f,

  2.0f, 3.0f, 4.0f,
  1.0f, 2.0f, 3.0f,

  2.0f, 3.0f, 4.0f
};
std::vector<float> Out(2*12*5);
std::fill(Out.begin(), Out.end(), -1);

  ///////////////////////////////////////////////////

  // //int number = 832;
  // ac_fixed<16, 8, true> check = 3.5f;
  // ac_fixed<16, 8, true> divde[2] = {0.5f, 1.5f};
  // ac_fixed<16, 8, true> test = check & divde[0];
  // std::cout << test <<std::endl;

  try
  {
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
      // SubmitProducerKernel_test<ARProducerID, pipe_array_p, pipe_array_epsilon>(q, in_buf_p, in_buf_epsilon);
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
      // SubmitConsumerKernel_test<ARConsumerID, pipe_array_theta_out>(q, out_buf);

      ////////////////////// TEST FOR Determine ////////////////////////////////
      // sycl::buffer In_theta_buf(In_theta);
      // sycl::buffer In_cost_obstacle_buf(In_cost_obstacle);
      // sycl::buffer In_cost_smooth_buf(In_cost_smooth);
      // // sycl::buffer<float, 1> In_cos_obstacle_buf{sycl::range{1}};
      // // sycl::buffer<float, 1> In_cost_smooth_buf{sycl::range{1}};
      // sycl::buffer Out_new_buf(Out_new);
      // sycl::buffer Out_final_buf(Out_final);
      // sycl::buffer Out_end_buf(Out_end);
      // // {
      // //   sycl::host_accessor a{In_cos_obstacle_buf};
      // //   a[0] = In_cost_obstacle;
      // //   sycl::host_accessor b{In_cost_smooth_buf};
      // //   b[0] = In_cost_smooth;
      // // }
      // SubmitProduce_test_determine<ARProduceKernel_deteermine_ID, ARDetermine_in_cost_obstacle_PipeArray, ARDetermine_in_cost_smooth_PipeArray, ARDetermine_in_theta_PipeArray>(q, In_cost_obstacle_buf, In_cost_smooth_buf, In_theta_buf);
      // SubmitConsume_test_determine<ARConsumeKernel_deteermine_ID, ARDetermine_out_new_theta_PipeArray, ARDetermine_out_final_theta_PipeArray, ARDetermine_out_end_PipeArray>(q, Out_new_buf, Out_final_buf, Out_end_buf);
    
     ////////////////////// TEST FOR Input ////////////////////////////////
      // sycl::buffer In_theta_loop_buf(In_theta_loop);
      // sycl::buffer In_theta_initial_buf(In_theta_initial);
      // sycl::buffer In_end_buf(In_end);
      // sycl::buffer Out1_buf(Out1);
      // sycl::buffer Out2_buf(Out2);
      // sycl::buffer Out3_buf(Out3);
      // sycl::buffer Out4_buf(Out4);
      // SubmitProduce_test_input<ARProduceKernel_input_ID, ARInput_in_theta_initial_PipeArray, ARInput_in_theta_loop_PipeArray, ARInput_in_end_PipeArray>(q, In_theta_initial_buf, In_theta_loop_buf, In_end_buf);
      // SubmitConsume_test_input<ARConsumeKernel_input_ID, ARInput_out1_PipeArray, ARInput_out2_PipeArray, ARInput_out3_PipeArray,ARInput_out4_PipeArray>(q, Out1_buf, Out2_buf, Out3_buf,Out4_buf);
  
      ////////////////////// TEST FOR knoisy ////////////////////////////////
      // sycl::buffer In_theta_buf(In_theta);
      // sycl::buffer In_RNG_buf(In_RNG);
      // sycl::buffer Out_buf(Out);
      // SubmitProduce_test_knoisy<ARProduceKernel_knoisy_ID, ARKnoisy_in_theta_PipeArray, ARKnoisy_in_RNG_PipeArray>(q, In_theta_buf, In_RNG_buf);
      // SubmitConsume_test_knoisy<ARConsumeKernel_knoisy_ID, ARKnoisy_out_PipeArray>(q, Out_buf);
      
      ////////////////////// TEST FOR input+knoisy ////////////////////////////////
      sycl::buffer In_theta_loop_buf(In_theta_loop);
      sycl::buffer In_theta_initial_buf(In_theta_initial);
      sycl::buffer In_end_buf(In_end);
      sycl::buffer In_RNG_buf(In_RNG);
      sycl::buffer Out_buf(Out);
      sycl::buffer Out1_buf(Out1);
      sycl::buffer Out2_buf(Out2);
      sycl::buffer Out3_buf(Out3);
      SubmitProduce_test_input_noisy<ARProduceKernel_input_ID, ARInput_in_theta_initial_PipeArray, ARInput_in_theta_loop_PipeArray, ARInput_in_end_PipeArray, ARKnoisy_in_RNG_PipeArray>(q, In_theta_initial_buf, In_theta_loop_buf, In_end_buf, In_RNG_buf);
      SubmitConsume_test_input_noisy<ARConsumeKernel_knoisy_ID, ARInput_out1_PipeArray, ARInput_out2_PipeArray, ARInput_out3_PipeArray, ARKnoisy_out_PipeArray>(q, Out1_buf, Out2_buf, Out3_buf, Out_buf);
      
    }
    std::cout << "submit finished\n";

    // validate the results
    // operator== for a vector checks sizes, then checks per-element
    // passed &= (out_data == in_data);

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

    // ///////////Print for determine/////////////////
    // std::cout << "output is: \n";
    // for (int i = 0; i < 2; i++)
    // {
    //   std::cout << "iteration: " << i << "\n";
    //   std::cout << Out_end[i] << std::endl;
    //   std::cout << "output_final is: \n";
    //   for (int j = 0; j < 3; j++)
    //   {
    //     for (int k = 0; k < 5; k++)
    //     {
    //       std::cout << Out_final[i * 15 + j * 5 + k] << ' ';
    //     }
    //     std::cout << "\n";
    //   }
    //   std::cout << "output_new is: \n";
    //   for (int w = 0; w < 3; w++)
    //   {
    //     for (int e = 0; e < 5; e++)
    //     {_
    //       std::cout << Out_new[i * 15 + w * 5 + e] << ' ';
    //     }
    //     std::cout << "\n";
    //   }
    // }
    // std::cout << "\n";

        ///////////Print for input/////////////////
    for (int i = 0; i < 2; i++)
    {
      std::cout << "iteration: " << i << "\n";
      for (int j = 0; j < 3; j++)
      {
        for (int k = 0; k < 5; k++)
        {
          std::cout <<"output1" <<Out1[i * 15 + j * 5 + k] << ' ';
          std::cout <<"output2" <<Out2[i * 15 + j * 5 + k] << ' ';
          std::cout <<"output3" <<Out3[i * 15 + j * 5 + k] << ' ';
          //std::cout <<"output4" <<Out4[i * 15 + j * 5 + k] << ' ';
        }
        std::cout << "\n";
      }
    }
    ///////////Print for knoisy/////////////////
    std::cout<<"\n";
    std::cout<<"matrix out: \n";
    for(int i=0; i<12; i++){
      std::cout<< Out[i]<<std::endl;
    }
    std::cout << "\n";
  }
  catch (sycl::exception const &e)
  {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n"
              << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND)
    {
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
  std::cout << "finished \n";

  ///////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////

  return 0;
}
