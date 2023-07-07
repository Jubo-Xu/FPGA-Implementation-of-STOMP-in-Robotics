
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


// declare cost_all
class ARProduceKernel_cost_all_ID;
class ARConsumeKernel_cost_all_ID;
class Costall_Kernel_ID;
class PipeArray_in_cost_all_ID;
class PipeArray_out_cost_all_ID;

// declare cost_single
class ARProduceKernel_cost_single_ID;
class ARConsumeKernel_cost_single_ID;
class Costsingle_Kernel_ID;
class PipeArray_in_cost_single_ID;
class PipeArray_out_cost_single_ID;
class PipeArray_out_theta_cost_single_ID;

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


// pipes for cost_All
using ARCostall_in_PipeArray = fpga_tools::PipeArray<PipeArray_in_cost_all_ID, float,1,12>;
using ARCostall_out_PipeArray = fpga_tools::PipeArray<PipeArray_out_cost_all_ID, float, 5,4>;

// pipes for cost_single
using ARCostsingle_in_PipeArray = fpga_tools::PipeArray<PipeArray_in_cost_single_ID, float,1,3>;
using ARCostsingle_out_PipeArray = sycl::ext::intel::pipe<PipeArray_out_cost_single_ID, float,1>;
using ARCostsingle_out_theta_PipeArray = fpga_tools::PipeArray<PipeArray_out_theta_cost_single_ID, float, 5,3>;

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

// for cost_all
//fpga_tools::Autorun<Costall_Kernel_ID> ar_kernel_parallel_to_serial{selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel<ARCostall_in_PipeArray, ARCostall_out_PipeArray  >{}};

// for cost_single
fpga_tools::Autorun<Costsingle_Kernel_ID> ar_kernel_parallel_to_serial{selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel_Single<ARCostsingle_in_PipeArray, ARCostsingle_out_PipeArray, ARCostsingle_out_theta_PipeArray>{}};

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
  //////////////////////////////////////////////////////////////////////////////////

   ///////////////////////// TEST FOR cost_All//////////////////////////////////
  // Submit
  template <typename KernelID, typename Pipe_in>
  sycl::event SubmitProduce_test_costall(sycl::queue & q,  sycl::buffer<float, 1> & buf_in)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor IN(buf_in, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<12>([&index_1,  &IN, &i](auto idx_1){
          Pipe_in::template PipeAt<idx_1>::write(IN[i*12 + index_1]);
          index_1++;
        });
      }
    }); });
  }

  // Consume
  template <typename KernelID, typename Pipe_out>
  sycl::event SubmitConsume_test_costall(sycl::queue & q, sycl::buffer<float, 1> & out_buf)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor OUT(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        size_t index = 0;
        fpga_tools::UnrolledLoop<4>([&index, &OUT, &i](auto idx){
            OUT[i*3+index] = Pipe_out::template PipeAt<idx>::read();
            index++;
        });
      }
    }); });
  }

   ///////////////////////// TEST FOR cost_single//////////////////////////////////
  // Submit
  template <typename KernelID, typename Pipe_in>
  sycl::event SubmitProduce_test_costsingle(sycl::queue & q,  sycl::buffer<float, 1> & buf_in)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor IN(buf_in, h, read_only);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<3>([&index_1,  &IN, &i](auto idx_1){
          Pipe_in::template PipeAt<idx_1>::write(IN[i*3 + index_1]);
          index_1++;
        });
      }
    }); });
  }

  // Consume
  template <typename KernelID, typename Pipe_out, typename Pipe_out_theta>
  sycl::event SubmitConsume_test_costsingle(sycl::queue & q, sycl::buffer<float, 1> & out_buf, sycl::buffer<float, 1> & out_theta_buf)
  {
    return q.submit([&](sycl::handler &h)
                    {
    sycl::accessor OUT(out_buf, h, write_only, no_init);
    sycl::accessor OUT_theta(out_theta_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<2; i++){
        OUT[i] = Pipe_out::read();
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &OUT_theta, &i](auto idx){
            OUT_theta[i*3+index] = Pipe_out_theta::template PipeAt<idx>::read();
            index++;
        });
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


////////////Test number for cost_all////////////////
// std::vector<float> In = {
//   1.0f, 2.0f, 3.0f,

//   2.0f, 3.0f, 4.0f,
//   1.0f, 2.0f, 3.0f,

//   2.0f, 3.0f, 4.0f
// };
// std::vector<float> Out(2*12);
// std::fill(Out.begin(), Out.end(), -1);

////////////Test number for cost_single////////////////
std::vector<float> In = {
  1.0f, 2.0f, 3.0f,

  2.0f, 3.0f, 4.0f,
};
std::vector<float> Out_theta(2*3);
std::fill(Out.begin(), Out.end(), -1);
std::vector<float> Out = {-1, -1};

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

      ////////////////////// TEST FOR cost_all ////////////////////////////////
      // sycl::buffer In_buf(In);
      // sycl::buffer Out_buf(Out);
      // SubmitProduce_test_costall<ARProduceKernel_cost_all_ID, ARCostall_in_PipeArray>(q, In_buf);
      // SubmitConsume_test_costall<ARConsumeKernel_cost_all_ID, ARCostall_out_PipeArray>(q, Out_buf);

      ////////////////////// TEST FOR cost_single ////////////////////////////////
      sycl::buffer In_buf(In);
      sycl::buffer Out_buf(Out);
      sycl::buffer Out_theta_buf(Out_theta);
      SubmitProduce_test_costsingle<ARProduceKernel_cost_single_ID, ARCostsingle_in_PipeArray>(q, In_buf);
      SubmitConsume_test_costsingle<ARConsumeKernel_cost_single_ID, ARCostsingle_out_PipeArray,ARCostsingle_out_theta_PipeArray>(q, Out_buf,Out_theta_buf);
   
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

    ///////////Print for cost_all/////////////
    // std::cout<<"\n";
    // std::cout<<"Output is: \n";
    // for(int i=0; i<12; i++){
    //   std::cout<< Out[i]<<std::endl;
    // }
    // std::cout << "\n";
   
    ///////////Print for cost_single/////////////
    std::cout<<"\n";
    std::cout<<"matrix out: \n";
    for(int i=0; i<3; i++){
      std::cout<< Out_theta[i]<<std::endl;
    }
    std::cout<<"output is: \n";
    for(int i=0; i<2; i++){
      std::cout<<"iteration: "<<i<<"\n";
      std::cout<<Out[i]<<std::endl;
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
