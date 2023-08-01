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
#include "Determination.hpp"
//#include "RNG.hpp"
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>


// Define some constants
#define N_main          5
#define k_main          4
#define DoF_main        3
#define end_sig_main    2
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
////////////////////// DECLARE THE ID ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////
///////////////// Declare the IDs for kernels ///////////////////////////
// ID for host kernels
class ARProduceKernel_host_ID;
class ARConsumeKernel_host_ID;
// ID for Obstacle cost
class ARObstacle_cost_ID;
class ARObstacle_cost_single_ID;
// ID for Smooth
class ARMMUL_ID;
class ARSparse_MMUL_ID;
class ARTUpdate_ID;
class ARSmooth_calc_ID;
// ID for Delta theta block
class ARDelta_theta_kernel1_ID;
class ARDelta_theta_kernel2_ID;
class ARDelta_theta_kernel3_ID;
// ID for Determination
class ARDetermine_block_ID;
class ARInput_block_ID;
class ARInput_2_kNoisy_ID;
// ID for RNG
class AR_RNG_ID;// maybe more IDs are needed, but for now we only assume one since we haven't finished it yet
////////////////////////////////////////////////////////////////////////
///////////////// Declare the IDs for pipes ////////////////////////////
class PipeArray_host_2_Input_ID;
class PipeArray_Last_2_Input_ID;
class PipeArray_end_sig_ID;
class PipeArray_Input_2_noisy_gen_ID;
class PipeArray_Input_2_Update_ID;
class PipeArray_Input_2_Sparse_ID;
class PipeArray_Input_2_Smooth_ID;
class PipeArray_RNG_out1_ID;
class PipeArray_RNG_out2_to_DT3_ID;
class PipeArray_Noisy_out_ID;
class PipeArray_Obstaclecost_out_ID;
class PipeArray_Delta_theta_kernel1_to_2_ID;
class PipeArray_Delta_theta_kernel2_to_3_ID;
class PipeArray_Delta_theta_kernel_out_2_MMUL_ID;
class PipeArray_Delta_theta_kernel_out_2_smooth_ID;
class PipeArray_MMUL_out1_ID;
class PipeArray_MMUL_out2_innerproduct_ID;
class PipeArray_Update_out_ID;
class PipeArray_Sparse_out_2_smooth_ID;
class PipeArray_smooth_out_2_determine_ID;
class PipeArray_obcost_single_out_cost_ID;
class PipeArray_obcost_single_out_theta_ID;
class PipeArray_Last_2_host_ID;
class PipeArray_Input_ID;
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////// DEFINE THE PIPES ///////////////////////////////////
using pipe_host_2_input = sycl::ext::intel::pipe<PipeArray_host_2_Input_ID, float, N_main>;
using pipearray__input = fpga_tools::PipeArray<PipeArray_Input_ID, float, N_main, DoF_main>;
using pipearray_last_2_input = fpga_tools::PipeArray<PipeArray_Last_2_Input_ID, float, N_main, DoF_main>;
using pipearray_end_sig = fpga_tools::PipeArray<PipeArray_end_sig_ID, int, 1, end_sig_main>;
using pipearray_input_2_noisy = fpga_tools::PipeArray<PipeArray_Input_2_noisy_gen_ID, float, N_main, DoF_main>;
using pipearray_input_2_update = fpga_tools::PipeArray<PipeArray_Input_2_Update_ID, float, N_main, DoF_main>;
using pipearray_input_2_sparse = fpga_tools::PipeArray<PipeArray_Input_2_Sparse_ID, float, N_main, DoF_main>;
using pipearray_input_2_smooth = fpga_tools::PipeArray<PipeArray_Input_2_Smooth_ID, float, N_main, DoF_main>;
//using pipearray_RNG_out1 = fpga_tools::PipeArray<PipeArray_RNG_out1_ID, float, N_main, DoF_main*k_main>;
//using pipearray_RNG_out2_2_DT3 = fpga_tools::PipeArray<PipeArray_RNG_out2_to_DT3_ID, float, N_main, DoF_main*k_main>;
using pipearray_RNG = fpga_tools::PipeArray<PipeArray_RNG_out1_ID, float, N_main, DoF_main*k_main>;
using pipearray_RNG2 = fpga_tools::PipeArray<PipeArray_RNG_out2_to_DT3_ID, float, N_main, DoF_main*k_main>;
using pipearray_noisy_out = fpga_tools::PipeArray<PipeArray_Noisy_out_ID, float, N_main, DoF_main*k_main>;
using pipearray_obscost_out = fpga_tools::PipeArray<PipeArray_Obstaclecost_out_ID, float, N_main, k_main>;
using pipearray_DT_1_2 = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel1_to_2_ID, float, N_main, k_main>;
using pipearray_DT_2_3 = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel2_to_3_ID, float, N_main, k_main>;
using pipearray_DT_out_2_MMUL = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel_out_2_MMUL_ID, float, N_main, DoF_main>;
using pipearray_DT_out_2_smooth = fpga_tools::PipeArray<PipeArray_Delta_theta_kernel_out_2_smooth_ID, float, N_main, DoF_main>;
using pipearray_MMUL_out1 = fpga_tools::PipeArray<PipeArray_MMUL_out1_ID, float, N_main, DoF_main>;
using pipearray_MMUL_out2 = fpga_tools::PipeArray<PipeArray_MMUL_out2_innerproduct_ID, float, N_main, DoF_main>;
using pipearray_update_out = fpga_tools::PipeArray<PipeArray_Update_out_ID, float, N_main, DoF_main>;
using pipearray_sparse_2_smooth = fpga_tools::PipeArray<PipeArray_Sparse_out_2_smooth_ID, float, N_main, DoF_main>;
using pipe_smooth_2_determ = sycl::ext::intel::pipe<PipeArray_smooth_out_2_determine_ID, float, N_main>;
using pipe_obcost_single_cost = sycl::ext::intel::pipe<PipeArray_obcost_single_out_cost_ID, float, N_main>;
using pipearray_obcost_single_theta = fpga_tools::PipeArray<PipeArray_obcost_single_out_theta_ID, float, N_main, DoF_main>;
using pipe_last_2_host = sycl::ext::intel::pipe<PipeArray_Last_2_host_ID, float, N_main>;
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

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
// Define input kernel
fpga_tools::Autorun<ARInput_block_ID> ar_kernel_input(selector, Determine_and_Connections::AR_input_block<pipe_host_2_input, pipearray__input, pipearray_end_sig, pipearray_input_2_noisy, pipearray_input_2_update, pipearray_input_2_sparse, pipearray_input_2_smooth>{});
// Define input to noisy kernel
fpga_tools::Autorun<ARInput_2_kNoisy_ID> ar_kernel_IN2noisy(selector, Determine_and_Connections::AR_Input_2_kNoisy<pipearray_input_2_noisy, pipearray_RNG, pipearray_noisy_out>{});
// Define obstacle cost kernel
fpga_tools::Autorun<ARObstacle_cost_ID> ar_kernel_ObsCost(selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel<pipearray_noisy_out, pipearray_obscost_out>{});
// Define obstacle cost single kernel
fpga_tools::Autorun<ARObstacle_cost_single_ID> ar_kernel_ObsCost_single(selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel_Single<pipearray_update_out, pipe_obcost_single_cost, pipearray_obcost_single_theta>{});
// Define Delta theta kernel 1
fpga_tools::Autorun<ARDelta_theta_kernel1_ID> ar_kernel_DT1(selector, Delta_Theta_Block::Theta_Calc_Kernel1<pipearray_obscost_out, pipearray_DT_1_2>{});
// Define Delta theta kernel 2
fpga_tools::Autorun<ARDelta_theta_kernel2_ID> ar_kernel_DT2(selector, Delta_Theta_Block::Theta_Calc_Kernel2<pipearray_DT_1_2, pipearray_DT_2_3>{});
// Define Delta theta kernel 3
fpga_tools::Autorun<ARDelta_theta_kernel3_ID> ar_kernel_DT3(selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipearray_DT_2_3, pipearray_RNG2, pipearray_DT_out_2_MMUL, pipearray_DT_out_2_smooth>{});
// Define MMUL kernel
fpga_tools::Autorun<ARMMUL_ID> ar_kernel_MMUL(selector, smooth::ARMMul<pipearray_DT_out_2_MMUL, pipearray_MMUL_out1, pipearray_MMUL_out2>{});
// Define Update kernel
fpga_tools::Autorun<ARTUpdate_ID> ar_kernel_Update(selector, smooth::ARTUpdate<pipearray_input_2_update, pipearray_MMUL_out1, pipearray_update_out>{});
// Define Sparse MMUL kernel
fpga_tools::Autorun<ARSparse_MMUL_ID> ar_kernel_Sparse(selector, smooth::ARSparseMul<pipearray_input_2_sparse, pipearray_sparse_2_smooth>{});
// Define Smooth cost kernel
fpga_tools::Autorun<ARSmooth_calc_ID> ar_kernel_Smooth(selector, smooth::ARSmooth<pipearray_end_sig, pipearray_MMUL_out2, pipearray_sparse_2_smooth, pipearray_input_2_smooth, pipearray_DT_out_2_smooth, pipe_smooth_2_determ>{});
// Define Determine block kernel
fpga_tools::Autorun<ARDetermine_block_ID> ar_kernel_determine(selector, Determine_and_Connections::AR_Determine_Block<pipe_obcost_single_cost, pipe_smooth_2_determ, pipearray_obcost_single_theta, pipearray_last_2_input, pipe_last_2_host, pipearray_end_sig>{});
// Define RNG kernel
//fpga_tools::Autorun<AR_RNG_ID> ar_kernel_RNG(selector, RNG::AR_RNG<pipearray_RNG_out1, pipearray_RNG_out2_2_DT3>{});
////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////

// Submit a kernel to read data from global memory and write to a pipe
//
template <typename KernelID, typename Pipe, typename Pipe_theta_loop, typename Pipe_RNG,typename Pipe_RNG2 >
sycl::event SubmitProducerKernel(sycl::queue& q, sycl::buffer<float, 1>& in_buf, sycl::buffer<float, 1>& in_theta_loop_buf, sycl::buffer<float, 1> & buf_in_RNG, sycl::buffer<float, 1> & buf_in_RNG2) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor in(in_buf, h, read_only);
    sycl::accessor IN_RNG(buf_in_RNG, h, read_only);
    sycl::accessor IN_RNG2(buf_in_RNG2, h, read_only);
    sycl::accessor IN_theta_loop(in_theta_loop_buf, h, read_only);
    int size = in_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < 5; i++) {
        size_t index_2 = 0;
        fpga_tools::UnrolledLoop<12>([&index_2, &IN_RNG2, &i](auto k) {
              Pipe_RNG2::template PipeAt<k>::write(IN_RNG2[1]);
              index_2++;
        });
        size_t index_1 = 0;
        fpga_tools::UnrolledLoop<12>([&index_1,  &IN_RNG, &i](auto idx_1){
          Pipe_RNG::template PipeAt<idx_1>::write(IN_RNG[1]);
          index_1++;
        });
        size_t index = 0;
        fpga_tools::UnrolledLoop<3>([&index, &IN_theta_loop, &i](auto idx_1) {
                    //val[index_in_val++] = PipeIn::template PipeAt<idx_1>::read();
              Pipe_theta_loop::template PipeAt<idx_1>::write(IN_theta_loop[2]);
              index++;
        });
        for(int j=0; j<3; j++){
          Pipe::write(in[j*3+i]);
        }
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
    int size = out_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
          out[i] = Pipe::read();
      }
    });
  });
}





int main(){

std::cout<<"un peu d'espoir\n";


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// TEST FOR AUTORUN //////////////////////////////////

// Number of total numbers of input
int count = DoF_main*N_main;
//bool passed = true;

std::vector<float> in_data(count), out_data(count);
// Clear the output buffer
std::fill(out_data.begin(), out_data.end(), -1);

//in_data = {4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 3.0f};
std::vector<float> In_data = {
  1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
  2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
  4.3f, 4.4f, 2.1f, 4.2f, 5.5f
};
std::vector<float> In_theta_loop = {
  1.2f, 2.1f, 3.1f, 4.5f, 5.4f,
  2.5f, 3.5f, 3.2f, 1.5f, 5.6f,
  4.3f, 4.4f, 2.1f, 4.2f, 5.5f,

  1.0f, 4.3f, 3.9f, 4.4f, 3.3f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
  1.0f, 2.0f, 3.0f, 4.0f, 5.0f
};
std::vector<float> In_RNG = {
  1.0f, 2.0f, 3.0f,

  2.0f, 3.0f, 4.0f,
  1.0f, 2.0f, 3.0f,

  2.0f, 3.0f, 4.0f
};
// Initialize input date ---> can be connected with ROS later to get the value

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
      sycl::buffer in_theta_loop_buf(In_theta_loop);
      sycl::buffer in_RNG_buf(In_RNG);
      sycl::buffer out_buf(out_data);
      // submit data to kernel
      SubmitProducerKernel<ARProduceKernel_host_ID, pipe_host_2_input, pipearray__input,pipearray_RNG, pipearray_RNG2>(q, in_buf,in_theta_loop_buf, in_RNG_buf, in_RNG_buf);
      // get data from kernel
      SubmitConsumerKernel<ARConsumeKernel_host_ID, pipe_last_2_host>(q, out_buf);
      

    }
    std::cout<<"submit finished\n";
    // Print the OUTPUT

    std::cout << "output_final is: \n";

    for (int i = 0; i < count; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        for (int k = 0; k < 5; k++)
        {
          std::cout << out_data[i * 15 + j * 5 + k] << ' ';
        }
        std::cout << "\n";
      }
    }
 

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
  std::cout<<"finished \n";

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


return 0;

}