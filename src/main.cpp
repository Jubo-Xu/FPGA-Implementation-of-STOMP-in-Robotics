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
#include "RNG.hpp"
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>


// Define some constants
#define N_main          16
#define k_main          4
#define DoF_main        3
#define end_sig_main    2
#define N_rng           2000
#define N_stat          32



// PRINTF("Hello, World!\n");
// PRINTF("Hello: %d\n", 123);
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
class AR_uni_rng_r32_1_ID;
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
class PipeArray_delta_rng1_ID;
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////// DEFINE THE PIPES ///////////////////////////////////
using pipe_host_2_input = sycl::ext::intel::pipe<PipeArray_host_2_Input_ID, float, N_main>;
using pipearray_last_2_input = fpga_tools::PipeArray<PipeArray_Last_2_Input_ID, float, N_main, DoF_main>;
using pipearray_end_sig = fpga_tools::PipeArray<PipeArray_end_sig_ID, int, 1, end_sig_main>;
using pipearray_input_2_noisy = fpga_tools::PipeArray<PipeArray_Input_2_noisy_gen_ID, float, N_main, DoF_main>;
using pipearray_input_2_update = fpga_tools::PipeArray<PipeArray_Input_2_Update_ID, float, N_main, DoF_main>;
using pipearray_input_2_sparse = fpga_tools::PipeArray<PipeArray_Input_2_Sparse_ID, float, N_main, DoF_main>;
using pipearray_input_2_smooth = fpga_tools::PipeArray<PipeArray_Input_2_Smooth_ID, float, N_main, DoF_main>;
using pipearray_RNG_out1 = fpga_tools::PipeArray<PipeArray_RNG_out1_ID, float, N_main, DoF_main*k_main>;
using pipearray_RNG_out2_2_DT3 = fpga_tools::PipeArray<PipeArray_RNG_out2_to_DT3_ID, float, N_main, DoF_main*k_main>;
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
using pipe_delta_rng1 = sycl::ext::intel::pipe<PipeArray_delta_rng1_ID, float, 10>;
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
fpga_tools::Autorun<ARInput_block_ID> ar_kernel_input(selector, Determine_and_Connections::AR_input_block<pipe_host_2_input, pipearray_last_2_input, pipearray_end_sig, pipearray_input_2_noisy, pipearray_input_2_update, pipearray_input_2_sparse, pipearray_input_2_smooth>{});
// Define input to noisy kernel
fpga_tools::Autorun<ARInput_2_kNoisy_ID> ar_kernel_IN2noisy(selector, Determine_and_Connections::AR_Input_2_kNoisy<pipearray_input_2_noisy, pipearray_RNG_out1, pipearray_noisy_out>{});
// Define obstacle cost kernel
fpga_tools::Autorun<ARObstacle_cost_ID> ar_kernel_ObsCost(selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel<pipearray_noisy_out, pipearray_obscost_out>{});
// Define obstacle cost single kernel
fpga_tools::Autorun<ARObstacle_cost_single_ID> ar_kernel_ObsCost_single(selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel_Single<pipearray_update_out, pipe_obcost_single_cost, pipearray_obcost_single_theta>{});
// Define Delta theta kernel 1
fpga_tools::Autorun<ARDelta_theta_kernel1_ID> ar_kernel_DT1(selector, Delta_Theta_Block::Theta_Calc_Kernel1<pipearray_obscost_out, pipearray_DT_1_2>{});
// Define Delta theta kernel 2
fpga_tools::Autorun<ARDelta_theta_kernel2_ID> ar_kernel_DT2(selector, Delta_Theta_Block::Theta_Calc_Kernel2<pipearray_DT_1_2, pipearray_DT_2_3>{});
// Define Delta theta kernel 3
fpga_tools::Autorun<ARDelta_theta_kernel3_ID> ar_kernel_DT3(selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipearray_DT_2_3, pipearray_RNG_out2_2_DT3, pipearray_DT_out_2_MMUL, pipearray_DT_out_2_smooth>{});
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
fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng1, 0, 1, 8>{});
fpga_tools::Autorun<AR_RNG_ID> ar_kernel_RNG(selector, RNG::AR_RNG_final<pipe_delta_rng1, pipearray_RNG_out1, pipearray_RNG_out2_2_DT3>{});
////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
////////////////////////// TEST FOR RNG //////////////////////////////
// Declare the IDs
// class RNG_kernel_test_ID;
// class RNG_consume_kernel_test_ID;
// class Pipe_RNG_test_ID;

// using myint = ac_int<1, false>;
// using myint4 = ac_int<4, true>;
// using fixed_3_2 = ac_fixed<6, 4, true>;

// // test for uniform RNG 6 bit
// class uniform_rng_6_test_ID;
// class uniform_rng_6_consume_ID;
// class Pipe_uniform_rng_test_ID;
// using pipe_uniform_rng_6_test = sycl::ext::intel::pipe<Pipe_uniform_rng_test_ID, ac_int<5, false>, 2>;

// //using pipe_rng_test = sycl::ext::intel::pipe<Pipe_RNG_test_ID, myint, 1>;
// using pipe_rng_test = sycl::ext::intel::pipe<Pipe_RNG_test_ID, float, 2>;

// //Declare the IDs for RNG kernels
// class AR_Gaussian_rng_kernel_ID;
// class AR_alias_table_ID;
// class AR_uni_rng_r32_single_ID;
// class AR_uni_rng_r6_ID;
// class AR_tri_rng_ID;
// //IDs for multiple rngs test
// class AR_uni_rng_r32_1_ID;
// class AR_uni_rng_r32_2_ID;
// class AR_uni_rng_r32_3_ID;
// class AR_uni_rng_r32_4_ID;
// class AR_uni_rng_r32_5_ID;
// class AR_uni_rng_r32_6_ID;
// class AR_uni_rng_r32_7_ID;
// class AR_uni_rng_r32_8_ID;
// class AR_uni_rng_r32_9_ID;
// class AR_uni_rng_r32_10_ID;
// class AR_uni_rng_r32_11_ID;
// class AR_uni_rng_r32_12_ID;
// class AR_uni_rng_r32_13_ID;
// class AR_uni_rng_r32_14_ID;
// class AR_uni_rng_r32_15_ID;
// class AR_uni_rng_r32_16_ID;
// class AR_uni_rng_r32_17_ID;
// class AR_uni_rng_r32_18_ID;
// class AR_uni_rng_r32_19_ID;
// class AR_uni_rng_r32_20_ID;
// class AR_uni_rng_r32_21_ID;
// class AR_uni_rng_r32_22_ID;
// class AR_uni_rng_r32_23_ID;
// class AR_uni_rng_r32_24_ID;
// class AR_uni_rng_r32_25_ID;
// class AR_uni_rng_r32_26_ID;
// class AR_uni_rng_r32_27_ID;
// class AR_uni_rng_r32_28_ID;
// class AR_uni_rng_r32_29_ID;
// class AR_uni_rng_r32_30_ID;
// class AR_uni_rng_r32_31_ID;
// class AR_uni_rng_r32_32_ID;
// class AR_uni_rng_r32_33_ID;
// class AR_uni_rng_r32_34_ID;
// class AR_uni_rng_r32_35_ID;
// class AR_uni_rng_r32_36_ID;
// class AR_uni_rng_r32_37_ID;
// class AR_uni_rng_r32_38_ID;
// class AR_uni_rng_r32_39_ID;
// class AR_uni_rng_r32_40_ID;
// class AR_uni_rng_r32_41_ID;
// class AR_uni_rng_r32_42_ID;
// class AR_uni_rng_r32_43_ID;
// class AR_uni_rng_r32_44_ID;
// class AR_uni_rng_r32_45_ID;
// class AR_uni_rng_r32_46_ID;
// class AR_uni_rng_r32_47_ID;
// class AR_uni_rng_r32_48_ID;
// class AR_uni_rng_r32_49_ID;
// class AR_uni_rng_r32_50_ID;
// class AR_uni_rng_r32_51_ID;
// class AR_uni_rng_r32_52_ID;
// class AR_uni_rng_r32_53_ID;
// class AR_uni_rng_r32_54_ID;
// class AR_uni_rng_r32_55_ID;
// class AR_uni_rng_r32_56_ID;
// class AR_uni_rng_r32_57_ID;
// class AR_uni_rng_r32_58_ID;
// class AR_uni_rng_r32_59_ID;
// class AR_uni_rng_r32_60_ID;
// class AR_uni_rng_r32_61_ID;
// class AR_uni_rng_r32_62_ID;
// class AR_uni_rng_r32_63_ID;
// class AR_uni_rng_r32_64_ID;
// class AR_hadmard_ID;


// //Declare the IDs for pipes
// class Pipe_uni_rng_r32_ID;
// class Pipe_uni_rng_r6_ID;
// class Pipe_alias_2_grng_ID;
// class PipeArray_tri_rng_2_grng_ID;
// class Pipe_grng_out_ID;
// //IDs for pipe of delta rngs
// class PipeArray_delta_rng1_ID;
// class PipeArray_delta_rng2_ID;
// class PipeArray_delta_rng3_ID;
// class PipeArray_delta_rng4_ID;
// class PipeArray_delta_rng5_ID;
// class PipeArray_delta_rng6_ID;
// class PipeArray_delta_rng7_ID;
// class PipeArray_delta_rng8_ID;
// class PipeArray_delta_rng9_ID;
// class PipeArray_delta_rng10_ID;
// class PipeArray_delta_rng11_ID;
// class PipeArray_delta_rng12_ID;
// class PipeArray_delta_rng13_ID;
// class PipeArray_delta_rng14_ID;
// class PipeArray_delta_rng15_ID;
// class PipeArray_delta_rng16_ID;
// class PipeArray_delta_rng17_ID;
// class PipeArray_delta_rng18_ID;
// class PipeArray_delta_rng19_ID;
// class PipeArray_delta_rng20_ID;
// class PipeArray_delta_rng21_ID;
// class PipeArray_delta_rng22_ID;
// class PipeArray_delta_rng23_ID;
// class PipeArray_delta_rng24_ID;
// class PipeArray_delta_rng25_ID;
// class PipeArray_delta_rng26_ID;
// class PipeArray_delta_rng27_ID;
// class PipeArray_delta_rng28_ID;
// class PipeArray_delta_rng29_ID;
// class PipeArray_delta_rng30_ID;
// class PipeArray_delta_rng31_ID;
// class PipeArray_delta_rng32_ID;
// class PipeArray_Hadmard_ID;
// //Define the Pipes
// using pipe_uni_rng_r32 = sycl::ext::intel::pipe<Pipe_uni_rng_r32_ID, float, 2>;
// using pipe_uni_rng_r6 = sycl::ext::intel::pipe<Pipe_uni_rng_r6_ID, RNG::int_6_bit, 2>;
// using pipe_alias_2_grng = sycl::ext::intel::pipe<Pipe_alias_2_grng_ID, RNG::int_5_bit, 2>;
// using pipearray_tri_rng_2_grng = fpga_tools::PipeArray<Pipe_grng_out_ID, float, 10, 16>;
// using pipe_grng_out = sycl::ext::intel::pipe<Pipe_grng_out_ID, float, 2>;
// //using pipearray_delta_rng = fpga_tools::PipeArray<PipeArray_delta_rng_ID, float, 5, 64>;
// using pipe_delta_rng1 = sycl::ext::intel::pipe<PipeArray_delta_rng1_ID, float, 10>;
// using pipe_delta_rng2 = sycl::ext::intel::pipe<PipeArray_delta_rng2_ID, float, 10>;
// using pipe_delta_rng3 = sycl::ext::intel::pipe<PipeArray_delta_rng3_ID, float, 10>;
// using pipe_delta_rng4 = sycl::ext::intel::pipe<PipeArray_delta_rng4_ID, float, 10>;
// using pipe_delta_rng5 = sycl::ext::intel::pipe<PipeArray_delta_rng5_ID, float, 10>;
// using pipe_delta_rng6 = sycl::ext::intel::pipe<PipeArray_delta_rng6_ID, float, 10>;
// using pipe_delta_rng7 = sycl::ext::intel::pipe<PipeArray_delta_rng7_ID, float, 10>;
// using pipe_delta_rng8 = sycl::ext::intel::pipe<PipeArray_delta_rng8_ID, float, 10>;
// using pipe_delta_rng9 = sycl::ext::intel::pipe<PipeArray_delta_rng9_ID, float, 10>;
// using pipe_delta_rng10 = sycl::ext::intel::pipe<PipeArray_delta_rng10_ID, float, 10>;
// using pipe_delta_rng11 = sycl::ext::intel::pipe<PipeArray_delta_rng11_ID, float, 10>;
// using pipe_delta_rng12 = sycl::ext::intel::pipe<PipeArray_delta_rng12_ID, float, 10>;
// using pipe_delta_rng13 = sycl::ext::intel::pipe<PipeArray_delta_rng13_ID, float, 10>;
// using pipe_delta_rng14 = sycl::ext::intel::pipe<PipeArray_delta_rng14_ID, float, 10>;
// using pipe_delta_rng15 = sycl::ext::intel::pipe<PipeArray_delta_rng15_ID, float, 10>;
// using pipe_delta_rng16 = sycl::ext::intel::pipe<PipeArray_delta_rng16_ID, float, 10>;
// using pipe_delta_rng17 = sycl::ext::intel::pipe<PipeArray_delta_rng17_ID, float, 10>;
// using pipe_delta_rng18 = sycl::ext::intel::pipe<PipeArray_delta_rng18_ID, float, 10>;
// using pipe_delta_rng19 = sycl::ext::intel::pipe<PipeArray_delta_rng19_ID, float, 10>;
// using pipe_delta_rng20 = sycl::ext::intel::pipe<PipeArray_delta_rng20_ID, float, 10>;
// using pipe_delta_rng21 = sycl::ext::intel::pipe<PipeArray_delta_rng21_ID, float, 10>;
// using pipe_delta_rng22 = sycl::ext::intel::pipe<PipeArray_delta_rng22_ID, float, 10>;
// using pipe_delta_rng23 = sycl::ext::intel::pipe<PipeArray_delta_rng23_ID, float, 10>;
// using pipe_delta_rng24 = sycl::ext::intel::pipe<PipeArray_delta_rng24_ID, float, 10>;
// using pipe_delta_rng25 = sycl::ext::intel::pipe<PipeArray_delta_rng25_ID, float, 10>;
// using pipe_delta_rng26 = sycl::ext::intel::pipe<PipeArray_delta_rng26_ID, float, 10>;
// using pipe_delta_rng27 = sycl::ext::intel::pipe<PipeArray_delta_rng27_ID, float, 10>;
// using pipe_delta_rng28 = sycl::ext::intel::pipe<PipeArray_delta_rng28_ID, float, 10>;
// using pipe_delta_rng29 = sycl::ext::intel::pipe<PipeArray_delta_rng29_ID, float, 10>;
// using pipe_delta_rng30 = sycl::ext::intel::pipe<PipeArray_delta_rng30_ID, float, 10>;
// using pipe_delta_rng31 = sycl::ext::intel::pipe<PipeArray_delta_rng31_ID, float, 10>;
// using pipe_delta_rng32 = sycl::ext::intel::pipe<PipeArray_delta_rng32_ID, float, 10>;
// using pipe_hadmard = sycl::ext::intel::pipe<PipeArray_Hadmard_ID, float, 10>;





// template <typename Pipe_out>
// struct RNG_test{
//   void operator()() const{
//     [[intel::fpga_register]] myint BUF[6];
//     #pragma unroll
//     for(int i=1; i<6; i++){
//       BUF[i] = 0;
//     }
//     BUF[0] = 1;
//     while(1){
//       // #pragma unroll
//       // for(int i=1; i<6; i++){
//       //   BUF[6-i] = ext::intel::fpga_reg(BUF[6-i-1]);
//       // }
//       RNG::FIFO_shift<myint, 6>(BUF);
//       Pipe_out::write(BUF[4]);
//     }
//   }   
// };

//fpga_tools::Autorun<RNG_kernel_test_ID> ar_kernel_rng_test(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_grng_out, RNG::fixed_point_4_28_unsigned>{});
//fpga_tools::Autorun<uniform_rng_6_test_ID> ar_kernel_uniform_rng_6_test(selector, RNG::AR_LUT_OP_r6_rng<pipe_uniform_rng_6_test>{});

//test for trianglular distribution
//fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_grng_out, 12, 13, 8>{});
//Define the kernels for RNG
// fpga_tools::Autorun<AR_uni_rng_r32_single_ID> ar_kernel_unirng_r32_single(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_uni_rng_r32, RNG::fixed_point_4_28_unsigned>{});
// fpga_tools::Autorun<AR_uni_rng_r6_ID> ar_kernel_unirng_r6(selector, RNG::AR_LUT_OP_r6_rng<pipe_uni_rng_r6>{});
// fpga_tools::Autorun<AR_alias_table_ID> ar_kernel_aliastable(selector, RNG::AR_Walker_Alias_Table<pipe_uni_rng_r32, pipe_uni_rng_r6, pipe_alias_2_grng>{});
// fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd<pipearray_tri_rng_2_grng>{});
// fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_grng_out>{});
//Define the kernels for RNG_2nd
// fpga_tools::Autorun<AR_uni_rng_r32_single_ID> ar_kernel_unirng_r32_single(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_uni_rng_r32, RNG::fixed_point_4_28_unsigned>{});
// fpga_tools::Autorun<AR_uni_rng_r6_ID> ar_kernel_unirng_r6(selector, RNG::AR_LUT_OP_r6_rng<pipe_uni_rng_r6>{});
// fpga_tools::Autorun<AR_alias_table_ID> ar_kernel_aliastable(selector, RNG::AR_Walker_Alias_Table<pipe_uni_rng_r32, pipe_uni_rng_r6, pipe_alias_2_grng>{});
// fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 0, 0>{});
// fpga_tools::Autorun<AR_uni_rng_r32_2_ID> ar_kernel_delta_rng_2(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 1, 1>{});
// fpga_tools::Autorun<AR_uni_rng_r32_3_ID> ar_kernel_delta_rng_3(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 2, 2>{});
// fpga_tools::Autorun<AR_uni_rng_r32_4_ID> ar_kernel_delta_rng_4(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 3, 3>{});
// fpga_tools::Autorun<AR_uni_rng_r32_5_ID> ar_kernel_delta_rng_5(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 4, 4>{});
// fpga_tools::Autorun<AR_uni_rng_r32_6_ID> ar_kernel_delta_rng_6(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 5, 5>{});
// fpga_tools::Autorun<AR_uni_rng_r32_7_ID> ar_kernel_delta_rng_7(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 6, 6>{});
// fpga_tools::Autorun<AR_uni_rng_r32_8_ID> ar_kernel_delta_rng_8(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 7, 7>{});
// fpga_tools::Autorun<AR_uni_rng_r32_9_ID> ar_kernel_delta_rng_9(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 8, 8>{});
// fpga_tools::Autorun<AR_uni_rng_r32_10_ID> ar_kernel_delta_rng_10(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 9, 9>{});
// fpga_tools::Autorun<AR_uni_rng_r32_11_ID> ar_kernel_delta_rng_11(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 10, 10>{});
// fpga_tools::Autorun<AR_uni_rng_r32_12_ID> ar_kernel_delta_rng_12(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 11, 11>{});
// fpga_tools::Autorun<AR_uni_rng_r32_13_ID> ar_kernel_delta_rng_13(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 12, 12>{});
// fpga_tools::Autorun<AR_uni_rng_r32_14_ID> ar_kernel_delta_rng_14(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 13, 13>{});
// fpga_tools::Autorun<AR_uni_rng_r32_15_ID> ar_kernel_delta_rng_15(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 14, 14>{});
// fpga_tools::Autorun<AR_uni_rng_r32_16_ID> ar_kernel_delta_rng_16(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 15, 15>{});
// fpga_tools::Autorun<AR_uni_rng_r32_17_ID> ar_kernel_delta_rng_17(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 16, 16>{});
// fpga_tools::Autorun<AR_uni_rng_r32_18_ID> ar_kernel_delta_rng_18(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 17, 17>{});
// fpga_tools::Autorun<AR_uni_rng_r32_19_ID> ar_kernel_delta_rng_19(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 18, 18>{});
// fpga_tools::Autorun<AR_uni_rng_r32_20_ID> ar_kernel_delta_rng_20(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 19, 19>{});
// fpga_tools::Autorun<AR_uni_rng_r32_21_ID> ar_kernel_delta_rng_21(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 20, 20>{});
// fpga_tools::Autorun<AR_uni_rng_r32_22_ID> ar_kernel_delta_rng_22(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 21, 21>{});
// fpga_tools::Autorun<AR_uni_rng_r32_23_ID> ar_kernel_delta_rng_23(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 22, 22>{});
// fpga_tools::Autorun<AR_uni_rng_r32_24_ID> ar_kernel_delta_rng_24(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 23, 23>{});
// fpga_tools::Autorun<AR_uni_rng_r32_25_ID> ar_kernel_delta_rng_25(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 24, 24>{});
// fpga_tools::Autorun<AR_uni_rng_r32_26_ID> ar_kernel_delta_rng_26(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 25, 25>{});
// fpga_tools::Autorun<AR_uni_rng_r32_27_ID> ar_kernel_delta_rng_27(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 26, 26>{});
// fpga_tools::Autorun<AR_uni_rng_r32_28_ID> ar_kernel_delta_rng_28(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 27, 27>{});
// fpga_tools::Autorun<AR_uni_rng_r32_29_ID> ar_kernel_delta_rng_29(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 28, 28>{});
// fpga_tools::Autorun<AR_uni_rng_r32_30_ID> ar_kernel_delta_rng_30(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 29, 29>{});
// fpga_tools::Autorun<AR_uni_rng_r32_31_ID> ar_kernel_delta_rng_31(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 30, 30>{});
// fpga_tools::Autorun<AR_uni_rng_r32_32_ID> ar_kernel_delta_rng_32(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 31, 31>{});
// fpga_tools::Autorun<AR_uni_rng_r32_33_ID> ar_kernel_delta_rng_33(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 32, 32>{});
// fpga_tools::Autorun<AR_uni_rng_r32_34_ID> ar_kernel_delta_rng_34(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 33, 33>{});
// fpga_tools::Autorun<AR_uni_rng_r32_35_ID> ar_kernel_delta_rng_35(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 34, 34>{});
// fpga_tools::Autorun<AR_uni_rng_r32_36_ID> ar_kernel_delta_rng_36(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 35, 35>{});
// fpga_tools::Autorun<AR_uni_rng_r32_37_ID> ar_kernel_delta_rng_37(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 36, 36>{});
// fpga_tools::Autorun<AR_uni_rng_r32_38_ID> ar_kernel_delta_rng_38(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 37, 37>{});
// fpga_tools::Autorun<AR_uni_rng_r32_39_ID> ar_kernel_delta_rng_39(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 38, 38>{});
// fpga_tools::Autorun<AR_uni_rng_r32_40_ID> ar_kernel_delta_rng_40(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 39, 39>{});
// fpga_tools::Autorun<AR_uni_rng_r32_41_ID> ar_kernel_delta_rng_41(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 40, 40>{});
// fpga_tools::Autorun<AR_uni_rng_r32_42_ID> ar_kernel_delta_rng_42(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 41, 41>{});
// fpga_tools::Autorun<AR_uni_rng_r32_43_ID> ar_kernel_delta_rng_43(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 42, 42>{});
// fpga_tools::Autorun<AR_uni_rng_r32_44_ID> ar_kernel_delta_rng_44(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 43, 43>{});
// fpga_tools::Autorun<AR_uni_rng_r32_45_ID> ar_kernel_delta_rng_45(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 44, 44>{});
// fpga_tools::Autorun<AR_uni_rng_r32_46_ID> ar_kernel_delta_rng_46(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 45, 45>{});
// fpga_tools::Autorun<AR_uni_rng_r32_47_ID> ar_kernel_delta_rng_47(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 46, 46>{});
// fpga_tools::Autorun<AR_uni_rng_r32_48_ID> ar_kernel_delta_rng_48(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 47, 47>{});
// fpga_tools::Autorun<AR_uni_rng_r32_49_ID> ar_kernel_delta_rng_49(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 48, 48>{});
// fpga_tools::Autorun<AR_uni_rng_r32_50_ID> ar_kernel_delta_rng_50(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 49, 49>{});
// fpga_tools::Autorun<AR_uni_rng_r32_51_ID> ar_kernel_delta_rng_51(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 50, 50>{});
// fpga_tools::Autorun<AR_uni_rng_r32_52_ID> ar_kernel_delta_rng_52(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 51, 51>{});
// fpga_tools::Autorun<AR_uni_rng_r32_53_ID> ar_kernel_delta_rng_53(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 52, 52>{});
// fpga_tools::Autorun<AR_uni_rng_r32_54_ID> ar_kernel_delta_rng_54(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 53, 53>{});
// fpga_tools::Autorun<AR_uni_rng_r32_55_ID> ar_kernel_delta_rng_55(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 54, 54>{});
// fpga_tools::Autorun<AR_uni_rng_r32_56_ID> ar_kernel_delta_rng_56(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 55, 55>{});
// fpga_tools::Autorun<AR_uni_rng_r32_57_ID> ar_kernel_delta_rng_57(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 56, 56>{});
// fpga_tools::Autorun<AR_uni_rng_r32_58_ID> ar_kernel_delta_rng_58(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 57, 57>{});
// fpga_tools::Autorun<AR_uni_rng_r32_59_ID> ar_kernel_delta_rng_59(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 58, 58>{});
// fpga_tools::Autorun<AR_uni_rng_r32_60_ID> ar_kernel_delta_rng_60(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 59, 59>{});
// fpga_tools::Autorun<AR_uni_rng_r32_61_ID> ar_kernel_delta_rng_61(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 60, 60>{});
// fpga_tools::Autorun<AR_uni_rng_r32_62_ID> ar_kernel_delta_rng_62(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 61, 61>{});
// fpga_tools::Autorun<AR_uni_rng_r32_63_ID> ar_kernel_delta_rng_63(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 62, 62>{});
// fpga_tools::Autorun<AR_uni_rng_r32_64_ID> ar_kernel_delta_rng_64(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 63, 63>{});

// fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd<pipearray_delta_rng, pipearray_tri_rng_2_grng>{});
// fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_grng_out>{});

//Define the kernels for RNG_third
// fpga_tools::Autorun<AR_uni_rng_r32_single_ID> ar_kernel_unirng_r32_single(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_uni_rng_r32, RNG::fixed_point_4_28_unsigned>{});
// fpga_tools::Autorun<AR_uni_rng_r6_ID> ar_kernel_unirng_r6(selector, RNG::AR_LUT_OP_r6_rng<pipe_uni_rng_r6>{});
// fpga_tools::Autorun<AR_alias_table_ID> ar_kernel_aliastable(selector, RNG::AR_Walker_Alias_Table<pipe_uni_rng_r32, pipe_uni_rng_r6, pipe_alias_2_grng>{});
// fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng1, 0>{});
// fpga_tools::Autorun<AR_uni_rng_r32_2_ID> ar_kernel_delta_rng_2(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng2, 1>{});
// fpga_tools::Autorun<AR_uni_rng_r32_3_ID> ar_kernel_delta_rng_3(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng3, 2>{});
// fpga_tools::Autorun<AR_uni_rng_r32_4_ID> ar_kernel_delta_rng_4(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng4, 3>{});
// fpga_tools::Autorun<AR_uni_rng_r32_5_ID> ar_kernel_delta_rng_5(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng5, 4>{});
// fpga_tools::Autorun<AR_uni_rng_r32_6_ID> ar_kernel_delta_rng_6(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng6, 5>{});
// fpga_tools::Autorun<AR_uni_rng_r32_7_ID> ar_kernel_delta_rng_7(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng7, 6>{});
// fpga_tools::Autorun<AR_uni_rng_r32_8_ID> ar_kernel_delta_rng_8(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng8, 7>{});
// fpga_tools::Autorun<AR_uni_rng_r32_9_ID> ar_kernel_delta_rng_9(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng9, 8>{});
// fpga_tools::Autorun<AR_uni_rng_r32_10_ID> ar_kernel_delta_rng_10(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng10, 9>{});
// fpga_tools::Autorun<AR_uni_rng_r32_11_ID> ar_kernel_delta_rng_11(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng11, 10>{});
// fpga_tools::Autorun<AR_uni_rng_r32_12_ID> ar_kernel_delta_rng_12(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng12, 11>{});
// fpga_tools::Autorun<AR_uni_rng_r32_13_ID> ar_kernel_delta_rng_13(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng13, 12>{});
// fpga_tools::Autorun<AR_uni_rng_r32_14_ID> ar_kernel_delta_rng_14(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng14, 13>{});
// fpga_tools::Autorun<AR_uni_rng_r32_15_ID> ar_kernel_delta_rng_15(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng15, 14>{});
// fpga_tools::Autorun<AR_uni_rng_r32_16_ID> ar_kernel_delta_rng_16(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng16, 15>{});
// fpga_tools::Autorun<AR_uni_rng_r32_17_ID> ar_kernel_delta_rng_17(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng17, 16>{});
// fpga_tools::Autorun<AR_uni_rng_r32_18_ID> ar_kernel_delta_rng_18(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng18, 17>{});
// fpga_tools::Autorun<AR_uni_rng_r32_19_ID> ar_kernel_delta_rng_19(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng19, 18>{});
// fpga_tools::Autorun<AR_uni_rng_r32_20_ID> ar_kernel_delta_rng_20(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng20, 19>{});
// fpga_tools::Autorun<AR_uni_rng_r32_21_ID> ar_kernel_delta_rng_21(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng21, 20>{});
// fpga_tools::Autorun<AR_uni_rng_r32_22_ID> ar_kernel_delta_rng_22(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng22, 21>{});
// fpga_tools::Autorun<AR_uni_rng_r32_23_ID> ar_kernel_delta_rng_23(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng23, 22>{});
// fpga_tools::Autorun<AR_uni_rng_r32_24_ID> ar_kernel_delta_rng_24(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng24, 23>{});
// fpga_tools::Autorun<AR_uni_rng_r32_25_ID> ar_kernel_delta_rng_25(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng25, 24>{});
// fpga_tools::Autorun<AR_uni_rng_r32_26_ID> ar_kernel_delta_rng_26(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng26, 25>{});
// fpga_tools::Autorun<AR_uni_rng_r32_27_ID> ar_kernel_delta_rng_27(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng27, 26>{});
// fpga_tools::Autorun<AR_uni_rng_r32_28_ID> ar_kernel_delta_rng_28(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng28, 27>{});
// fpga_tools::Autorun<AR_uni_rng_r32_29_ID> ar_kernel_delta_rng_29(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng29, 28>{});
// fpga_tools::Autorun<AR_uni_rng_r32_30_ID> ar_kernel_delta_rng_30(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng30, 29>{});
// fpga_tools::Autorun<AR_uni_rng_r32_31_ID> ar_kernel_delta_rng_31(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng31, 30>{});
// fpga_tools::Autorun<AR_uni_rng_r32_32_ID> ar_kernel_delta_rng_32(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng32, 31>{});

// fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd<pipe_delta_rng1, pipe_delta_rng2, pipe_delta_rng3, pipe_delta_rng4, pipe_delta_rng5, pipe_delta_rng6, pipe_delta_rng7, pipe_delta_rng8, pipe_delta_rng9, pipe_delta_rng10, pipe_delta_rng11, pipe_delta_rng12, pipe_delta_rng13, pipe_delta_rng14, pipe_delta_rng15, pipe_delta_rng16, pipe_delta_rng17, pipe_delta_rng18, pipe_delta_rng19, pipe_delta_rng20, pipe_delta_rng21, pipe_delta_rng22, pipe_delta_rng23, pipe_delta_rng24, pipe_delta_rng25, pipe_delta_rng26, pipe_delta_rng27, pipe_delta_rng28, pipe_delta_rng29, pipe_delta_rng30, pipe_delta_rng31, pipe_delta_rng32, pipearray_tri_rng_2_grng>{});
// fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_grng_out>{});

//Define kernels for rng test fourth
// fpga_tools::Autorun<AR_uni_rng_r32_single_ID> ar_kernel_unirng_r32_single(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_uni_rng_r32, RNG::fixed_point_4_28_unsigned>{});
// fpga_tools::Autorun<AR_uni_rng_r6_ID> ar_kernel_unirng_r6(selector, RNG::AR_LUT_OP_r6_rng<pipe_uni_rng_r6>{});
// fpga_tools::Autorun<AR_alias_table_ID> ar_kernel_aliastable(selector, RNG::AR_Walker_Alias_Table<pipe_uni_rng_r32, pipe_uni_rng_r6, pipe_alias_2_grng>{});
// fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng1, 0, 1, 0>{});
// fpga_tools::Autorun<AR_uni_rng_r32_2_ID> ar_kernel_delta_rng_2(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng2, 2, 3, 1>{});
// fpga_tools::Autorun<AR_uni_rng_r32_3_ID> ar_kernel_delta_rng_3(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng3, 4, 5, 2>{});
// fpga_tools::Autorun<AR_uni_rng_r32_4_ID> ar_kernel_delta_rng_4(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng4, 6, 7, 3>{});
// fpga_tools::Autorun<AR_uni_rng_r32_5_ID> ar_kernel_delta_rng_5(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng5, 8, 9, 4>{});
// fpga_tools::Autorun<AR_uni_rng_r32_6_ID> ar_kernel_delta_rng_6(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng6, 10, 11, 5>{});
// fpga_tools::Autorun<AR_uni_rng_r32_7_ID> ar_kernel_delta_rng_7(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng7, 12, 13, 6>{});
// fpga_tools::Autorun<AR_uni_rng_r32_8_ID> ar_kernel_delta_rng_8(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng8, 14, 15, 7>{});
// fpga_tools::Autorun<AR_uni_rng_r32_9_ID> ar_kernel_delta_rng_9(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng9, 16, 17, 8>{});
// fpga_tools::Autorun<AR_uni_rng_r32_10_ID> ar_kernel_delta_rng_10(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng10, 18, 19, 9>{});
// fpga_tools::Autorun<AR_uni_rng_r32_11_ID> ar_kernel_delta_rng_11(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng11, 20, 21, 10>{});
// fpga_tools::Autorun<AR_uni_rng_r32_12_ID> ar_kernel_delta_rng_12(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng12, 22, 23, 11>{});
// fpga_tools::Autorun<AR_uni_rng_r32_13_ID> ar_kernel_delta_rng_13(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng13, 24, 25, 12>{});
// fpga_tools::Autorun<AR_uni_rng_r32_14_ID> ar_kernel_delta_rng_14(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng14, 26, 27, 13>{});
// fpga_tools::Autorun<AR_uni_rng_r32_15_ID> ar_kernel_delta_rng_15(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng15, 28, 29, 14>{});
// fpga_tools::Autorun<AR_uni_rng_r32_16_ID> ar_kernel_delta_rng_16(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng16, 30, 31, 15>{});
// fpga_tools::Autorun<AR_uni_rng_r32_17_ID> ar_kernel_delta_rng_17(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng17, 1, 2, 16>{});
// fpga_tools::Autorun<AR_uni_rng_r32_18_ID> ar_kernel_delta_rng_18(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng18, 3, 4, 17>{});
// fpga_tools::Autorun<AR_uni_rng_r32_19_ID> ar_kernel_delta_rng_19(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng19, 5, 6, 18>{});
// fpga_tools::Autorun<AR_uni_rng_r32_20_ID> ar_kernel_delta_rng_20(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng20, 7, 8, 19>{});
// fpga_tools::Autorun<AR_uni_rng_r32_21_ID> ar_kernel_delta_rng_21(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng21, 9, 10, 20>{});
// fpga_tools::Autorun<AR_uni_rng_r32_22_ID> ar_kernel_delta_rng_22(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng22, 11, 12, 21>{});
// fpga_tools::Autorun<AR_uni_rng_r32_23_ID> ar_kernel_delta_rng_23(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng23, 13, 14, 22>{});
// fpga_tools::Autorun<AR_uni_rng_r32_24_ID> ar_kernel_delta_rng_24(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng24, 15, 16, 23>{});
// fpga_tools::Autorun<AR_uni_rng_r32_25_ID> ar_kernel_delta_rng_25(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng25, 17, 18, 24>{});
// fpga_tools::Autorun<AR_uni_rng_r32_26_ID> ar_kernel_delta_rng_26(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng26, 19, 20, 25>{});
// fpga_tools::Autorun<AR_uni_rng_r32_27_ID> ar_kernel_delta_rng_27(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng27, 21, 22, 26>{});
// fpga_tools::Autorun<AR_uni_rng_r32_28_ID> ar_kernel_delta_rng_28(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng28, 23, 24, 27>{});
// fpga_tools::Autorun<AR_uni_rng_r32_29_ID> ar_kernel_delta_rng_29(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng29, 25, 26, 28>{});
// fpga_tools::Autorun<AR_uni_rng_r32_30_ID> ar_kernel_delta_rng_30(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng30, 27, 28, 29>{});
// fpga_tools::Autorun<AR_uni_rng_r32_31_ID> ar_kernel_delta_rng_31(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng31, 29, 30, 30>{});
// fpga_tools::Autorun<AR_uni_rng_r32_32_ID> ar_kernel_delta_rng_32(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng32, 31, 0, 31>{});

//fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_tri_rng_2_grng<pipe_delta_rng1, pipe_delta_rng2, pipe_delta_rng3, pipe_delta_rng4, pipe_delta_rng5, pipe_delta_rng6, pipe_delta_rng7, pipe_delta_rng8, pipe_delta_rng9, pipe_delta_rng10, pipe_delta_rng11, pipe_delta_rng12, pipe_delta_rng13, pipe_delta_rng14, pipe_delta_rng15, pipe_delta_rng16, pipe_delta_rng17, pipe_delta_rng18, pipe_delta_rng19, pipe_delta_rng20, pipe_delta_rng21, pipe_delta_rng22, pipe_delta_rng23, pipe_delta_rng24, pipe_delta_rng25, pipe_delta_rng26, pipe_delta_rng27, pipe_delta_rng28, pipe_delta_rng29, pipe_delta_rng30, pipe_delta_rng31, pipe_delta_rng32, pipearray_tri_rng_2_grng>{});
//fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_tri_rng_2_grng<pipe_delta_rng1, pipe_delta_rng2, pipe_delta_rng3, pipe_delta_rng4, pipe_delta_rng5, pipe_delta_rng6, pipe_delta_rng7, pipe_delta_rng8, pipe_delta_rng9, pipe_delta_rng10, pipe_delta_rng11, pipe_delta_rng12, pipe_delta_rng13, pipe_delta_rng14, pipe_delta_rng15, pipe_delta_rng16, pipearray_tri_rng_2_grng>{});
//fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_grng_out>{});
//fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_hadmard>{});
//fpga_tools::Autorun<AR_hadmard_ID> ar_kernel_hadmard(selector, RNG::Hadmard_Transform_16<pipe_hadmard, pipe_grng_out>{});

//test for rng with only connecting one triangular distribution generator and hadmard transform
// fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng1, 0, 1, 8>{});
// fpga_tools::Autorun<AR_hadmard_ID> ar_kernel_hadmard(selector, RNG::Hadmard_Transform_16<pipe_delta_rng1, pipe_grng_out>{});
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
    int size = out_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
          out[i] = Pipe::read();
      }
    });
  });
}

// TEST FOR RNG
// template <typename KernelID, typename Pipe>
// sycl::event SubmitConsume_RNG(sycl::queue& q, sycl::buffer<float, 1>& out_buf){
//   return q.submit([&](sycl::handler& h){
//     sycl::accessor out(out_buf, h, write_only, no_init);
//     h.single_task<KernelID>([=] {
//       for(int i=0; i<N_rng; i++){
//         out[i] = Pipe::read();
//       }
//     });
//   });
// }

// template <typename KernelID, typename Pipe>
// sycl::event SubmitConsume_uniform_RNG_6_test(sycl::queue& q, sycl::buffer<ac_int<5, false>, 1>& out_buf){
//   return q.submit([&](sycl::handler& h){
//     sycl::accessor out(out_buf, h, write_only, no_init);
//     h.single_task<KernelID>([=] {
//       for(int i=0; i<100; i++){
//         out[i] = Pipe::read();
//       }
//     });
//   });
// }





int main(){

std::cout<<"Let's go: \n";


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// TEST FOR AUTORUN //////////////////////////////////

// Number of total numbers of input
int count = DoF_main*N_main;

//bool passed = true;

std::vector<float> out_data(count);
// Clear the output buffer
std::fill(out_data.begin(), out_data.end(), -1);
// Initialize input date ---> can be connected with ROS later to get the value
// for(int i=0; i<DoF_main*N_main; i++){
//   in_data[i] = (float)i;
// }
// std::vector<float> in_data = {
//     0.2,        1.6,         0.5,
//     0.253333,   1.60667,     0.503333,
//     0.306667,   1.61333,     0.506667,
//     0.36,       1.62,        0.51,
//     0.413333,   1.62667,     0.513333,
//     0.466667,   1.63333,     0.516667,
//     0.52,       1.64,        0.52,
//     0.573333,   1.64667,     0.523333,
//     0.626667,   1.65333,     0.526667,
//     0.68,       1.66,        0.53,
//     0.733333,   1.66667,     0.533333,
//     0.786667,   1.67333,     0.536667,
//     0.84,       1.68,        0.54,
//     0.893333,   1.68667,     0.543333,
//     0.946667,   1.69333,     0.546667,
//     1,          1.7,         0.55
// };
// std::vector<float> in_data = {
//     0.2,        1.5,          0.5,
//     0.253333,   1.5,          0.5,
//     0.306667,   1.5,          0.5,
//     0.36,       1.5,          0.5,
//     0.413333,   1.5,          0.5,
//     0.466667,   1.5,          0.5,
//     0.52,       1.5,          0.5,
//     0.573333,   1.5,          0.5,
//     0.626667,   1.5,          0.5,
//     0.68,       1.5,          0.5,
//     0.733333,   1.5,          0.5,
//     0.786667,   1.5,          0.5,
//     0.84,       1.5,          0.5,
//     0.893333,   1.5,          0.5,
//     0.946667,   1.5,          0.5,
//     1,          1.5,          0.5
// };
// std::vector<float> in_data = {
//     0.2,      1.5,       0.5,
//     0.233333, 1.48667,   0.493333,
//     0.266667, 1.47333,   0.486667,
//     0.3,      1.46,      0.48,
//     0.333333, 1.44667,   0.473333,
//     0.366667, 1.43333,   0.466667,
//     0.4,      1.42,      0.46,
//     0.433333, 1.40667,   0.453333,
//     0.466667, 1.39333,   0.446667,
//     0.5,      1.38,      0.44,
//     0.533333, 1.36667,   0.433333,
//     0.566667, 1.35333,   0.426667,
//     0.6,      1.34,      0.42,
//     0.633333, 1.32667,   0.413333,
//     0.666667, 1.31333,   0.406667,
//     0.7,      1.3,       0.4
// };
std::vector<float> in_data = {
    0.3,      1.7,      0.6,
    0.326667, 1.68,     0.590667,
    0.353333, 1.66,     0.581333,
    0.38,     1.64,     0.572,
    0.406667, 1.62,     0.562667,
    0.433333, 1.6,      0.553333,
    0.46,     1.58,     0.544,
    0.486667, 1.56,     0.534667,
    0.513333, 1.54,     0.525333,
    0.54,     1.52,     0.516,
    0.566667, 1.5,      0.506667,
    0.593333, 1.48,     0.497333,
    0.62,     1.46,     0.488,
    0.646667, 1.44,     0.478667,
    0.673333, 1.42,     0.469333,
    0.7,      1.4,      0.46
};
// RNG::fixed_point_delta_0_25 a = 0.125;
// RNG::fixed_point_delta_0_25 b = 0.125;
// ac_fixed<32+3, -1, true> c = a+b;
// std::cout<<"test of fixed point: \n";
// std::cout<<c<<std::endl;

// float prob[32];
// RNG::find_prob(prob);
// std::cout<<"find the probability for rng: \n";
// for(int i=0; i<32; i++){
//   std::cout<<prob[i]<<' ';
// }
// std::cout<<"\n";

//test for statistic calc
// float check_prob[10] = {-3.76, -3.15, -2.15, -1.85, -0.35, 0.35, 1.2, 1.2, 3.3, 4};
// int COUNT_test[32] = {1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1};
// int COUNT[32];
// for(int i=0; i<32; i++){
//   COUNT[i] = 0;
// }
// int N = sizeof(check_prob)/sizeof(float);
// RNG::rng_analyze_delta_0_25(check_prob, COUNT, N);

// std::cout<<"check the statistic calc: \n";
// for(int i=0; i<32; i++)
// std::cout<<"find the probability for rng: \n";
// for(int i=0; i<32; i++){
//   std::cout<<prob[i]<<' ';
// }
// std::cout<<"\n";

//test for statistic calc
// float check_prob[10] = {-3.76, -3.15, -2.15, -1.85, -0.35, 0.35, 1.2, 1.2, 3.3, 4};
// int COUNT_test[32] = {1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1};
// int COUNT[32];
// for(int i=0; i<32; i++){
//   COUNT[i] = 0;
// }
// int N = sizeof(check_prob)/sizeof(float);
// RNG::rng_analyze_delta_0_25(check_prob, COUNT, N);

// std::cout<<"check the statistic calc: \n";
// for(int i=0; i<32; i++){
//   std::cout<<COUNT[i]<<' ';
// }
// std::cout<<"\n";


//in_data = {4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 3.0f};

// DATA FOR RNG TEST
// std::vector<float> OUT(N_rng);
//OUT = {0, 0, 0, 0, 0, 0};
// DATA FOR UNIFORM RNG 6 BIT TEST
//std::vector<ac_int<5, false>> OUT_uniform_rng_6(100);


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
      // submit data to kernel
      SubmitProducerKernel<ARProduceKernel_host_ID, pipe_host_2_input>(q, in_buf);
      // get data from kernel
      SubmitConsumerKernel<ARConsumeKernel_host_ID, pipe_last_2_host>(q, out_buf);

      // TEST FOR RNG
      // sycl::buffer out_buf(OUT);
      // SubmitConsume_RNG<RNG_consume_kernel_test_ID, pipe_grng_out>(q, out_buf);
      // TEST FOR UNIFORM RNG 6 BIT 
      // sycl::buffer out_buf_uniform_rng_6(OUT_uniform_rng_6);
      // SubmitConsume_uniform_RNG_6_test<uniform_rng_6_consume_ID, pipe_uniform_rng_6_test>(q, out_buf_uniform_rng_6);
      

    }
    std::cout<<"submit finished\n";
    // Print the OUTPUT
    std::cout<<"output is: \n";
    for(int i=0; i<DoF_main*N_main; i++){
      std::cout<<out_data[i]<<",\n";
    }
    std::cout<<"\n";
    //print the out for RNG test
    // std::cout<<"Output is: \n";
    // for(int i=0; i<N_rng; i++){
    //   std::cout<<OUT[i]<<std::endl;
    // }
    // std::cout<<"\n";

    // int COUNT[N_stat];
    // float OUT_stat[N_rng];
    // for(int i=0; i<N_rng; i++){
    //   OUT_stat[i] = OUT[i];
    // }

    // for(int i=0; i<N_stat; i++){
    //   COUNT[i] = 0;
    // }
    // // analyze the rng
    // RNG::rng_analyze_delta_0_25(OUT_stat, COUNT, N_rng);
    // // print the statistical analysis
    // std::cout<<"print the statistical analysis: \n";
    // for(int i=0; i<N_stat; i++){
    //   std::cout<<COUNT[i]<<' ';
    // }
    // std::cout<<"\n";

    // //print the out for uniform rng 6 bit test
    // std::cout<<"Output is: \n";
    // for(int i=0; i<100; i++){
    //   std::cout<<OUT_uniform_rng_6[i]<<std::endl;
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
  std::cout<<"finished \n";

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


return 0;

}