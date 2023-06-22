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
#define N_main          5
#define k_main          4
#define DoF_main        3
#define end_sig_main    2
#define N_rng           200
#define N_stat          32
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
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////// DEFINE THE PIPES ///////////////////////////////////
using pipe_host_2_input = sycl::ext::intel::pipe<PipeArray_host_2_Input_ID, float, N_main>;
using pipearray_last_2_input = fpga_tools::PipeArray<PipeArray_Last_2_host_ID, float, N_main, DoF_main>;
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
// fpga_tools::Autorun<ARInput_block_ID> ar_kernel_input(selector, Determine_and_Connections::AR_input_block<pipe_host_2_input, pipearray_last_2_input, pipearray_end_sig, pipearray_input_2_noisy, pipearray_input_2_update, pipearray_input_2_sparse, pipearray_input_2_smooth>{});
// // Define input to noisy kernel
// fpga_tools::Autorun<ARInput_2_kNoisy_ID> ar_kernel_IN2noisy(selector, Determine_and_Connections::AR_Input_2_kNoisy<pipearray_input_2_noisy, pipearray_RNG_out1, pipearray_noisy_out>{});
// // Define obstacle cost kernel
// fpga_tools::Autorun<ARObstacle_cost_ID> ar_kernel_ObsCost(selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel<pipearray_noisy_out, pipearray_obscost_out>{});
// // Define obstacle cost single kernel
// fpga_tools::Autorun<ARObstacle_cost_single_ID> ar_kernel_ObsCost_single(selector, Obstacle_CostFunction::CostFunction_Autorun_Kernel_Single<pipearray_update_out, pipe_obcost_single_cost, pipearray_obcost_single_theta>{});
// // Define Delta theta kernel 1
// fpga_tools::Autorun<ARDelta_theta_kernel1_ID> ar_kernel_DT1(selector, Delta_Theta_Block::Theta_Calc_Kernel1<pipearray_obscost_out, pipearray_DT_1_2>{});
// // Define Delta theta kernel 2
// fpga_tools::Autorun<ARDelta_theta_kernel2_ID> ar_kernel_DT2(selector, Delta_Theta_Block::Theta_Calc_Kernel2<pipearray_DT_1_2, pipearray_DT_2_3>{});
// // Define Delta theta kernel 3
// fpga_tools::Autorun<ARDelta_theta_kernel3_ID> ar_kernel_DT3(selector, Delta_Theta_Block::Theta_Calc_Kernel3<pipearray_DT_2_3, pipearray_RNG_out2_2_DT3, pipearray_DT_out_2_MMUL, pipearray_DT_out_2_smooth>{});
// // Define MMUL kernel
// fpga_tools::Autorun<ARMMUL_ID> ar_kernel_MMUL(selector, smooth::ARMMul<pipearray_DT_out_2_MMUL, pipearray_MMUL_out1, pipearray_MMUL_out2>{});
// // Define Update kernel
// fpga_tools::Autorun<ARTUpdate_ID> ar_kernel_Update(selector, smooth::ARTUpdate<pipearray_input_2_update, pipearray_MMUL_out1, pipearray_update_out>{});
// // Define Sparse MMUL kernel
// fpga_tools::Autorun<ARSparse_MMUL_ID> ar_kernel_Sparse(selector, smooth::ARSparseMul<pipearray_input_2_sparse, pipearray_sparse_2_smooth>{});
// // Define Smooth cost kernel
// fpga_tools::Autorun<ARSmooth_calc_ID> ar_kernel_Smooth(selector, smooth::ARSmooth<pipearray_end_sig, pipearray_MMUL_out2, pipearray_sparse_2_smooth, pipearray_input_2_smooth, pipearray_DT_out_2_smooth, pipe_smooth_2_determ>{});
// // Define Determine block kernel
// fpga_tools::Autorun<ARDetermine_block_ID> ar_kernel_determine(selector, Determine_and_Connections::AR_Determine_Block<pipe_obcost_single_cost, pipe_smooth_2_determ, pipearray_obcost_single_theta, pipearray_last_2_input, pipe_last_2_host, pipearray_end_sig>{});
// // Define RNG kernel
// fpga_tools::Autorun<AR_RNG_ID> ar_kernel_RNG(selector, RNG::AR_RNG<pipearray_RNG_out1, pipearray_RNG_out2_2_DT3>{});
////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
////////////////////////// TEST FOR RNG //////////////////////////////
// Declare the IDs
class RNG_kernel_test_ID;
class RNG_consume_kernel_test_ID;
class Pipe_RNG_test_ID;

using myint = ac_int<1, false>;
using myint4 = ac_int<4, true>;
using fixed_3_2 = ac_fixed<6, 4, true>;

// test for uniform RNG 6 bit
class uniform_rng_6_test_ID;
class uniform_rng_6_consume_ID;
class Pipe_uniform_rng_test_ID;
using pipe_uniform_rng_6_test = sycl::ext::intel::pipe<Pipe_uniform_rng_test_ID, ac_int<5, false>, 2>;

//using pipe_rng_test = sycl::ext::intel::pipe<Pipe_RNG_test_ID, myint, 1>;
using pipe_rng_test = sycl::ext::intel::pipe<Pipe_RNG_test_ID, float, 2>;

//Declare the IDs for RNG kernels
class AR_Gaussian_rng_kernel_ID;
class AR_alias_table_ID;
class AR_uni_rng_r32_single_ID;
class AR_uni_rng_r6_ID;
class AR_tri_rng_ID;
//IDs for multiple rngs test
class AR_uni_rng_r32_1_ID;
class AR_uni_rng_r32_2_ID;
class AR_uni_rng_r32_3_ID;
class AR_uni_rng_r32_4_ID;
class AR_uni_rng_r32_5_ID;
class AR_uni_rng_r32_6_ID;
class AR_uni_rng_r32_7_ID;
class AR_uni_rng_r32_8_ID;
class AR_uni_rng_r32_9_ID;
class AR_uni_rng_r32_10_ID;
class AR_uni_rng_r32_11_ID;
class AR_uni_rng_r32_12_ID;
class AR_uni_rng_r32_13_ID;
class AR_uni_rng_r32_14_ID;
class AR_uni_rng_r32_15_ID;
class AR_uni_rng_r32_16_ID;
class AR_uni_rng_r32_17_ID;
class AR_uni_rng_r32_18_ID;
class AR_uni_rng_r32_19_ID;
class AR_uni_rng_r32_20_ID;
class AR_uni_rng_r32_21_ID;
class AR_uni_rng_r32_22_ID;
class AR_uni_rng_r32_23_ID;
class AR_uni_rng_r32_24_ID;
class AR_uni_rng_r32_25_ID;
class AR_uni_rng_r32_26_ID;
class AR_uni_rng_r32_27_ID;
class AR_uni_rng_r32_28_ID;
class AR_uni_rng_r32_29_ID;
class AR_uni_rng_r32_30_ID;
class AR_uni_rng_r32_31_ID;
class AR_uni_rng_r32_32_ID;
class AR_uni_rng_r32_33_ID;
class AR_uni_rng_r32_34_ID;
class AR_uni_rng_r32_35_ID;
class AR_uni_rng_r32_36_ID;
class AR_uni_rng_r32_37_ID;
class AR_uni_rng_r32_38_ID;
class AR_uni_rng_r32_39_ID;
class AR_uni_rng_r32_40_ID;
class AR_uni_rng_r32_41_ID;
class AR_uni_rng_r32_42_ID;
class AR_uni_rng_r32_43_ID;
class AR_uni_rng_r32_44_ID;
class AR_uni_rng_r32_45_ID;
class AR_uni_rng_r32_46_ID;
class AR_uni_rng_r32_47_ID;
class AR_uni_rng_r32_48_ID;
class AR_uni_rng_r32_49_ID;
class AR_uni_rng_r32_50_ID;
class AR_uni_rng_r32_51_ID;
class AR_uni_rng_r32_52_ID;
class AR_uni_rng_r32_53_ID;
class AR_uni_rng_r32_54_ID;
class AR_uni_rng_r32_55_ID;
class AR_uni_rng_r32_56_ID;
class AR_uni_rng_r32_57_ID;
class AR_uni_rng_r32_58_ID;
class AR_uni_rng_r32_59_ID;
class AR_uni_rng_r32_60_ID;
class AR_uni_rng_r32_61_ID;
class AR_uni_rng_r32_62_ID;
class AR_uni_rng_r32_63_ID;
class AR_uni_rng_r32_64_ID;


//Declare the IDs for pipes
class Pipe_uni_rng_r32_ID;
class Pipe_uni_rng_r6_ID;
class Pipe_alias_2_grng_ID;
class PipeArray_tri_rng_2_grng_ID;
class Pipe_grng_out_ID;
//IDs for pipe of delta rngs
class PipeArray_delta_rng_ID;
//Define the Pipes
using pipe_uni_rng_r32 = sycl::ext::intel::pipe<Pipe_uni_rng_r32_ID, float, 2>;
using pipe_uni_rng_r6 = sycl::ext::intel::pipe<Pipe_uni_rng_r6_ID, RNG::int_6_bit, 2>;
using pipe_alias_2_grng = sycl::ext::intel::pipe<Pipe_alias_2_grng_ID, RNG::int_5_bit, 2>;
using pipearray_tri_rng_2_grng = fpga_tools::PipeArray<Pipe_grng_out_ID, float, 5, 32>;
using pipe_grng_out = sycl::ext::intel::pipe<Pipe_grng_out_ID, float, 2>;
using pipearray_delta_rng = fpga_tools::PipeArray<PipeArray_delta_rng_ID, float, 5, 64>;




template <typename Pipe_out>
struct RNG_test{
  void operator()() const{
    [[intel::fpga_register]] myint BUF[6];
    #pragma unroll
    for(int i=1; i<6; i++){
      BUF[i] = 0;
    }
    BUF[0] = 1;
    while(1){
      // #pragma unroll
      // for(int i=1; i<6; i++){
      //   BUF[6-i] = ext::intel::fpga_reg(BUF[6-i-1]);
      // }
      RNG::FIFO_shift<myint, 6>(BUF);
      Pipe_out::write(BUF[4]);
    }
  }   
};

//fpga_tools::Autorun<RNG_kernel_test_ID> ar_kernel_rng_test(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_grng_out, RNG::fixed_point_4_28_unsigned>{});
//fpga_tools::Autorun<uniform_rng_6_test_ID> ar_kernel_uniform_rng_6_test(selector, RNG::AR_LUT_OP_r6_rng<pipe_uniform_rng_6_test>{});

//test for trianglular distribution
//fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_grng_out>{});
//Define the kernels for RNG
// fpga_tools::Autorun<AR_uni_rng_r32_single_ID> ar_kernel_unirng_r32_single(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_uni_rng_r32, RNG::fixed_point_4_28_unsigned>{});
// fpga_tools::Autorun<AR_uni_rng_r6_ID> ar_kernel_unirng_r6(selector, RNG::AR_LUT_OP_r6_rng<pipe_uni_rng_r6>{});
// fpga_tools::Autorun<AR_alias_table_ID> ar_kernel_aliastable(selector, RNG::AR_Walker_Alias_Table<pipe_uni_rng_r32, pipe_uni_rng_r6, pipe_alias_2_grng>{});
// fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd<pipearray_tri_rng_2_grng>{});
// fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_grng_out>{});
//Define the kernels for RNG_2nd
fpga_tools::Autorun<AR_uni_rng_r32_single_ID> ar_kernel_unirng_r32_single(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48<pipe_uni_rng_r32, RNG::fixed_point_4_28_unsigned>{});
fpga_tools::Autorun<AR_uni_rng_r6_ID> ar_kernel_unirng_r6(selector, RNG::AR_LUT_OP_r6_rng<pipe_uni_rng_r6>{});
fpga_tools::Autorun<AR_alias_table_ID> ar_kernel_aliastable(selector, RNG::AR_Walker_Alias_Table<pipe_uni_rng_r32, pipe_uni_rng_r6, pipe_alias_2_grng>{});
fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 0, 0>{});
fpga_tools::Autorun<AR_uni_rng_r32_2_ID> ar_kernel_delta_rng_2(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 1, 1>{});
fpga_tools::Autorun<AR_uni_rng_r32_3_ID> ar_kernel_delta_rng_3(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 2, 2>{});
fpga_tools::Autorun<AR_uni_rng_r32_4_ID> ar_kernel_delta_rng_4(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 3, 3>{});
fpga_tools::Autorun<AR_uni_rng_r32_5_ID> ar_kernel_delta_rng_5(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 4, 4>{});
fpga_tools::Autorun<AR_uni_rng_r32_6_ID> ar_kernel_delta_rng_6(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 5, 5>{});
fpga_tools::Autorun<AR_uni_rng_r32_7_ID> ar_kernel_delta_rng_7(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 6, 6>{});
fpga_tools::Autorun<AR_uni_rng_r32_8_ID> ar_kernel_delta_rng_8(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 7, 7>{});
fpga_tools::Autorun<AR_uni_rng_r32_9_ID> ar_kernel_delta_rng_9(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 8, 8>{});
fpga_tools::Autorun<AR_uni_rng_r32_10_ID> ar_kernel_delta_rng_10(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 9, 9>{});
fpga_tools::Autorun<AR_uni_rng_r32_11_ID> ar_kernel_delta_rng_11(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 10, 10>{});
fpga_tools::Autorun<AR_uni_rng_r32_12_ID> ar_kernel_delta_rng_12(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 11, 11>{});
fpga_tools::Autorun<AR_uni_rng_r32_13_ID> ar_kernel_delta_rng_13(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 12, 12>{});
fpga_tools::Autorun<AR_uni_rng_r32_14_ID> ar_kernel_delta_rng_14(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 13, 13>{});
fpga_tools::Autorun<AR_uni_rng_r32_15_ID> ar_kernel_delta_rng_15(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 14, 14>{});
fpga_tools::Autorun<AR_uni_rng_r32_16_ID> ar_kernel_delta_rng_16(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 15, 15>{});
fpga_tools::Autorun<AR_uni_rng_r32_17_ID> ar_kernel_delta_rng_17(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 16, 16>{});
fpga_tools::Autorun<AR_uni_rng_r32_18_ID> ar_kernel_delta_rng_18(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 17, 17>{});
fpga_tools::Autorun<AR_uni_rng_r32_19_ID> ar_kernel_delta_rng_19(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 18, 18>{});
fpga_tools::Autorun<AR_uni_rng_r32_20_ID> ar_kernel_delta_rng_20(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 19, 19>{});
fpga_tools::Autorun<AR_uni_rng_r32_21_ID> ar_kernel_delta_rng_21(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 20, 20>{});
fpga_tools::Autorun<AR_uni_rng_r32_22_ID> ar_kernel_delta_rng_22(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 21, 21>{});
fpga_tools::Autorun<AR_uni_rng_r32_23_ID> ar_kernel_delta_rng_23(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 22, 22>{});
fpga_tools::Autorun<AR_uni_rng_r32_24_ID> ar_kernel_delta_rng_24(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 23, 23>{});
fpga_tools::Autorun<AR_uni_rng_r32_25_ID> ar_kernel_delta_rng_25(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 24, 24>{});
fpga_tools::Autorun<AR_uni_rng_r32_26_ID> ar_kernel_delta_rng_26(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 25, 25>{});
fpga_tools::Autorun<AR_uni_rng_r32_27_ID> ar_kernel_delta_rng_27(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 26, 26>{});
fpga_tools::Autorun<AR_uni_rng_r32_28_ID> ar_kernel_delta_rng_28(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 27, 27>{});
fpga_tools::Autorun<AR_uni_rng_r32_29_ID> ar_kernel_delta_rng_29(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 28, 28>{});
fpga_tools::Autorun<AR_uni_rng_r32_30_ID> ar_kernel_delta_rng_30(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 29, 29>{});
fpga_tools::Autorun<AR_uni_rng_r32_31_ID> ar_kernel_delta_rng_31(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 30, 30>{});
fpga_tools::Autorun<AR_uni_rng_r32_32_ID> ar_kernel_delta_rng_32(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 31, 31>{});
fpga_tools::Autorun<AR_uni_rng_r32_33_ID> ar_kernel_delta_rng_33(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 32, 32>{});
fpga_tools::Autorun<AR_uni_rng_r32_34_ID> ar_kernel_delta_rng_34(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 33, 33>{});
fpga_tools::Autorun<AR_uni_rng_r32_35_ID> ar_kernel_delta_rng_35(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 34, 34>{});
fpga_tools::Autorun<AR_uni_rng_r32_36_ID> ar_kernel_delta_rng_36(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 35, 35>{});
fpga_tools::Autorun<AR_uni_rng_r32_37_ID> ar_kernel_delta_rng_37(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 36, 36>{});
fpga_tools::Autorun<AR_uni_rng_r32_38_ID> ar_kernel_delta_rng_38(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 37, 37>{});
fpga_tools::Autorun<AR_uni_rng_r32_39_ID> ar_kernel_delta_rng_39(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 38, 38>{});
fpga_tools::Autorun<AR_uni_rng_r32_40_ID> ar_kernel_delta_rng_40(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 39, 39>{});
fpga_tools::Autorun<AR_uni_rng_r32_41_ID> ar_kernel_delta_rng_41(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 40, 40>{});
fpga_tools::Autorun<AR_uni_rng_r32_42_ID> ar_kernel_delta_rng_42(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 41, 41>{});
fpga_tools::Autorun<AR_uni_rng_r32_43_ID> ar_kernel_delta_rng_43(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 42, 42>{});
fpga_tools::Autorun<AR_uni_rng_r32_44_ID> ar_kernel_delta_rng_44(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 43, 43>{});
fpga_tools::Autorun<AR_uni_rng_r32_45_ID> ar_kernel_delta_rng_45(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 44, 44>{});
fpga_tools::Autorun<AR_uni_rng_r32_46_ID> ar_kernel_delta_rng_46(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 45, 45>{});
fpga_tools::Autorun<AR_uni_rng_r32_47_ID> ar_kernel_delta_rng_47(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 46, 46>{});
fpga_tools::Autorun<AR_uni_rng_r32_48_ID> ar_kernel_delta_rng_48(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 47, 47>{});
fpga_tools::Autorun<AR_uni_rng_r32_49_ID> ar_kernel_delta_rng_49(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 48, 48>{});
fpga_tools::Autorun<AR_uni_rng_r32_50_ID> ar_kernel_delta_rng_50(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 49, 49>{});
fpga_tools::Autorun<AR_uni_rng_r32_51_ID> ar_kernel_delta_rng_51(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 50, 50>{});
fpga_tools::Autorun<AR_uni_rng_r32_52_ID> ar_kernel_delta_rng_52(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 51, 51>{});
fpga_tools::Autorun<AR_uni_rng_r32_53_ID> ar_kernel_delta_rng_53(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 52, 52>{});
fpga_tools::Autorun<AR_uni_rng_r32_54_ID> ar_kernel_delta_rng_54(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 53, 53>{});
fpga_tools::Autorun<AR_uni_rng_r32_55_ID> ar_kernel_delta_rng_55(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 54, 54>{});
fpga_tools::Autorun<AR_uni_rng_r32_56_ID> ar_kernel_delta_rng_56(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 55, 55>{});
fpga_tools::Autorun<AR_uni_rng_r32_57_ID> ar_kernel_delta_rng_57(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 56, 56>{});
fpga_tools::Autorun<AR_uni_rng_r32_58_ID> ar_kernel_delta_rng_58(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 57, 57>{});
fpga_tools::Autorun<AR_uni_rng_r32_59_ID> ar_kernel_delta_rng_59(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 58, 58>{});
fpga_tools::Autorun<AR_uni_rng_r32_60_ID> ar_kernel_delta_rng_60(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 59, 59>{});
fpga_tools::Autorun<AR_uni_rng_r32_61_ID> ar_kernel_delta_rng_61(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 60, 60>{});
fpga_tools::Autorun<AR_uni_rng_r32_62_ID> ar_kernel_delta_rng_62(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 61, 61>{});
fpga_tools::Autorun<AR_uni_rng_r32_63_ID> ar_kernel_delta_rng_63(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 62, 62>{});
fpga_tools::Autorun<AR_uni_rng_r32_64_ID> ar_kernel_delta_rng_64(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipearray_delta_rng, 63, 63>{});

fpga_tools::Autorun<AR_tri_rng_ID> ar_kernel_tri_rng(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd<pipearray_delta_rng, pipearray_tri_rng_2_grng>{});
fpga_tools::Autorun<AR_Gaussian_rng_kernel_ID> ar_kernel_gaussian_rng(selector, RNG::AR_Gaussian_RNG_single<pipe_alias_2_grng, pipearray_tri_rng_2_grng, pipe_grng_out>{});
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
template <typename KernelID, typename Pipe>
sycl::event SubmitConsume_RNG(sycl::queue& q, sycl::buffer<float, 1>& out_buf){
  return q.submit([&](sycl::handler& h){
    sycl::accessor out(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<N_rng; i++){
        out[i] = Pipe::read();
      }
    });
  });
}

template <typename KernelID, typename Pipe>
sycl::event SubmitConsume_uniform_RNG_6_test(sycl::queue& q, sycl::buffer<ac_int<5, false>, 1>& out_buf){
  return q.submit([&](sycl::handler& h){
    sycl::accessor out(out_buf, h, write_only, no_init);
    h.single_task<KernelID>([=] {
      for(int i=0; i<100; i++){
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
// for(int i=0; i<32; i++){
//   std::cout<<COUNT[i]<<' ';
// }
// std::cout<<"\n";


//in_data = {4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f, 1.0f, 3.0f};
// Initialize input date ---> can be connected with ROS later to get the value

// DATA FOR RNG TEST
std::vector<float> OUT(N_rng);
//OUT = {0, 0, 0, 0, 0, 0};
// DATA FOR UNIFORM RNG 6 BIT TEST
std::vector<ac_int<5, false>> OUT_uniform_rng_6(100);

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
      // sycl::buffer in_buf(in_data);
      // sycl::buffer out_buf(out_data);
      // // submit data to kernel
      // SubmitProducerKernel<ARProduceKernel_host_ID, pipe_host_2_input>(q, in_buf);
      // // get data from kernel
      // SubmitConsumerKernel<ARConsumeKernel_host_ID, pipe_last_2_host>(q, out_buf);

      // TEST FOR RNG
      sycl::buffer out_buf(OUT);
      SubmitConsume_RNG<RNG_consume_kernel_test_ID, pipe_grng_out>(q, out_buf);
      // TEST FOR UNIFORM RNG 6 BIT 
      // sycl::buffer out_buf_uniform_rng_6(OUT_uniform_rng_6);
      // SubmitConsume_uniform_RNG_6_test<uniform_rng_6_consume_ID, pipe_uniform_rng_6_test>(q, out_buf_uniform_rng_6);
      

    }
    std::cout<<"submit finished\n";
    // Print the OUTPUT

    //print the out for RNG test
    std::cout<<"Output is: \n";
    for(int i=0; i<N_rng; i++){
      std::cout<<OUT[i]<<std::endl;
    }
    std::cout<<"\n";

    int COUNT[N_stat];
    float OUT_stat[N_rng];
    for(int i=0; i<N_rng; i++){
      OUT_stat[i] = OUT[i];
    }
    float STAT[N_stat];
    int total = 0;
    for(int i=0; i<N_stat; i++){
      COUNT[i] = 0;
    }
    // analyze the rng
    RNG::rng_analyze_delta_0_25(OUT_stat, COUNT, N_rng);
    for(int i=0; i<N_stat; i++){
      total += COUNT[i];
    }
    for(int i=0; i<N_stat; i++){
      STAT[i] = COUNT[i]/total;
    }
    // print the statistical analysis
    std::cout<<"print the statistical analysis: \n";
    for(int i=0; i<N_stat; i++){
      std::cout<<COUNT[i]<<' ';
    }
    std::cout<<"\n";

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