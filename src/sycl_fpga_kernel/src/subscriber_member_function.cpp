#include <functional>
#include <memory>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "tutorial_interfaces/msg/num.hpp"                                       // CHANGE
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
#include "Adder_Tree.h"
#include "CostFunction.h"
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include "CostSmooth.hpp"
#include "Determination.hpp"
#include "RNG.hpp"
#include "Delta_Theta_Block.hpp"
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/prototype/host_pipes.hpp>

using std::placeholders::_1;
using namespace std::chrono_literals;

// #define DoF     3
// #define N       1
#define N_main          16
#define k_main          4
#define DoF_main        3
#define end_sig_main    2
#define N_rng           2000
#define N_stat          32

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
auto selector = sycl::ext::intel::fpga_emulator_selector_v;

// class AR_test_kernel_ID;
// class AR_consume_ID;
// class Pipe_test_ID;
// class Pipe_out_ID;




// fpga_tools::Autorun<AR_test_kernel_ID> ar_kernel_test(selector, AR_test_kernel{});
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
// ID for host to AR
class AR_sub_2_kernel_ID;
class AR_test_kernel_ID;

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
class PipeArray_pub_2_kernel_ID;
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
//using pipe_last_2_host = sycl::ext::intel::pipe<PipeArray_Last_2_host_ID, float, N_main>;
using pipe_delta_rng1 = sycl::ext::intel::pipe<PipeArray_delta_rng1_ID, float, 10>;
using pipe_host_2_kernel = sycl::ext::intel::prototype::pipe<PipeArray_pub_2_kernel_ID, float, 8>;
using pipe_consume = sycl::ext::intel::prototype::pipe<PipeArray_Last_2_host_ID, float, 8>;
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
struct AR_test_kernel{
  void operator()() const{
    while(1){
      float out = pipe_host_2_kernel::read();
      pipe_host_2_input::write(out);
    }
  }
};
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
fpga_tools::Autorun<ARDetermine_block_ID> ar_kernel_determine(selector, Determine_and_Connections::AR_Determine_Block<pipe_obcost_single_cost, pipe_smooth_2_determ, pipearray_obcost_single_theta, pipearray_last_2_input, pipe_consume, pipearray_end_sig>{});
// Define RNG kernel
fpga_tools::Autorun<AR_uni_rng_r32_1_ID> ar_kernel_delta_rng_1(selector, RNG::AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test<pipe_delta_rng1, 0, 1, 8>{});
fpga_tools::Autorun<AR_RNG_ID> ar_kernel_RNG(selector, RNG::AR_RNG_final<pipe_delta_rng1, pipearray_RNG_out1, pipearray_RNG_out2_2_DT3>{});

fpga_tools::Autorun<AR_test_kernel_ID> ar_kernel_test(selector, AR_test_kernel{});
////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////

// template <typename KernelID, typename Pipe>
// sycl::event SubmitConsumerKernel(sycl::queue& q, sycl::buffer<float, 1>& out_buf) {
//   return q.submit([&](sycl::handler& h) {
//     sycl::accessor out(out_buf, h, sycl::write_only, sycl::no_init);
//     int size = out_buf.size();
//     h.single_task<KernelID>([=] {
//       for (int i = 0; i < size; i++) {
//           out[i] = Pipe::read();
//       }
//     });
//   });
// }

template <typename Pipe>
class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber(sycl::queue &q)
  : Node("minimal_subscriber"), index(0), queue_(q)
  {
    // rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    // custom_qos.durability = RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
    // rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    // custom_qos.history.depth = 20; 
    subscription_ = this->create_subscription<tutorial_interfaces::msg::Num>(    // CHANGE
      "topic", 20, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    //timer_ = create_wall_timer(std::chrono::seconds(5), std::bind(&MinimalSubscriber::subscribeToTopic, this));
  }
  mutable float receive[DoF_main*N_main];

private:
  // void subscribeToTopic()
  // {
  //   subscription_ = this->create_subscription<tutorial_interfaces::msg::Num>(
  //       "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
  // }
  void topic_callback(const tutorial_interfaces::msg::Num & msg) const  // CHANGE
  {
    //RCLCPP_INFO_STREAM(this->get_logger(), "I heard: '" << msg.num << "'");     // CHANGE
    receive[index] = msg.num;
    std::cout<<"check: "<<msg.num<<std::endl;
    Pipe::write(queue_, msg.num);
    
    index++;
    if (index >= DoF_main*N_main) {
      RCLCPP_INFO(this->get_logger(), "Received all values. Stopping subscriber.");
      rclcpp::shutdown();
    }
    
  }
  rclcpp::Subscription<tutorial_interfaces::msg::Num>::SharedPtr subscription_;  // CHANGE
  mutable size_t index;
  sycl::queue& queue_;
  //rclcpp::TimerBase::SharedPtr timer_;
};

//define the publish back to the original file
template <typename Pipe>
class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher(sycl::queue &q)
  : Node("minimal_publisher"), count_(0), max_count_(DoF_main*N_main), queue_(q)
  {
    publisher_ = this->create_publisher<tutorial_interfaces::msg::Num>("sycl_kernel", 20);  // CHANGE
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    if (this->count_ >= this->max_count_) {
      // Stop publishing once the desired count is reached
      RCLCPP_INFO_STREAM(this->get_logger(), "Finished publishing");
      timer_->cancel();
      return;
    }
    auto message = tutorial_interfaces::msg::Num();                                   // CHANGE
    message.num = Pipe::read(queue_);                                                    // CHANGE
    this->count_++;
    RCLCPP_INFO_STREAM(this->get_logger(), "Publishing: '" << message.num << "'");    // CHANGE
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<tutorial_interfaces::msg::Num>::SharedPtr publisher_;             // CHANGE
  size_t count_;
  size_t max_count_;
  sycl::queue& queue_;
};

int main(int argc, char * argv[])
{
  sycl::queue q(selector, exception_handler);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<MinimalSubscriber<pipe_host_2_kernel>>(q);

  std::cout<<"Instantiation finished, lets go: \n";
  rclcpp::spin(node);
  

  std::cout<<"check the result: \n";
  for(int i=0; i<DoF_main*N_main; i++){
    std::cout<<node->receive[i]<<' ';
  }
  std::cout<<"\n";


  
  std::vector<float> IN(DoF_main*N_main);
  for(int i=0; i<DoF_main*N_main; i++){
    IN[i] = node->receive[i];
  }

std::vector<float> out_data(DoF_main*N_main);
// Clear the output buffer
std::fill(out_data.begin(), out_data.end(), -1);

try { 
    // create the queue
    

    sycl::device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // stream data through the Autorun kernel
    std::cout << "Running the Autorun kernel test\n";
    {
      sycl::buffer buf(IN);
      sycl::buffer out_buf(out_data);
      //SubmitConsumerKernel<AR_consume_ID, pipe_consume>(q, out_buf);
      // q.submit([&](sycl::handler &h) {
      //   sycl::accessor a{buf, h};
      //   h.single_task([=] {
      //     #pragma unroll
      //     for(int i=0; i<8; i++){
      //       a[i] = a[i] + 1;
      //     }
      //   });
      // });
    }
    std::cout<<"submit finished\n";
    // for(int i=0; i<8; i++){
    //   std::cout<<IN[i] <<' ';
    // }
    // std::cout<<"\n";
    // for(int i=0; i<8; i++){
    //   std::cout<<out_data[i]<<' ';
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
  

  rclcpp::init(argc, argv);
  auto node_out = std::make_shared<MinimalPublisher<pipe_consume>>(q);
  std::cout<<"fpga execuation finished \n";
  rclcpp::spin(node_out);
  rclcpp::shutdown();

  return 0;
  //rclcpp::shutdown();

  
}