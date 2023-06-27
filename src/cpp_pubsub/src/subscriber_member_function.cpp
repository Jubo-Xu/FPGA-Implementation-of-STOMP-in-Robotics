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
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/prototype/host_pipes.hpp>

using std::placeholders::_1;
using namespace std::chrono_literals;

#define DoF     3
#define N       1

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

class AR_test_kernel_ID;
class AR_consume_ID;
class Pipe_test_ID;
class Pipe_out_ID;
using pipe_host_2_kernel = sycl::ext::intel::prototype::pipe<Pipe_test_ID, float, 8>;
using pipe_consume = sycl::ext::intel::prototype::pipe<Pipe_out_ID, float, 8>;

struct AR_test_kernel{
  void operator()() const{
    while(1){
      float out = pipe_host_2_kernel::read();
      out = out+0.3;
      pipe_consume::write(out);
    }
  }
};

fpga_tools::Autorun<AR_test_kernel_ID> ar_kernel_test(selector, AR_test_kernel{});

template <typename KernelID, typename Pipe>
sycl::event SubmitConsumerKernel(sycl::queue& q, sycl::buffer<float, 1>& out_buf) {
  return q.submit([&](sycl::handler& h) {
    sycl::accessor out(out_buf, h, sycl::write_only, sycl::no_init);
    int size = out_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
          out[i] = Pipe::read();
      }
    });
  });
}

template <typename Pipe>
class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber(sycl::queue &q)
  : Node("minimal_subscriber"), index(0), queue_(q)
  {
    subscription_ = this->create_subscription<tutorial_interfaces::msg::Num>(    // CHANGE
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
  }
  mutable float receive[DoF*N];

private:
  void topic_callback(const tutorial_interfaces::msg::Num & msg) const  // CHANGE
  {
    //RCLCPP_INFO_STREAM(this->get_logger(), "I heard: '" << msg.num << "'");     // CHANGE
    receive[index] = msg.num;
    Pipe::write(queue_, msg.num);
    index++;
    if (index >= DoF*N) {
      RCLCPP_INFO(this->get_logger(), "Received all values. Stopping subscriber.");
      rclcpp::shutdown();
    }
  }
  rclcpp::Subscription<tutorial_interfaces::msg::Num>::SharedPtr subscription_;  // CHANGE
  mutable size_t index;
  sycl::queue& queue_;
};

//define the publish back to the original file
template <typename Pipe>
class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher(sycl::queue &q)
  : Node("minimal_publisher"), count_(0), max_count_(DoF*N), queue_(q)
  {
    publisher_ = this->create_publisher<tutorial_interfaces::msg::Num>("sycl_kernel", 10);  // CHANGE
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
  rclcpp::spin(node);
  
  std::cout<<"check the result: \n";
  for(int i=0; i<DoF*N; i++){
    std::cout<<node->receive[i]<<' ';
  }
  std::cout<<"\n";


  
  std::vector<float> IN(DoF*N);
  for(int i=0; i<DoF*N; i++){
    IN[i] = node->receive[i];
  }

std::vector<float> out_data(DoF*N);
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
  std::cout<<"finished \n";

  rclcpp::init(argc, argv);
  auto node_out = std::make_shared<MinimalPublisher<pipe_consume>>(q);
  rclcpp::spin(node_out);
  rclcpp::shutdown();

  return 0;
  //rclcpp::shutdown();

  
}