#include <chrono>
#include <memory>
#include <functional>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "tutorial_interfaces/msg/num.hpp"                                            // CHANGE

using namespace std::chrono_literals;
using std::placeholders::_1;

#define DoF   3
#define N     16

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0), max_count_(DoF*N)
  {
  
    publisher_ = this->create_publisher<tutorial_interfaces::msg::Num>("topic", 20);  // CHANGE
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

// int send[DoF*N] = {1, 2, 3};
// float send[DoF*N] = {0.1, 0.2, 0.3};
float send[DoF*N] = {
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
    //message.num = this->count_++;                                                     // CHANGE
    message.num = send[count_];
    count_++;
    RCLCPP_INFO_STREAM(this->get_logger(), "Publishing: '" << message.num << "'");    // CHANGE
    publisher_->publish(message);
  }

  
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<tutorial_interfaces::msg::Num>::SharedPtr publisher_;             // CHANGE
  size_t count_;
  size_t max_count_;
  
};

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber_check"), index(0)
  {
    subscription_ = this->create_subscription<tutorial_interfaces::msg::Num>(    // CHANGE
      "sycl_kernel", 20, std::bind(&MinimalSubscriber::topic_callback, this, _1));
  }
  mutable float receive[DoF*N];

private:
  void topic_callback(const tutorial_interfaces::msg::Num & msg) const  // CHANGE
  {
    //RCLCPP_INFO_STREAM(this->get_logger(), "I heard: '" << msg.num << "'");     // CHANGE
    receive[index] = msg.num;
    index++;
    if (index >= DoF*N) {
      RCLCPP_INFO(this->get_logger(), "Received all values. Stopping subscriber.");
      rclcpp::shutdown();
    }
  }
  rclcpp::Subscription<tutorial_interfaces::msg::Num>::SharedPtr subscription_;  // CHANGE
  mutable size_t index;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node_pub = std::make_shared<MinimalPublisher>();
  auto node_sub = std::make_shared<MinimalSubscriber>();
  // Start the subscriber spin loop in a separate thread

  //rclcpp::spin(node);
  //rclcpp::shutdown();
   // Wait for the desired number of messages or until interrupted
  std::thread thread_pub([&node_pub]() {
    rclcpp::spin(node_pub);
  });
  std::thread thread_sub([&node_sub]() {
    rclcpp::spin(node_sub);
    std::cout<<"check the result: \n";
    for(int i=0; i<DoF*N; i++){
      std::cout<<node_sub->receive[i]<<' ';
    }
    std::cout<<"\n";
  });
  
  thread_pub.join();
  thread_sub.join();
  rclcpp::shutdown();
  return 0;
}
