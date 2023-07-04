
#include <rclcpp/rclcpp.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <iostream>
#include <rclcpp_action/rclcpp_action.hpp>
#include <chrono>   
#include "tutorial_interfaces/msg/num.hpp"  
#include <memory>
#include <functional>
#include <thread>
#include <condition_variable>
#include <mutex>

using namespace std::chrono_literals;
using std::placeholders::_1;

#define DoF      3
#define N        16

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0), max_count_(48)
  {
    publisher_ = this->create_publisher<tutorial_interfaces::msg::Num>("topic", 20);  // CHANGE
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

  float send[48];

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
    message.num = send[count_];  
    count_++;                                                   // CHANGE
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
  mutable float receive[48];

private:
  void topic_callback(const tutorial_interfaces::msg::Num & msg) const  // CHANGE
  {
    //RCLCPP_INFO_STREAM(this->get_logger(), "I heard: '" << msg.num << "'");     // CHANGE
    receive[index] = msg.num;
    index++;
    if (index >= 48) {
      RCLCPP_INFO(this->get_logger(), "Received all values. Stopping subscriber.");
      rclcpp::shutdown();
      //return;
    }
  }
  rclcpp::Subscription<tutorial_interfaces::msg::Num>::SharedPtr subscription_;  // CHANGE
  mutable size_t index;
};

void interpolation(float* traj, float* start, float* end, float num){
  float theta1 = start[0];
  float theta2 = start[1];
  float theta3 = start[2];
  for(int i=0; i<num; i++){
    traj[3*i] = theta1;
    traj[3*i+1] = theta2;
    traj[3*i+2] = theta3;
    theta1 += (end[0] - start[0])/(num-1);
    theta2 += (end[1] - start[1])/(num-1);
    theta3 += (end[2] - start[2])/(num-1);
    
  }
}

int main(int argc, char** argv){

    float start[3] = {0.3, 1.7, 0.6};
    float end[3] = {0.7, 1.4, 0.46};
    float Initial_traj[3*16];
    //float traj_from_fpga[3*16];
     std::vector<std::vector<double>> final_traj(16); 
    interpolation(Initial_traj, start, end, 16);
    std::cout<<"check interpolation: \n";
    for(int i=0; i<3*16; i++){
      std::cout<<Initial_traj[i]<<std::endl;
    }
    std::cout<<"\n";

    rclcpp::init(argc, argv);
    auto node1 = rclcpp::Node::make_shared("arm_action_client"); 
    auto node_pub = std::make_shared<MinimalPublisher>();
    auto node_sub = std::make_shared<MinimalSubscriber>();
    //std::vector<double> position1(3); 
    // position1[0] = 0.0; 
    // position1[1] = 0.0; 
    // position1[2] = 0.0;

    std::vector<double> position2(3); 
    std::vector<double> position(3);
    //float Initial_traj[3] = {0.0, 0.0, 0.0};

    // Define a condition variable and mutex for synchronization
    std::condition_variable cv;
    //std::condition_variable cv_2;
    std::mutex mtx;
    // Define a flag to indicate if position2 is updated
    bool isPosition2Updated = false;
    //bool isShutDown = false;
    // position2[0] = 0.2; 
    // position2[1] = 0.2; 
    // position2[2] = 0.2; 
    std::thread thread_pub([&node_pub, &Initial_traj]() {
        for(int i=0; i<48; i++){
            node_pub->send[i] = Initial_traj[i];
        }
        rclcpp::spin(node_pub);
        // Notify the action thread once publishing is finished
    });
    std::thread thread_sub([&node_sub, &final_traj, &position, &mtx, &isPosition2Updated, &cv]() {
        
          std::lock_guard<std::mutex> lock(mtx);
          rclcpp::spin(node_sub);
          std::cout<<"check the result: \n";
          for(int i=0; i<48; i++){
            std::cout<<node_sub->receive[i]<<' ';
          }
          std::cout<<"\n";
          for(int i=0; i<16; i++){
            // std::vector<double> position(3);
            position[0] = (double)node_sub->receive[i*3];
            position[1] = (double)node_sub->receive[i*3+1];
            position[2] = (double)node_sub->receive[i*3+2];
            // final_traj[i][0] = (double)node_sub->receive[i*3];
            // final_traj[i][1] = (double)node_sub->receive[i*3+1];
            // final_traj[i][2] = (double)node_sub->receive[i*3+2];
            final_traj[i] = position;
          }
          isPosition2Updated = true;
          cv.notify_one();
        
    });

    thread_pub.detach();
    thread_sub.detach();

    std::thread thread_action([&position2, &final_traj, &isPosition2Updated, &cv, &mtx, &argc, &argv]() {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&]() { return isPosition2Updated; });
      rclcpp::init(argc, argv);
      auto node = rclcpp::Node::make_shared("arm_action_client"); 
      auto action_client = rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(node,"/joint_trajectory_controller/follow_joint_trajectory");
      while(!action_client->wait_for_action_server()){
        RCLCPP_ERROR(node->get_logger(),"Action server not available after WAITING"); 
            rclcpp::shutdown(); 
        }

      auto goal_msg = control_msgs::action::FollowJointTrajectory::Goal();

      std::vector<std::string> joint_names; 
      joint_names.push_back("joint_1"); 
      joint_names.push_back("joint_2"); 
      joint_names.push_back("joint_3"); 
      //joint_names.push_back("gripper_joint_1");
      goal_msg.trajectory.joint_names = joint_names;
      goal_msg.trajectory.points.resize(16); 
    
    
      for(int i=0; i<16; i++){
        goal_msg.trajectory.points[i].positions = final_traj[i]; 
        goal_msg.trajectory.points[i].time_from_start = rclcpp::Duration(i,0);
      }
      // goal_msg.trajectory.points[0].positions = position1; 
      // goal_msg.trajectory.points[0].time_from_start = rclcpp::Duration(1,0);
      // goal_msg.trajectory.points[1].positions = position2; 
      // goal_msg.trajectory.points[1].time_from_start = rclcpp::Duration(4,0);

      //sending the goal
      RCLCPP_INFO(node->get_logger(),"Sending the goal message"); 

      action_client->async_send_goal(goal_msg); 

      // isShutDown = true;
      // cv_2.notify_one();
      
    });

    // std::thread thread_shut([&isShutDown, &cv_2, &mtx]() {
    //   std::unique_lock<std::mutex> lock(mtx);
    //   cv_2.wait(lock, [&]() { return isShutDown; });
    //   rclcpp::shutdown();
    // });
    
    // thread_pub.join();
    // thread_sub.join();
    //thread_pub.detach();

    thread_action.join();
    //thread_shut.join();

    //rclcpp::init(argc, argv);

    // while(!action_client->wait_for_action_server()){
    //     RCLCPP_ERROR(node->get_logger(),"Action server not available after WAITING"); 
    //         rclcpp::shutdown(); 
    //     }
    
    //rclcpp::shutdown();
    return 0;

}
