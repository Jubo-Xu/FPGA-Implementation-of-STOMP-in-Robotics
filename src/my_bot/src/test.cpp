#include <rclcpp/rclcpp.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <iostream>
#include <rclcpp_action/rclcpp_action.hpp>
#include <chrono>   

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
    float start[3] = {0.2, 1.5, 0.5};
    float end[3] = {0.7, 1.3, 0.4};
    float Initial_traj[3*16];
    std::vector<std::vector<double>> final_traj(16); 
    interpolation(Initial_traj, start, end, 16);
    std::cout<<"check interpolation: \n";
    for(int i=0; i<3*16; i++){
      std::cout<<Initial_traj[i]<<std::endl;
    }
    std::cout<<"\n";
    for(int i=0; i<16; i++){
        std::vector<double> position(3);
        position[0] = Initial_traj[3*i];
        position[1] = Initial_traj[3*i+1];
        position[2] = Initial_traj[3*i+2];
        final_traj[i] = position;
    }
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
    goal_msg.trajectory.joint_names = joint_names;
    goal_msg.trajectory.points.resize(16); 
    
    // std::vector<double> position1(3); 
    // position1[0] = 0.0; 
    // position1[1] = 0.0; 
    // position1[2] = 0.0;

    // std::vector<double> position2(3); 
    // position2[0] = 1.0; 
    // position2[1] = 1.0; 
    // position2[2] = 1.0; 
    // goal_msg.trajectory.points[0].positions = position1; 
    // goal_msg.trajectory.points[0].time_from_start = rclcpp::Duration(1,0);
    // goal_msg.trajectory.points[1].positions = position2; 
    // goal_msg.trajectory.points[1].time_from_start = rclcpp::Duration(2,0);
    for(int i=0; i<16; i++){
        goal_msg.trajectory.points[i].positions = final_traj[i]; 
        goal_msg.trajectory.points[i].time_from_start = rclcpp::Duration(i,0);
      }

    //sending the goal
    RCLCPP_INFO(node->get_logger(),"Sending the goal message"); 

    action_client->async_send_goal(goal_msg); 

    

}