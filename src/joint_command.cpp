
#include <rclcpp/rclcpp.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <iostream>
#include <rclcpp_action/rclcpp_action.hpp>
#include <chrono>   

int main(int argc, char** argv){
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
    goal_msg.trajectory.points.resize(2); 
    
    std::vector<double> position1(3); 
    position1[0] = 0.0; 
    position1[1] = 0.0; 
    position1[2] = 0.0;

    std::vector<double> position2(3); 
    position2[0] = 1.0; 
    position2[1] = 1.0; 
    position2[2] = 1.0; 
    goal_msg.trajectory.points[0].positions = position1; 
    goal_msg.trajectory.points[0].time_from_start = rclcpp::Duration(1,0);
    goal_msg.trajectory.points[1].positions = position2; 
    goal_msg.trajectory.points[1].time_from_start = rclcpp::Duration(2,0);

    //sending the goal
    RCLCPP_INFO(node->get_logger(),"Sending the goal message"); 

    action_client->async_send_goal(goal_msg); 

    

}