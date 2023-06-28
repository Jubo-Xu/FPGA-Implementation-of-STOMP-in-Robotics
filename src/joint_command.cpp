
#include <rclcpp/rclcpp.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <iostream>
#include <rclcpp_action/rclcpp_action.hpp>
#include <chrono>  
#include <Eigen/Dense>
#include <stomp/stomp.h>
#include "stomp/simple_optimization_task.h"

using Trajectory = Eigen::MatrixXd; 
static const std::size_t NUM_DIMENSIONS = 3;                               /**< Number of parameters to optimize */
static const std::size_t NUM_TIMESTEPS = 6;                               /**< Number of timesteps */
static const double DELTA_T = 0.1;                                         /**< Timestep in seconds */
static const std::vector<double> START_POS =   {0.2, 1.6, 0.5} ;        /**< Trajectory starting position */
static const std::vector<double> END_POS = {0.9, 1.6, 0.5} ;          /**< Trajectory ending posiiton {0.2, 1.6, 0.5} */
static const std::vector<double> BIAS_THRESHOLD = {0.3, 1.6, 0.50}; /**< Threshold to determine whether two
                                                                              trajectories are equal */
static const std::vector<double> STD_DEV = { 1.0, 1.0, 1.0 }; 
stomp::StompConfiguration create3DOFConfiguration()
{
  //! [Create Config]
  using namespace stomp;

  StompConfiguration c;
  c.num_timesteps = NUM_TIMESTEPS;
  c.num_iterations = 40;
  c.num_dimensions = NUM_DIMENSIONS;
  c.delta_t = DELTA_T;
  c.control_cost_weight = 0.0;
  c.initialization_method = TrajectoryInitializations::LINEAR_INTERPOLATION;
  c.num_iterations_after_valid = 0;
  c.num_rollouts = 20;
  c.max_rollouts = 20;
  //! [Create Config]

  return c;
}

bool compareDiff(const Trajectory& optimized, const Trajectory& desired, const std::vector<double>& thresholds)
{
  auto num_dimensions = optimized.rows();
  Trajectory diff = Trajectory::Zero(num_dimensions, optimized.cols());
  for (auto d = 0u; d < num_dimensions; d++)
  {
    diff.row(d) = optimized.row(d) - desired.row(d);
    diff.row(d).cwiseAbs();
    if ((diff.row(d).array() > thresholds[d]).any())
    {
      return false;
    }
  }

  return true;
}

void interpolate(const std::vector<double>& start,
                 const std::vector<double>& end,
                 std::size_t num_timesteps,
                 Trajectory& traj)
{
  auto dimensions = start.size();
  traj = Eigen::MatrixXd::Zero(dimensions, num_timesteps);
  for (auto d = 0u; d < dimensions; d++)
  {
    double delta = (end[d] - start[d]) / (num_timesteps - 1);
    for (auto t = 0u; t < num_timesteps; t++)
    {
      traj(d, t) = start[d] + t * delta;
    }
  }
}


int main(int argc, char** argv){
    using namespace stomp_examples;
  using namespace stomp;

  /**< Creating a Task with a trajectory bias **/
  
  Trajectory trajectory_bias;
  interpolate(START_POS, END_POS, NUM_TIMESTEPS, trajectory_bias);

  //! [Create Task Object]
  TaskPtr task(new SimpleOptimizationTask(trajectory_bias, BIAS_THRESHOLD, STD_DEV));
  //! [Create Task Object]

  //! [Create STOMP]
  /**< Creating STOMP to find a trajectory close enough to the bias **/
  StompConfiguration config = create3DOFConfiguration();
  Stomp stomp(config, task);
  //! [Create STOMP]

  //! [Solve]
  /**< Optimizing a trajectory close enough to the bias is produced **/
  Trajectory optimized;
  if (stomp.solve(START_POS, END_POS, optimized))
  {
    std::cout <<"number of cols "<<optimized.cols()<<std::endl;
    std::cout << "STOMP succeeded" << std::endl;
    std::cout << "optimized path" << std::endl << optimized <<std::endl; 
  }
  else
  {
    std::cout << "A valid solution was not found" << std::endl;
    
  }
  //! [Solve]

  /**< Further verifying the results */
  if (compareDiff(optimized, trajectory_bias, BIAS_THRESHOLD))
  {
    std::cout << "The solution is within the expected thresholds" << std::endl;
  }
  else
  {
    std::cout << "The solution exceeded the required thresholds" << std::endl;
    return -1;
  }
  std::vector<std::vector<double>> vec_traj(optimized.cols()); 
  std::vector<double> temp_vec(optimized.rows()); 
  for(int i = 0; i < optimized.cols(); ++i){
    for(int j = 0; j < optimized.rows() ; ++j){
        temp_vec[j] = (optimized(j,i)); 
    }
    vec_traj[i] = temp_vec; 

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
    goal_msg.trajectory.points.resize(optimized.cols()); 
    
    std::vector<double> position1(3); 
    position1[0] = 0.0; 
    position1[1] = 0.0; 
    position1[2] = 0.0;

    std::vector<double> position2(3); 
    position2[0] = 1.0; 
    position2[1] = 1.0; 
    position2[2] = 1.0; 

    for(int i = 0; i < optimized.cols(); i++){
        goal_msg.trajectory.points[i].positions = vec_traj[i]; 
        goal_msg.trajectory.points[i].time_from_start = rclcpp::Duration(i,0);

    }

    
    // goal_msg.trajectory.points[0].positions = position1; 
    // goal_msg.trajectory.points[0].time_from_start = rclcpp::Duration(1,0);
    // goal_msg.trajectory.points[1].positions = position2; 
    // goal_msg.trajectory.points[1].time_from_start = rclcpp::Duration(2,0);

    //sending the goal
    RCLCPP_INFO(node->get_logger(),"Sending the goal message"); 

    action_client->async_send_goal(goal_msg); 

    

}