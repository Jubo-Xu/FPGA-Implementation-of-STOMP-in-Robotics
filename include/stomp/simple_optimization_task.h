/**
 * @file simple_optimization_task.h
 * @brief A simple task for showing how to use STOMP
 *
 * @author Jorge Nicho
 * @date Dec 14, 2016
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2016, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EXAMPLES_SIMPLE_OPTIMIZATION_TASK_H_
#define EXAMPLES_SIMPLE_OPTIMIZATION_TASK_H_

#include <stomp/task.h>
#include <algorithm>  
#include<iostream>

namespace stomp_examples
{
//! [SimpleOptimizationTask Inherit]
/** @brief A dummy task for testing STOMP */
class SimpleOptimizationTask : public stomp::Task
{
public:
  /**
   * @brief A simple task for demonstrating how to use Stomp
   * @param parameters_bias default parameter bias used for computing cost.
   * @param bias_thresholds threshold to determine whether two trajectories are equal
   * @param std_dev standard deviation used for generating noisy parameters
   */
  SimpleOptimizationTask(const Eigen::MatrixXd& parameters_bias,
                         const std::vector<double>& bias_thresholds,
                         const std::vector<double>& std_dev)
    : parameters_bias_(parameters_bias), bias_thresholds_(bias_thresholds), std_dev_(std_dev)
  {
    // generate smoothing matrix
    int num_timesteps = parameters_bias.cols();
    stomp::generateSmoothingMatrix(num_timesteps, 1.0, smoothing_M_);
    srand(time(0));
  }

  //! [SimpleOptimizationTask Inherit]
  /**
   * @brief Generates a noisy trajectory from the parameters.
   * @param parameters        A matrix [num_dimensions][num_parameters] of the current optimized parameters
   * @param start_timestep    The start index into the 'parameters' array, usually 0.
   * @param num_timesteps     The number of elements to use from 'parameters' starting from 'start_timestep'
   * @param iteration_number  The current iteration count in the optimization loop
   * @param rollout_number    The index of the noisy trajectory.
   * @param parameters_noise  The parameters + noise
   * @param noise             The noise applied to the parameters
   * @return True if cost were properly computed, otherwise false
   */
  bool generateNoisyParameters(const Eigen::MatrixXd& parameters,
                               std::size_t start_timestep,
                               std::size_t num_timesteps,
                               int iteration_number,
                               int rollout_number,
                               Eigen::MatrixXd& parameters_noise,
                               Eigen::MatrixXd& noise) override
  {
    double rand_noise;
    for (std::size_t d = 0; d < parameters.rows(); d++)
    {
      for (std::size_t t = 0; t < parameters.cols(); t++)
      {
        rand_noise = static_cast<double>(rand() % RAND_MAX) / static_cast<double>(RAND_MAX - 1);  // 0 to 1
        rand_noise = 2 * (0.5 - rand_noise);
        noise(d, t) = rand_noise * std_dev_[d];
      }
    }

    parameters_noise = parameters + noise;

    return true;
  }

  /**
   * @brief computes the state costs as a function of the distance from the bias parameters
   * @param parameters        A matrix [num_dimensions][num_parameters] of the policy parameters to execute
   * @param start_timestep    The start index into the 'parameters' array, usually 0.
   * @param num_timesteps     The number of elements to use from 'parameters' starting from 'start_timestep'
   * @param iteration_number  The current iteration count in the optimization loop
   * @param costs             A vector containing the state costs per timestep.
   * @param validity          Whether or not the trajectory is valid
   * @return True if cost were properly computed, otherwise false
   */
  bool computeCosts(const Eigen::MatrixXd& parameters,
                    std::size_t start_timestep,
                    std::size_t num_timesteps,
                    int iteration_number,
                    Eigen::VectorXd& costs,
                    bool& validity) override
  {
    return computeNoisyCosts(parameters, start_timestep, num_timesteps, iteration_number, -1, costs, validity);
  }

  /**
   * @brief computes the state costs as a function of the distance from the bias parameters
   * @param parameters        A matrix [num_dimensions][num_parameters] of the policy parameters to execute
   * @param start_timestep    The start index into the 'parameters' array, usually 0.
   * @param num_timesteps     The number of elements to use from 'parameters' starting from 'start_timestep'
   * @param iteration_number  The current iteration count in the optimization loop
   * @param rollout_number    The index of the noisy trajectory.
   * @param costs             A vector containing the state costs per timestep.
   * @param validity          Whether or not the trajectory is valid
   * @return True if cost were properly computed, otherwise false
   */

  double cost_function(const double& theta1, const double& theta2, const double& theta3){
    #include <iostream>
double cost_function(const double &theta1, const double &theta2,
                     const double &theta3) {

 

  constexpr double L1 = 0.4 + 0.15;

 

  constexpr double L2 = 0.5;

 

  constexpr double L3 = 0.25;

 

  constexpr double sphere_rad = 0.5;

 

  constexpr double joint1_width = 0.06;

 

  constexpr double joint2_width = 0.04;

 

  constexpr double center_x = 1;

 

  constexpr double center_y = 0.1;

 

  constexpr double center_z = 0;

 

  double x1;
  double y1;
  double z1;
  double x2;
  double y2;
  double z2;
  if (theta1 < M_PI / 2) {
    std::cout << "0\n";
    x1 = L2 * sin(theta2) * sin(theta1) + 0.05 * sin(theta1);
    y1 = -(L2 * sin(theta2) * cos(theta1) - 0.05 * cos(theta1));
    z1 = L1 + L2 * cos(theta2);
    x2 = (L2 * sin(theta2) + L3 * sin(theta2 + theta3)) * sin(theta1);
    y2 = (L2 * sin(theta2) + L3 * sin(theta2 + theta3)) * cos(theta1);
    z2 = z1 + L3 * cos(theta2 + theta3);
  } else if (theta1 >= M_PI / 2) {
    std::cout << "1\n";
    x1 = L2 * sin(theta2) * sin(theta1) + 0.05 * sin(theta1 - M_PI);
    y1 = -(L2 * sin(theta2) * cos(theta1) - 0.05 * cos(theta1 - M_PI));
    z1 = L1 + L2 * cos(theta2);
    x2 = (L2 * sin(theta2) + L3 * sin(theta2 + theta3)) * sin(theta1) +
         0.1 * sin(theta1 - M_PI);
    y2 = -((L2 * sin(theta2) + L3 * sin(theta2 + theta3)) * cos(theta1) -
           0.1 * cos(theta1 - M_PI));
    z2 = z1 + L3 * cos(theta2 + theta3);
  }
  double distance1 = sqrt(pow((x1 - center_x), 2) + pow(y1 - center_y, 2) +
                          pow(z1 - center_z, 2)) -
                     sphere_rad;
  double distance2 = sqrt(pow(x2 - center_x, 2) + pow(y2 - center_y, 2) +
                          pow(z2 - center_z, 2)) -
                     sphere_rad;
  distance1 = std::max(0.5 + joint1_width - distance1, 0.0);
  distance2 = std::max(0.5 + joint2_width - distance2, 0.0);
  std::cout << "cost distance2 " << distance2 << std::endl;
  std::cout << "x1= " << x1 << ", y1=" << y1 << ", z1= " << z1 << std::endl;
  std::cout << "x2= " << x2 << ", y2=" << y2 << ", z2= " << z2 << std::endl;
  double distance = distance1 + distance2;
  return distance;
}
  }
  
  bool computeNoisyCosts(const Eigen::MatrixXd& parameters,
                         std::size_t start_timestep,
                         std::size_t num_timesteps,
                         int iteration_number,
                         int rollout_number,
                         Eigen::VectorXd& costs,
                         bool& validity) override
  {
    
    costs.setZero(num_timesteps);
    double diff;
    double cost; 
    validity = true;

    for (std::size_t t = 0u; t < num_timesteps; t++)
    {
      costs(t) = cost_function(parameters(0, t),parameters(1,t),parameters(2,t)); 
 
    }
    return true;
  }

  /**
   * @brief Filters the given parameters which is applied after the update. It could be used for clipping of joint
   * limits or projecting into the null space of the Jacobian.
   *
   * @param start_timestep    The start index into the 'parameters' array, usually 0.
   * @param num_timesteps     The number of elements to use from 'parameters' starting from 'start_timestep'
   * @param iteration_number  The current iteration count in the optimization loop
   * @param parameters        The optimized parameters
   * @param updates           The updates to the parameters
   * @return                  True if successful, otherwise false
   */
  bool filterParameterUpdates(std::size_t start_timestep,
                              std::size_t num_timesteps,
                              int iteration_number,
                              const Eigen::MatrixXd& parameters,
                              Eigen::MatrixXd& updates) override
  {
    return smoothParameterUpdates(start_timestep, num_timesteps, iteration_number, updates);
  }

protected:
  /**
   * @brief Perform a smooth update given a noisy update
   * @param start_timestep starting timestep
   * @param num_timesteps number of timesteps
   * @param iteration_number number of interations allowed
   * @param updates returned smooth update
   * @return True if successful, otherwise false
   */
  bool smoothParameterUpdates(std::size_t start_timestep,
                              std::size_t num_timesteps,
                              int iteration_number,
                              Eigen::MatrixXd& updates)
  {
    for (auto d = 0u; d < updates.rows(); d++)
    {
      updates.row(d).transpose() = smoothing_M_ * (updates.row(d).transpose());
    }

    return true;
  }

protected:
  Eigen::MatrixXd parameters_bias_;     /**< Parameter bias used for computing cost for the test */
  std::vector<double> bias_thresholds_; /**< Threshold to determine whether two trajectories are equal */
  std::vector<double> std_dev_;         /**< Standard deviation used for generating noisy parameters */
  Eigen::MatrixXd smoothing_M_;         /**< Matrix used for smoothing the trajectory */
};

}  // namespace stomp_examples

#endif /* EXAMPLES_SIMPLE_OPTIMIZATION_TASK_H_ */
