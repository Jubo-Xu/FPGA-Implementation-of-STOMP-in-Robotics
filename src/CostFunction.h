#ifndef __COSTFUNCTION_H__
#define __COSTFUNCTION_H__

#include <sycl/sycl.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
//#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ap_float_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
 
#define TOTAL_BIT 32
#define INT_BIT  16
#define SIGN_BIT  true
#define Dimension 3
#define N_B       2
#define N_cost    5

// Define the constants for autorun blocks
#define DoF       3
#define Num_k     4

// Define the geometric configuration of robotics arm(would be changed later)
#define L1        1
#define L2        1
#define L3        1

// Define the parameters for obstacle sphere
#define sphere_r             1.0f
#define sphere_center_x      1.0f
#define sphere_center_y      1.0f
#define sphere_center_z      1.0f
#define epsilon              1.0f
#define rb                   1.0f

using namespace sycl;
using fixed_input = ac_fixed<TOTAL_BIT, INT_BIT, SIGN_BIT>;
using fixed_square = ac_fixed<2*TOTAL_BIT, 2*INT_BIT, SIGN_BIT>;
using fixed_sqrt  = ac_fixed<2*TOTAL_BIT, 2*INT_BIT, SIGN_BIT>;

class ConstructFromFloat;





/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// Define the operations as functions to be called in autorun kernels
namespace Obstacle_CostFunction{
struct Array_DIM_NB {
  [[intel::fpga_register]] float Array[Dimension][N_B];
};

struct Array_NB{
  [[intel::fpga_register]] float Array[N_B];
};

struct Array_NUMk{
  [[intel::fpga_register]] float Array[Num_k];
};

struct Array_NUMk_DoF{
  [[intel::fpga_register]] float Array[Num_k][DoF];
};

struct Array_DoF{
  [[intel::fpga_register]] float Array[DoF];
};
//Array_DIM_NB Calc_Forward_Kinematics(float *INPUT);
//Array_NB Calc_Distance_AR(Array_DIM_NB Cartesian_input);
//Array_NB Calc_Velocity_AR(Array_DIM_NB Cartesian_input, Array_DIM_NB Cartesian_last);
Array_DIM_NB Calc_Forward_Kinematics(Array_NUMk_DoF INPUT, int i)
{
  Array_DIM_NB out;
  // Define the basic cos and sin terms
  float COS_theta1 = sycl::cos(INPUT.Array[i][0]);
  float SIN_theta1 = sycl::sin(INPUT.Array[i][0]);
  float COS_theta2 = sycl::cos(INPUT.Array[i][1]);
  float SIN_theta2 = sycl::sin(INPUT.Array[i][1]);
  float COS_theta23 = sycl::cos(INPUT.Array[i][1]+INPUT.Array[i][2]);
  float SIN_theta23 = sycl::sin(INPUT.Array[i][1]+INPUT.Array[i][2]);

  // Find the positions
  // point1
  out.Array[0][0] = L2 * COS_theta2 * COS_theta1;
  out.Array[1][0] = L2 * COS_theta2 * SIN_theta1;
  out.Array[2][0] = L1 + L2 * SIN_theta2;
  // point 2
  out.Array[0][1] = (L2*COS_theta2 + L3*COS_theta23) * COS_theta1;
  out.Array[1][1] = (L2*COS_theta2 + L3*COS_theta23) * SIN_theta1;
  out.Array[2][1] = L1 + L2 * SIN_theta2 + L3 * SIN_theta23;

  return out;
}

Array_DIM_NB Calc_Forward_Kinematics_single(Array_DoF INPUT)
{
  Array_DIM_NB out;
  // Define the basic cos and sin terms
  float COS_theta1 = sycl::cos(INPUT.Array[0]);
  float SIN_theta1 = sycl::sin(INPUT.Array[0]);
  float COS_theta2 = sycl::cos(INPUT.Array[1]);
  float SIN_theta2 = sycl::sin(INPUT.Array[1]);
  float COS_theta23 = sycl::cos(INPUT.Array[1]+INPUT.Array[2]);
  float SIN_theta23 = sycl::sin(INPUT.Array[1]+INPUT.Array[2]);

  // Find the positions
  // point1
  out.Array[0][0] = L2 * COS_theta2 * COS_theta1;
  out.Array[1][0] = L2 * COS_theta2 * SIN_theta1;
  out.Array[2][0] = L1 + L2 * SIN_theta2;
  // point 2
  out.Array[0][1] = (L2*COS_theta2 + L3*COS_theta23) * COS_theta1;
  out.Array[1][1] = (L2*COS_theta2 + L3*COS_theta23) * SIN_theta1;
  out.Array[2][1] = L1 + L2 * SIN_theta2 + L3 * SIN_theta23;

  return out;
}

Array_NB Calc_Distance_AR(Array_DIM_NB Cartesian_input)
{
    Array_NB Distance;
    [[intel::fpga_register]] float Sphere_center[3] = {sphere_center_x, sphere_center_y, sphere_center_z};
    float Sphere_radius = sphere_r;
    #pragma unroll
    for(int i=0; i<N_B; i++){
        [[intel::fpga_register]] float square_term[Dimension];
        #pragma unroll
        for(int j=0; j<Dimension; j++){
            square_term[j] = (Cartesian_input.Array[j][i]-Sphere_center[j])*(Cartesian_input.Array[j][i]-Sphere_center[j]);
        }
        float square_add = square_term[0] + square_term[1] + square_term[2];
        Distance.Array[i] = sycl::sqrt(square_add) - Sphere_radius;
    }

    return Distance;
}

Array_NB Calc_Velocity_AR(Array_DIM_NB Cartesian_input, Array_DIM_NB &Cartesian_last)
{
    Array_NB velocity;
    #pragma unroll
    for(int i=0; i<N_B; i++){
        [[intel::fpga_register]] float Diff_1st[3];
        #pragma unroll
        for(int j=0; j<Dimension; j++){
            Diff_1st[j] = Cartesian_input.Array[j][i] - Cartesian_last.Array[j][i];
            Cartesian_last.Array[j][i] = Cartesian_input.Array[j][i];
            
        }
        velocity.Array[i] = sycl::sqrt(Diff_1st[0]*Diff_1st[0] + Diff_1st[1]*Diff_1st[1] + Diff_1st[2]*Diff_1st[2]);
     }

     return velocity;
}

template <typename Pipe_in, typename Pipe_out>
struct CostFunction_Autorun_Kernel{
  void operator()() const {
    Array_DIM_NB Cartesian_last;
    #pragma unroll
    for(int i=0; i<Dimension; i++){
      #pragma unroll
      for(int j=0; j<N_B; j++){
        Cartesian_last.Array[i][j] = 0.0f;
      }
    }
    [[intel::fpga_register]] int COUNT = 0;

    while(1){
      Array_NUMk_DoF Input_to_Kernel;
      Array_DIM_NB Cartesian;
      Array_NB Distance;
      Array_NB Velocity;
      size_t index_in = 0;
      size_t index_out = 0;
      [[intel::fpga_register]] float Out_of_Kernel[Num_k] = {0.0f, 0.0f, 0.0f, 0.0f};
      float epsilon_float = epsilon;
      float r_b_float = rb;

      // set the counter to reset the cartesian_last
      if(COUNT>N_cost-1){
        COUNT = 0;
        #pragma unroll
        for(int i=0; i<Dimension; i++){
          #pragma unroll
          for(int j=0; j<N_B; j++){
            Cartesian_last.Array[i][j] = 0.0f;
          }
        }
      }
      else{
        COUNT++;
      }
      // the input Pipe array has the size of 
      fpga_tools::UnrolledLoop<Num_k*DoF>([&index_in, &Input_to_Kernel](auto i) {
        Input_to_Kernel.Array[index_in%Num_k][index_in/Num_k] = Pipe_in::template PipeAt<i>::read();
        index_in ++;
      });
      #pragma unroll
      for(int i=0; i<Num_k; i++){
        Cartesian = Calc_Forward_Kinematics(Input_to_Kernel, i);
        Distance = Calc_Distance_AR(Cartesian);
        Velocity = Calc_Velocity_AR(Cartesian, Cartesian_last);
        #pragma unroll
        for(size_t k=0; k<N_B; k++){
            if(epsilon_float+r_b_float-sycl::ext::intel::fpga_reg(Distance.Array[k]) > 0){
                Out_of_Kernel[i] = sycl::ext::intel::fpga_reg(Out_of_Kernel[i]) + (epsilon_float+r_b_float-sycl::ext::intel::fpga_reg(Distance.Array[k]))*sycl::ext::intel::fpga_reg(Velocity.Array[k]);
            }
            else{
                Out_of_Kernel[i] = sycl::ext::intel::fpga_reg(Out_of_Kernel[i]);
            }
        }
      }

      fpga_tools::UnrolledLoop<Num_k>([&index_out, &Out_of_Kernel](auto i) {
          Pipe_out::template PipeAt<i>::write(Out_of_Kernel[index_out++]);
      });

    }
  }
};

template <typename Pipe_in, typename Pipe_out, typename Pipe_out_theta>
struct CostFunction_Autorun_Kernel_Single{
  void operator()() const {
    Array_DIM_NB Cartesian_last;
    #pragma unroll
    for(int i=0; i<Dimension; i++){
      #pragma unroll
      for(int j=0; j<N_B; j++){
        Cartesian_last.Array[i][j] = 0.0f;
      }
    }
    [[intel::fpga_register]] int COUNT = 0;

    while(1){
      //Array_NUMk_DoF Input_to_Kernel;
      //[[intel::fpga_register]] float Input_to_Kernel[DoF];
      Array_DoF Input_to_Kernel;
      Array_DIM_NB Cartesian;
      Array_NB Distance;
      Array_NB Velocity;
      size_t index_in = 0;
      size_t index_out = 0;
      float Out_of_Kernel = 0.0f;
      float epsilon_float = epsilon;
      float r_b_float = rb;

      // set the counter to reset the cartesian_last
      if(COUNT>N_cost){
        COUNT = 0;
        #pragma unroll
        for(int i=0; i<Dimension; i++){
          #pragma unroll
          for(int j=0; j<N_B; j++){
            Cartesian_last.Array[i][j] = 0.0f;
          }
        }
      }
      else{
        COUNT++;
      }
      // the input Pipe array has the size of 
      fpga_tools::UnrolledLoop<DoF>([&index_in, &Input_to_Kernel](auto i) {
        Input_to_Kernel.Array[index_in++] = Pipe_in::template PipeAt<i>::read();
      });

        Cartesian = Calc_Forward_Kinematics_single(Input_to_Kernel);
        Distance = Calc_Distance_AR(Cartesian);
        Velocity = Calc_Velocity_AR(Cartesian, Cartesian_last);
        #pragma unroll
        for(size_t k=0; k<N_B; k++){
            if(epsilon_float+r_b_float-sycl::ext::intel::fpga_reg(Distance.Array[k]) > 0){
                Out_of_Kernel = sycl::ext::intel::fpga_reg(Out_of_Kernel) + (epsilon_float+r_b_float-sycl::ext::intel::fpga_reg(Distance.Array[k]))*sycl::ext::intel::fpga_reg(Velocity.Array[k]);
            }
            else{
                Out_of_Kernel = sycl::ext::intel::fpga_reg(Out_of_Kernel);
            }
        }

      // fpga_tools::UnrolledLoop<Num_k>([&index_out, &Out_of_Kernel](auto i) {
      //     Pipe_out::template PipeAt<i>::write(Out_of_Kernel[index_out++]);
      // });
      Pipe_out::write(Out_of_Kernel);
      //size_t index_out = 0;
      fpga_tools::UnrolledLoop<DoF>([&index_out, &Input_to_Kernel](auto idx) {
        Pipe_out_theta::template PipeAt<idx>::write(Input_to_Kernel.Array[index_out++]);
      });

    }
  }
};
}


/////////////////////////////////////////////////////////////////////////////
////////////////////////// DEFINE IN CLASS //////////////////////////////////

class CostFunction{
    private:
        // Define the parameters for float case
        float epsilon_float = 0.5;
        float r_b_float = 0.5;
        // Define the parameters for fixed case
        fixed_sqrt epsilon_fixed = 0.5f;
        fixed_sqrt r_b_fixed = 0.5f;
    public:
        float Obstacle_x = 1.0f;
        float Obstacle_y = 1.0f;
        float Obstacle_a = 1.0f;
        float Obstacle_b = 1.0f;

        
        // For Autorun cases and simpliest test case, where there is only one obstacle, and represent that as a sphere, which can be generalized
        [[intel::fpga_register]] float Input_to_Kernel[DoF][Num_k];
        [[intel::fpga_register]] float INPUT[DoF];
        [[intel::fpga_register]] float Cartesian_Pos[Dimension][N_B];
        [[intel::fpga_register]] float Distance[N_B];
        [[intel::fpga_register]] float Velocity[N_B];
        [[intel::fpga_register]] float Out_of_Kernel[Num_k];

        [[intel::fpga_register]] float Sphere_radius = 1.0f;
        [[intel::fpga_register]] float Sphere_center[3] = {1.0f, 1.0f, 1.0f};

        CostFunction();

        template<typename T_out, typename T_in>
        buffer<T_out, 1> Distance_Calc_fixed(queue &q, buffer<float, Dimension> &BF, int N, T_in ob_x, T_in ob_y, T_in a, T_in b);
        
        //sycl::event Distance_Calc_float(queue &q, buffer<float, Dimension> &BF, buffer<float, 1> &BF_out, int N);

        //sycl::event Velocity_Calc_float(queue &q, buffer<float, Dimension> &BF, buffer<float, 1> &BF_out, int N);

        // Define the cost functions for autorun blocks
        void Forward_Kinematics(void);
        void Find_Distance_AR_float(void);
        void Find_Velocity_AR_float(void);
        void Find_Obstacle_cost_AR(void);
        

        


};

#endif