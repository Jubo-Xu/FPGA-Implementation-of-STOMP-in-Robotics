#ifndef __COSTFUNCTION_H__
#define __COSTFUNCTION_H__

#include <sycl/sycl.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
//#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ap_float_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#define TOTAL_BIT 32
#define INT_BIT  16
#define SIGN_BIT  true
#define Dimension 3
#define N_B       2

// Define the constants for autorun blocks
#define DoF       3
#define Num_k     4

// Define the geometric configuration of robotics arm(would be changed later)
#define L1        1
#define L2        1
#define L3        1

using namespace sycl;
using fixed_input = ac_fixed<TOTAL_BIT, INT_BIT, SIGN_BIT>;
using fixed_square = ac_fixed<2*TOTAL_BIT, 2*INT_BIT, SIGN_BIT>;
using fixed_sqrt  = ac_fixed<2*TOTAL_BIT, 2*INT_BIT, SIGN_BIT>;

class ConstructFromFloat;

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