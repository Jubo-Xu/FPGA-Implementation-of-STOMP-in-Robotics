#ifndef __DELTA_THETA_BLOCK_HPP__
#define __DELTA_THETA_BLOCK_HPP__

#include <sycl/sycl.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
//#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ap_float_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

#define Num_k_Theta_Calc    4
#define h_constant          8
#define DoF_Theta_Calc      3

namespace Delta_Theta_Block{

    // Define the Find max and min block, which is based on bubble sort -->
    // This can give the max and min at the same time, maybe can reduce the resource
    // by comparing with using both max and min function of sycl
    struct Array_Num_k{
        [[intel::fpga_register]] float Array[Num_k_Theta_Calc];
    };

    Array_Num_k Find_MAX_and_MIN(Array_Num_k Input)
    {
        Array_Num_k output;
        [[intel::fpga_register]] float EU1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        [[intel::fpga_register]] float EU2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        [[intel::fpga_register]] float EU3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        [[intel::fpga_register]] float EU4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        // First EU
        EU1[0] = (Input.Array[0] > Input.Array[1]) ? Input.Array[1] : Input.Array[0];
        EU1[1] = (Input.Array[0] > Input.Array[1]) ? Input.Array[0] : Input.Array[1];
        EU1[2] = Input.Array[2];
        EU1[3] = Input.Array[3];
        // Second EU
        EU2[0] = EU1[0];
        EU2[1] = (EU1[1] > EU1[2]) ? EU1[2] : EU1[1];
        EU2[2] = (EU1[1] > EU1[2]) ? EU1[1] : EU1[2];
        EU2[3] = EU1[3];
        // Third EU
        EU3[0] = (EU2[0] > EU2[1]) ? EU2[1] : EU2[0];
        EU3[1] = (EU2[0] > EU2[1]) ? EU2[0] : EU2[1];
        EU3[2] = (EU2[2] > EU2[3]) ? EU2[3] : EU2[2];
        EU3[3] = (EU2[2] > EU2[3]) ? EU2[2] : EU2[3];
        // Fourth EU
        EU4[0] = EU3[0];
        EU4[1] = (EU3[1] > EU3[2]) ? EU3[2] : EU3[1];
        EU4[2] = (EU3[1] > EU3[2]) ? EU3[1] : EU3[2];
        EU4[3] = EU3[3];
        // Last Comparison
        output.Array[0] = (EU4[0] > EU4[1]) ? EU4[1] : EU4[0];
        output.Array[1] = (EU4[0] > EU4[1]) ? EU4[0] : EU4[1];
        output.Array[2] = EU4[2];
        output.Array[3] = EU4[3];

        return output;
        
    }


    /*--------------------------------------------------------------------------------------
     To seperate the whole hardware block into several kernels in here is because we want to
     use the autorun for our case, maybe the hardware is compiled as fully pipelined, but we
     want to connect the sycl code with ros2 so that we could get some data from ros2 node, and 
     then start running automatically, therefore the input is from host code, and since the 
     operation is between the pipe read and pipe write, we need to reduce the operation complexity
     so that the whole data transformation process can be as fast as possible to mimic the real 
     pipelined processing for hardware in software
     ---------------------------------------------------------------------------------------
    */

    // Define the first kernel ---> calculate the -h*(q-min(q))/(max(q)-min(q))
    template <typename Pipe_in, typename Pipe_out>
    struct Theta_Calc_Kernel1{
        void operator ()() const{
            while(1){
                Array_Num_k INPUT;
                Array_Num_k Max_Min_out;
                [[intel::fpga_register]] float out_of_Subtraction[Num_k_Theta_Calc+1];
                [[intel::fpga_register]] float OUT[Num_k_Theta_Calc];
                size_t index_in = 0;
                size_t index_out = 0;
                fpga_tools::UnrolledLoop<Num_k_Theta_Calc>([&index_in, &INPUT](auto i) {
                    INPUT.Array[index_in++] = Pipe_in::template PipeAt<i>::read();
                });
                Max_Min_out = Find_MAX_and_MIN(INPUT);
                #pragma unroll
                for(int i=0; i<Num_k_Theta_Calc; i++){
                    out_of_Subtraction[i] = INPUT.Array[i] - Max_Min_out.Array[0];
                }
                out_of_Subtraction[Num_k_Theta_Calc] = Max_Min_out.Array[Num_k_Theta_Calc-1] - Max_Min_out.Array[0];

                // Now declare the dividers and multiplication of h, which should be left shift of some bits in hardware
                #pragma unroll
                for(int i=0; i<Num_k_Theta_Calc; i++){
                    OUT[i] = (out_of_Subtraction[i]/out_of_Subtraction[Num_k_Theta_Calc])*h_constant;
                }
                fpga_tools::UnrolledLoop<Num_k_Theta_Calc>([&index_out, &OUT](auto i) {
                    Pipe_out::template PipeAt<i>::write(OUT[index_out++]);
                });
            }
        }
    };

    // Define the second kernel ---> calculate the Exponential, adder tree, and divider
    template <typename Pipe_in, typename Pipe_out>
    struct Theta_Calc_Kernel2{
        void operator ()() const{
            while(1){
                [[intel::fpga_register]] float INPUT[Num_k_Theta_Calc];
                [[intel::fpga_register]] float EXP_out[Num_k_Theta_Calc];
                [[intel::fpga_register]] float OUT[Num_k_Theta_Calc];
                float Adder_Tree_out = 0.0f;
                size_t index_in = 0;
                size_t index_out = 0;
                fpga_tools::UnrolledLoop<Num_k_Theta_Calc>([&index_in, &INPUT](auto i) {
                    INPUT[index_in++] = Pipe_in::template PipeAt<i>::read();
                });
                // For the exponential part, now we use the exponential block for float, later we can change
                // that into fixed point by using ac_fixed type, but we need to test the tradeoff between
                // accurracy and resource consumption
                #pragma unroll
                for(int i=0; i<Num_k_Theta_Calc; i++){
                    EXP_out[i] = sycl::exp(INPUT[i]);
                }
                #pragma unroll
                for(int j=0; j<Num_k_Theta_Calc; j++){
                    Adder_Tree_out = sycl::ext::intel::fpga_reg(Adder_Tree_out) + sycl::ext::intel::fpga_reg(EXP_out[j]);
                }
                #pragma unroll
                for(int k=0; k<Num_k_Theta_Calc; k++){
                    OUT[k] = EXP_out[k]/Adder_Tree_out;
                }
                fpga_tools::UnrolledLoop<Num_k_Theta_Calc>([&index_out, &OUT](auto i) {
                    Pipe_out::template PipeAt<i>::write(OUT[index_out++]);
                });
            }
        }
    };

    // Define the third kernel ---> calculate the sum of p_i*epsilon_i
    template <typename Pipe_in_1, typename Pipe_in_2, typename Pipe_out, typename Pipe_out2>
    struct Theta_Calc_Kernel3{
        void operator ()() const{
            while(1){
                [[intel::fpga_register]] float INPUT_P[Num_k_Theta_Calc];
                [[intel::fpga_register]] float INPUT_EPSILON[DoF_Theta_Calc*Num_k_Theta_Calc];
                [[intel::fpga_register]] float Multi_OUT[DoF_Theta_Calc][Num_k_Theta_Calc];
                [[intel::fpga_register]] float Adder_Tree_out[DoF_Theta_Calc] = {0.0f, 0.0f, 0.0f};
                size_t index_in_p = 0;
                size_t index_in_epsilon = 0;
                size_t index_out = 0;
                size_t index_out_2 = 0;
                fpga_tools::UnrolledLoop<Num_k_Theta_Calc>([&index_in_p, &INPUT_P](auto i) {
                    INPUT_P[index_in_p++] = Pipe_in_1::template PipeAt<i>::read();
                });
                fpga_tools::UnrolledLoop<DoF_Theta_Calc*Num_k_Theta_Calc>([&index_in_epsilon, &INPUT_EPSILON](auto i) {
                    INPUT_EPSILON[index_in_epsilon++] = Pipe_in_2::template PipeAt<i>::read();
                });
                #pragma unroll
                for(int i=0; i<DoF_Theta_Calc; i++){
                #pragma unroll
                for(int j=0; j<Num_k_Theta_Calc; j++){
                        Multi_OUT[i][j] = INPUT_P[j]*INPUT_EPSILON[(i*Num_k_Theta_Calc)+j];
                }
                    #pragma unroll
                    for(int k=0; k<Num_k_Theta_Calc; k++){
                        Adder_Tree_out[i] = sycl::ext::intel::fpga_reg(Adder_Tree_out[i]) + sycl::ext::intel::fpga_reg(Multi_OUT[i][k]);
                    }
                }
                fpga_tools::UnrolledLoop<DoF_Theta_Calc>([&index_out, &Adder_Tree_out](auto i) {
                    Pipe_out::template PipeAt<i>::write(Adder_Tree_out[index_out++]);
                });
                fpga_tools::UnrolledLoop<DoF_Theta_Calc>([&index_out_2, &Adder_Tree_out](auto idx_2) {
                    Pipe_out2::template PipeAt<idx_2>::write(Adder_Tree_out[index_out_2++]);
                });
            }
        }
    };

}

#endif