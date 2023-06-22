#ifndef __RNG_HPP__
#define __RNG_HPP__

#include <sycl/sycl.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
//#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ap_float_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"


// The RNG will be completed later
namespace RNG{

    #define K      32+1 //plus one here is because each element of array is not exactly same to D flipflop
    #define r      32
    #define n      1024
    #define INT    1
    #define num_p  32
    #define delta  0.25
    #define sigma    0.6
    #define N_stat 32

    using int_1_bit = ac_int<1, false>;
    using int_6_bit = ac_int<5, false>;
    using int_5_bit = ac_int<5, false>;
    using fixed_point_4_28 = ac_fixed<r, INT, true>;
    using fixed_point_4_28_unsigned = ac_fixed<r, 0, false>;
    //using fixed_point_delta_0_25 = ac_fixed<r+2, -2, true>;
    using fixed_point_delta_0_25 = ac_fixed<r, -2, true>;

    template <typename T, int k>
    void FIFO_shift(T *IN){
      #pragma unroll
      for(int i=1; i<k; i++){
        IN[k-i] = ext::intel::fpga_reg(IN[k-i-1]);
      }
    }

    //function to find the probabilities ---> this function is mainly used to design the rng, not for kernel
    void find_prob(float *IN){
        for(int i=0; i<num_p; i++){
            float pi = 3.1415926535f;
            IN[i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*(delta*(i-num_p/2))*(delta*(i-num_p/2))/(sigma*sigma));
        }
    }
    //function to find the statistic information ---> this function is mainly used to analyze the rng, not for kernel
    void statistic_calc(int *COUNT, float *X, float val, int size){
        if(size == 2){
            if(val>X[1]){
                COUNT[1] += 1;
            }
            else{
                COUNT[0] = COUNT[0]+1;
            }  
        }
        else if(val<X[size/2]){
            statistic_calc(COUNT, X, val, size/2);
        }
        else{
            statistic_calc(COUNT+size/2, X+size/2, val, size/2);
        }
    }
    //function to find the statistics for delta = 0.25 ---> this function is mainly used to analyze the rng, not for kernel
    void rng_analyze_delta_0_25(float *IN, int *COUNT, int N){
        float X[N_stat] = {
            -4, -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25,
            -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
             0,  0.25,  0.5,  0.75,  1,  1.25,  1.5,  1.75,
             2,  2.25,  2.5,  2.75,  3,  3.25,  3.5,  3.75
        };
        for(int i=0; i<N; i++){
            statistic_calc(COUNT, X, IN[i], N_stat);
            
        }
    }


    struct STRUCT_of_LUT_SR_RNG_r32{
        [[intel::fpga_register]] int_1_bit r_out[K-1];
        [[intel::fpga_register]] int_1_bit fifo_out[K-1];
        [[intel::fpga_register]] int_1_bit rng_out[K-1];

        [[intel::fpga_register]] int_1_bit fifo_0[K];
        [[intel::fpga_register]] int_1_bit fifo_1[K-2];
        [[intel::fpga_register]] int_1_bit fifo_2[K];
        [[intel::fpga_register]] int_1_bit fifo_3[K];
        [[intel::fpga_register]] int_1_bit fifo_4[K];
        [[intel::fpga_register]] int_1_bit fifo_5[K];
        [[intel::fpga_register]] int_1_bit fifo_6[K-1];
        [[intel::fpga_register]] int_1_bit fifo_7[K];
        [[intel::fpga_register]] int_1_bit fifo_8[K-6];
        [[intel::fpga_register]] int_1_bit fifo_9[K];
        [[intel::fpga_register]] int_1_bit fifo_10[K];
        [[intel::fpga_register]] int_1_bit fifo_11[K];
        [[intel::fpga_register]] int_1_bit fifo_12[K];
        [[intel::fpga_register]] int_1_bit fifo_13[K];
        [[intel::fpga_register]] int_1_bit fifo_14[K-11];
        [[intel::fpga_register]] int_1_bit fifo_15[K];
        [[intel::fpga_register]] int_1_bit fifo_16[K];
        [[intel::fpga_register]] int_1_bit fifo_17[K];
        [[intel::fpga_register]] int_1_bit fifo_18[K];
        [[intel::fpga_register]] int_1_bit fifo_19[K];
        [[intel::fpga_register]] int_1_bit fifo_20[K];
        [[intel::fpga_register]] int_1_bit fifo_21[K];
        [[intel::fpga_register]] int_1_bit fifo_22[K];
        [[intel::fpga_register]] int_1_bit fifo_23[K];
        [[intel::fpga_register]] int_1_bit fifo_24[K-3];
        [[intel::fpga_register]] int_1_bit fifo_25[K-2];
        [[intel::fpga_register]] int_1_bit fifo_26[K];
        [[intel::fpga_register]] int_1_bit fifo_27[K-3];
        [[intel::fpga_register]] int_1_bit fifo_28[K];
        [[intel::fpga_register]] int_1_bit fifo_29[K-4];
        [[intel::fpga_register]] int_1_bit fifo_30[K];
        [[intel::fpga_register]] int_1_bit fifo_31[K];
    };

    struct STATE_regs_of_uniform_rng{
        [[intel::fpga_register]] int_1_bit state[n] = {
            1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 
            1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
            0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
            1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 
            0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 
            1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 
            0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 
            0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
            1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 
            1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 
            0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 
            0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 
            0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 
            0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 
            1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 
            1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0
        };
    };

    //Define the struct for LUT_op_r6
    struct STRUCT_of_LUT_OP_r6{
        [[intel::fpga_register]] int_1_bit r_out[6];
        [[intel::fpga_register]] int_1_bit fifo_out[6];
        //[[intel::fpga_register]] int_1_bit rng_out[6];
    };

    void STRUCT_initialization(STRUCT_of_LUT_SR_RNG_r32 &IN){
        #pragma unroll
        for(int i=0; i<K-1; i++){
            IN.r_out[i] = 0;
            IN.fifo_out[i] = 0;
            IN.rng_out[i] = 0;
            IN.fifo_6[i] = 0;
        }
        #pragma unroll
        for(int i=0; i<K; i++){
            IN.fifo_0[i] = 0;
            IN.fifo_2[i] = 0;
            IN.fifo_3[i] = 0;
            IN.fifo_4[i] = 0;
            IN.fifo_5[i] = 0;
            IN.fifo_7[i] = 0;
            IN.fifo_9[i] = 0;
            IN.fifo_10[i] = 0;
            IN.fifo_11[i] = 0;
            IN.fifo_12[i] = 0;
            IN.fifo_13[i] = 0;
            IN.fifo_15[i] = 0;
            IN.fifo_16[i] = 0;
            IN.fifo_17[i] = 0;
            IN.fifo_18[i] = 0;
            IN.fifo_19[i] = 0;
            IN.fifo_20[i] = 0;
            IN.fifo_21[i] = 0;
            IN.fifo_22[i] = 0;
            IN.fifo_23[i] = 0;
            IN.fifo_26[i] = 0;
            IN.fifo_28[i] = 0;
            IN.fifo_30[i] = 0;
            IN.fifo_31[i] = 0;
        }

        #pragma unroll
        for(int i=0; i<K-2; i++){
            IN.fifo_1[i] = 0;
            IN.fifo_25[i] = 0;
        }

        #pragma unroll
        for(int i=0; i<K-6; i++){
            IN.fifo_8[i] = 0;
        }

        #pragma unroll
        for(int i=0; i<K-11; i++){
            IN.fifo_14[i] = 0;
        }

        #pragma unroll
        for(int i=0; i<K-3; i++){
            IN.fifo_24[i] = 0;
            IN.fifo_27[i] = 0;
        }

        #pragma unroll
        for(int i=0; i<K-4; i++){
            IN.fifo_29[i] = 0;
        }

    }

    void STRUCT_initialization_r6(STRUCT_of_LUT_OP_r6 &IN){
        #pragma unroll
        for(int i=0; i<6; i++){
            IN.r_out[i] = 0;
        }
        //IN.fifo_out = {0, 1, 0, 0, 1, 1};
        // IN.fifo_out[0] = 0;
        // IN.fifo_out[1] = 1;
        // IN.fifo_out[2] = 0;
        // IN.fifo_out[3] = 0;
        // IN.fifo_out[4] = 1;
        // IN.fifo_out[5] = 1;
        IN.fifo_out[0] = 0;
        IN.fifo_out[1] = 0;
        IN.fifo_out[2] = 1;
        IN.fifo_out[3] = 1;
        IN.fifo_out[4] = 0;
        IN.fifo_out[5] = 0;
    }

    void SR_extension(STRUCT_of_LUT_SR_RNG_r32 &IN){
        //for fifo_0
        IN.fifo_0[0] = IN.r_out[1];
        FIFO_shift<int_1_bit, K>(IN.fifo_0);
        IN.fifo_out[0] = IN.fifo_0[K-1];
        //for fifo_1
        IN.fifo_1[0] = IN.r_out[2];
        FIFO_shift<int_1_bit, K-2>(IN.fifo_1);
        IN.fifo_out[1] = IN.fifo_1[K-2-1];
        //for fifo_2
        IN.fifo_2[0] = IN.r_out[3];
        FIFO_shift<int_1_bit, K>(IN.fifo_2);
        IN.fifo_out[2] = IN.fifo_2[K-1];
        //for fifo_3
        IN.fifo_3[0] = IN.r_out[4];
        FIFO_shift<int_1_bit, K>(IN.fifo_3);
        IN.fifo_out[3] = IN.fifo_3[K-1];
        //for fifo_4
        IN.fifo_4[0] = IN.r_out[5];
        FIFO_shift<int_1_bit, K>(IN.fifo_4);
        IN.fifo_out[4] = IN.fifo_4[K-1];
        //for fifo_5
        IN.fifo_5[0] = IN.r_out[6];
        FIFO_shift<int_1_bit, K>(IN.fifo_5);
        IN.fifo_out[5] = IN.fifo_5[K-1];
        //for fifo_6
        IN.fifo_6[0] = IN.r_out[7];
        FIFO_shift<int_1_bit, K-1>(IN.fifo_6);
        IN.fifo_out[6] = IN.fifo_6[K-1-1];
        //for fifo_7
        IN.fifo_7[0] = IN.r_out[8];
        FIFO_shift<int_1_bit, K>(IN.fifo_7);
        IN.fifo_out[7] = IN.fifo_7[K-1];
        //for fifo_8
        IN.fifo_8[0] = IN.r_out[9];
        FIFO_shift<int_1_bit, K-6>(IN.fifo_8);
        IN.fifo_out[8] = IN.fifo_8[K-6-1];
        //for fifo_9
        IN.fifo_9[0] = IN.r_out[10];
        FIFO_shift<int_1_bit, K>(IN.fifo_9);
        IN.fifo_out[9] = IN.fifo_9[K-1];
        //for fifo_10
        IN.fifo_10[0] = IN.r_out[11];
        FIFO_shift<int_1_bit, K>(IN.fifo_10);
        IN.fifo_out[10] = IN.fifo_10[K-1];
        //for fifo_11
        IN.fifo_11[0] = IN.r_out[12];
        FIFO_shift<int_1_bit, K>(IN.fifo_11);
        IN.fifo_out[11] = IN.fifo_11[K-1];
        //for fifo_12
        IN.fifo_12[0] = IN.r_out[13];
        FIFO_shift<int_1_bit, K>(IN.fifo_12);
        IN.fifo_out[12] = IN.fifo_12[K-1];
        //for fifo_13
        IN.fifo_13[0] = IN.r_out[14];
        FIFO_shift<int_1_bit, K>(IN.fifo_13);
        IN.fifo_out[13] = IN.fifo_13[K-1];
        //for fifo_14
        IN.fifo_14[0] = IN.r_out[15];
        FIFO_shift<int_1_bit, K-11>(IN.fifo_14);
        IN.fifo_out[14] = IN.fifo_14[K-11-1];
        //for fifo_15
        IN.fifo_15[0] = IN.r_out[16];
        FIFO_shift<int_1_bit, K>(IN.fifo_15);
        IN.fifo_out[15] = IN.fifo_15[K-1];
        //for fifo_16
        IN.fifo_16[0] = IN.r_out[17];
        FIFO_shift<int_1_bit, K>(IN.fifo_16);
        IN.fifo_out[16] = IN.fifo_16[K-1];
        //for fifo_17
        IN.fifo_17[0] = IN.r_out[18];
        FIFO_shift<int_1_bit, K>(IN.fifo_17);
        IN.fifo_out[17] = IN.fifo_17[K-1];
        //for fifo_18
        IN.fifo_18[0] = IN.r_out[19];
        FIFO_shift<int_1_bit, K>(IN.fifo_18);
        IN.fifo_out[18] = IN.fifo_18[K-1];
        //for fifo_19
        IN.fifo_19[0] = IN.r_out[20];
        FIFO_shift<int_1_bit, K>(IN.fifo_19);
        IN.fifo_out[19] = IN.fifo_19[K-1];
        //for fifo_20
        IN.fifo_20[0] = IN.r_out[21];
        FIFO_shift<int_1_bit, K>(IN.fifo_20);
        IN.fifo_out[20] = IN.fifo_20[K-1];
        //for fifo_21
        IN.fifo_21[0] = IN.r_out[22];
        FIFO_shift<int_1_bit, K>(IN.fifo_21);
        IN.fifo_out[21] = IN.fifo_21[K-1];
        //for fifo_22
        IN.fifo_22[0] = IN.r_out[23];
        FIFO_shift<int_1_bit, K>(IN.fifo_22);
        IN.fifo_out[22] = IN.fifo_22[K-1];
        //for fifo_23
        IN.fifo_23[0] = IN.r_out[24];
        FIFO_shift<int_1_bit, K>(IN.fifo_23);
        IN.fifo_out[23] = IN.fifo_23[K-1];
        //for fifo_24
        IN.fifo_24[0] = IN.r_out[25];
        FIFO_shift<int_1_bit, K-3>(IN.fifo_24);
        IN.fifo_out[24] = IN.fifo_24[K-3-1];
        //for fifo_25
        IN.fifo_25[0] = IN.r_out[26];
        FIFO_shift<int_1_bit, K-2>(IN.fifo_25);
        IN.fifo_out[25] = IN.fifo_25[K-2-1];
        //for fifo_26
        IN.fifo_26[0] = IN.r_out[27];
        FIFO_shift<int_1_bit, K>(IN.fifo_26);
        IN.fifo_out[26] = IN.fifo_26[K-1];
        //for fifo_27
        IN.fifo_27[0] = IN.r_out[28];
        FIFO_shift<int_1_bit, K-3>(IN.fifo_27);
        IN.fifo_out[27] = IN.fifo_27[K-3-1];
        //for fifo_28
        IN.fifo_28[0] = IN.r_out[29];
        FIFO_shift<int_1_bit, K>(IN.fifo_28);
        IN.fifo_out[28] = IN.fifo_28[K-1];
        //for fifo_29
        IN.fifo_29[0] = IN.r_out[30];
        FIFO_shift<int_1_bit, K-4>(IN.fifo_29);
        IN.fifo_out[29] = IN.fifo_29[K-4-1];
        //for fifo_30
        IN.fifo_30[0] = IN.r_out[31];
        FIFO_shift<int_1_bit, K>(IN.fifo_30);
        IN.fifo_out[30] = IN.fifo_30[K-1];
        //for fifo_31
        IN.fifo_31[0] = IN.r_out[0];
        FIFO_shift<int_1_bit, K>(IN.fifo_31);
        IN.fifo_out[31] = IN.fifo_31[K-1];
    }

    void LUT_SR_RNG_r32_init_with_state(STRUCT_of_LUT_SR_RNG_r32 &IN){
        [[intel::fpga_register]] int_1_bit STATE[n] = {
            1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 
            1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
            0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
            1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 
            0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 
            1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 
            0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,
            0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 
            0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
            1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 
            1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 
            0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 
            0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 
            0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 
            0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 
            1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 
            1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0
        };

        for(int i=0; i<n; i++){
            #pragma unroll
            for(int j=0; j<K-1; j++){
                if(j == 23){
                    IN.r_out[j] = STATE[i];
                }
                else{
                    IN.r_out[j] = IN.fifo_out[j];
                }
            }
            SR_extension(IN);
        }
    }

    void LUT_SR_RNG_r32_init_with_state_2nd(STRUCT_of_LUT_SR_RNG_r32 &IN, STATE_regs_of_uniform_rng &STATE, int idx){
        for(int i=0; i<n; i++){
            #pragma unroll
            for(int j=0; j<K-1; j++){
                if(j == 23){
                    IN.r_out[j] = STATE.state[(idx*32 + i)%n];
                }
                else{
                    IN.r_out[j] = IN.fifo_out[j];
                }
            }
            SR_extension(IN);
        }
    }

    void LUT_SR_rng_r32_XORs(STRUCT_of_LUT_SR_RNG_r32 &IN){
        IN.r_out[0] = 0 ^ IN.fifo_out[16] ^ IN.fifo_out[0] ^ IN.fifo_out[12] ^ IN.fifo_out[27];
        IN.r_out[1] = 0 ^ IN.fifo_out[16] ^ IN.fifo_out[12] ^ IN.fifo_out[20] ^ IN.fifo_out[24] ^ IN.fifo_out[1];
        IN.r_out[2] = 0 ^ IN.fifo_out[2] ^ IN.fifo_out[15] ^ IN.fifo_out[0] ^ IN.fifo_out[11] ^ IN.fifo_out[1];
        IN.r_out[3] = 0 ^ IN.fifo_out[3] ^ IN.fifo_out[2] ^ IN.fifo_out[4] ^ IN.fifo_out[5] ^ IN.fifo_out[25];
        IN.r_out[4] = 0 ^ IN.fifo_out[7] ^ IN.fifo_out[19] ^ IN.fifo_out[4] ^ IN.fifo_out[23] ^ IN.fifo_out[26];
        IN.r_out[5] = 0 ^ IN.fifo_out[10] ^ IN.fifo_out[3] ^ IN.fifo_out[0] ^ IN.fifo_out[5] ^ IN.fifo_out[6];
        IN.r_out[6] = 0 ^ IN.fifo_out[3] ^ IN.fifo_out[9] ^ IN.fifo_out[11] ^ IN.fifo_out[20] ^ IN.fifo_out[6];
        IN.r_out[7] = 0 ^ IN.fifo_out[7] ^ IN.fifo_out[22] ^ IN.fifo_out[24] ^ IN.fifo_out[28] ^ IN.fifo_out[14];
        IN.r_out[8] = 0 ^ IN.fifo_out[7] ^ IN.fifo_out[16] ^ IN.fifo_out[18] ^ IN.fifo_out[8];
        IN.r_out[9] = 0 ^ IN.fifo_out[9] ^ IN.fifo_out[0] ^ IN.fifo_out[18] ^ IN.fifo_out[23] ^ IN.fifo_out[13];
        IN.r_out[10] = 0 ^ IN.fifo_out[10] ^ IN.fifo_out[9] ^ IN.fifo_out[21] ^ IN.fifo_out[29] ^ IN.fifo_out[28];
        IN.r_out[11] = 0 ^ IN.fifo_out[3] ^ IN.fifo_out[15] ^ IN.fifo_out[11] ^ IN.fifo_out[6];
        IN.r_out[12] = 0 ^ IN.fifo_out[18] ^ IN.fifo_out[12] ^ IN.fifo_out[26] ^ IN.fifo_out[24] ^ IN.fifo_out[6];
        IN.r_out[13] = 0 ^ IN.fifo_out[4] ^ IN.fifo_out[31] ^ IN.fifo_out[8] ^ IN.fifo_out[13] ^ IN.fifo_out[30];
        IN.r_out[14] = 0 ^ IN.fifo_out[15] ^ IN.fifo_out[30] ^ IN.fifo_out[5] ^ IN.fifo_out[14] ^ IN.fifo_out[27];
        IN.r_out[15] = 0 ^ IN.fifo_out[15] ^ IN.fifo_out[4] ^ IN.fifo_out[22] ^ IN.fifo_out[8] ^ IN.fifo_out[26];
        IN.r_out[16] = 0 ^ IN.fifo_out[16] ^ IN.fifo_out[2] ^ IN.fifo_out[9] ^ IN.fifo_out[26] ^ IN.fifo_out[14];
        IN.r_out[17] = 0 ^ IN.fifo_out[17] ^ IN.fifo_out[19] ^ IN.fifo_out[12] ^ IN.fifo_out[31] ^ IN.fifo_out[8];
        IN.r_out[18] = 0 ^ IN.fifo_out[21] ^ IN.fifo_out[18] ^ IN.fifo_out[20] ^ IN.fifo_out[5] ^ IN.fifo_out[14];
        IN.r_out[19] = 0 ^ IN.fifo_out[19] ^ IN.fifo_out[4] ^ IN.fifo_out[25] ^ IN.fifo_out[1];
        IN.r_out[20] = 0 ^ IN.fifo_out[10] ^ IN.fifo_out[17] ^ IN.fifo_out[2] ^ IN.fifo_out[20] ^ IN.fifo_out[1];
        IN.r_out[21] = 0 ^ IN.fifo_out[17] ^ IN.fifo_out[16] ^ IN.fifo_out[21] ^ IN.fifo_out[11] ^ IN.fifo_out[25];
        IN.r_out[22] = 0 ^ IN.fifo_out[17] ^ IN.fifo_out[2] ^ IN.fifo_out[22] ^ IN.fifo_out[8] ^ IN.fifo_out[28];
        IN.r_out[23] = 0 ^ IN.fifo_out[0] ^ IN.fifo_out[21] ^ IN.fifo_out[23] ^ IN.fifo_out[29];
        IN.r_out[24] = 0 ^ IN.fifo_out[11] ^ IN.fifo_out[24] ^ IN.fifo_out[13] ^ IN.fifo_out[6] ^ IN.fifo_out[27];
        IN.r_out[25] = 0 ^ IN.fifo_out[7] ^ IN.fifo_out[22] ^ IN.fifo_out[12] ^ IN.fifo_out[30] ^ IN.fifo_out[25];
        IN.r_out[26] = 0 ^ IN.fifo_out[10] ^ IN.fifo_out[21] ^ IN.fifo_out[18] ^ IN.fifo_out[22] ^ IN.fifo_out[26];
        IN.r_out[27] = 0 ^ IN.fifo_out[17] ^ IN.fifo_out[29] ^ IN.fifo_out[13] ^ IN.fifo_out[5] ^ IN.fifo_out[27];
        IN.r_out[28] = 0 ^ IN.fifo_out[10] ^ IN.fifo_out[19] ^ IN.fifo_out[15] ^ IN.fifo_out[28] ^ IN.fifo_out[14];
        IN.r_out[29] = 0 ^ IN.fifo_out[23] ^ IN.fifo_out[31] ^ IN.fifo_out[29] ^ IN.fifo_out[20] ^ IN.fifo_out[30];
        IN.r_out[30] = 0 ^ IN.fifo_out[9] ^ IN.fifo_out[31] ^ IN.fifo_out[30] ^ IN.fifo_out[25] ^ IN.fifo_out[28];
        IN.r_out[31] = 0 ^ IN.fifo_out[31] ^ IN.fifo_out[29] ^ IN.fifo_out[24] ^ IN.fifo_out[13] ^ IN.fifo_out[1];
    }

    void LUT_SR_rng_r32_permutation(STRUCT_of_LUT_SR_RNG_r32 &IN){
        IN.rng_out[0] = IN.r_out[6];
        IN.rng_out[1] = IN.r_out[10];
        IN.rng_out[2] = IN.r_out[26];
        IN.rng_out[3] = IN.r_out[14];
        IN.rng_out[4] = IN.r_out[2];
        IN.rng_out[5] = IN.r_out[30];
        IN.rng_out[6] = IN.r_out[0];
        IN.rng_out[7] = IN.r_out[22];
        IN.rng_out[8] = IN.r_out[24];
        IN.rng_out[9] = IN.r_out[27];
        IN.rng_out[10] = IN.r_out[29];
        IN.rng_out[11] = IN.r_out[13];
        IN.rng_out[12] = IN.r_out[8];
        IN.rng_out[13] = IN.r_out[28];
        IN.rng_out[14] = IN.r_out[7];
        IN.rng_out[15] = IN.r_out[3];
        IN.rng_out[16] = IN.r_out[23];
        IN.rng_out[17] = IN.r_out[9];
        IN.rng_out[18] = IN.r_out[21];
        IN.rng_out[19] = IN.r_out[19];
        IN.rng_out[20] = IN.r_out[20];
        IN.rng_out[21] = IN.r_out[25];
        IN.rng_out[22] = IN.r_out[18];
        IN.rng_out[23] = IN.r_out[17];
        IN.rng_out[24] = IN.r_out[31];
        IN.rng_out[25] = IN.r_out[1];
        IN.rng_out[26] = IN.r_out[4];
        IN.rng_out[27] = IN.r_out[16];
        IN.rng_out[28] = IN.r_out[12];
        IN.rng_out[29] = IN.r_out[5];
        IN.rng_out[30] = IN.r_out[11];
        IN.rng_out[31] = IN.r_out[15];
    }

    void LUT_OP_rng_r6(STRUCT_of_LUT_OP_r6 &IN){
        IN.r_out[0] = IN.fifo_out[3] ^ IN.fifo_out[4] ^ IN.fifo_out[5];
        IN.r_out[1] = IN.fifo_out[0] ^ IN.fifo_out[1] ^ IN.fifo_out[5];
        IN.r_out[2] = IN.fifo_out[0] ^ IN.fifo_out[4] ^ IN.fifo_out[5];
        IN.r_out[3] = IN.fifo_out[1] ^ IN.fifo_out[2] ^ IN.fifo_out[3];
        IN.r_out[4] = IN.fifo_out[0] ^ IN.fifo_out[1] ^ IN.fifo_out[4];
        IN.r_out[5] = IN.fifo_out[2] ^ IN.fifo_out[3];
        // #pragma unroll
        // for(int i=1; i<6; i++){
        //     IN.fifo_out[6-i] = ext::intel::fpga_reg(IN.r_out[6-i-1]);
        // }
        #pragma unroll
        for(int i=0; i<6; i++){
            IN.fifo_out[i] = IN.r_out[i];
        }
        
    }

    template <typename Pipe_out, typename AC_type>
    struct AR_rng_n1024_r32_t5_k32_s1c48{
        void operator()() const{
            STRUCT_of_LUT_SR_RNG_r32 REGs;
            STRUCT_initialization(REGs);
            LUT_SR_RNG_r32_init_with_state(REGs);
            while(1){
                [[intel::fpga_register]] int_1_bit rng_out[K-1];
                //fixed_point_4_28 ac_fixed_out = 0.0f;
                AC_type ac_fixed_out = 0.0f;
                LUT_SR_rng_r32_XORs(REGs);
                //Permutation of the output
                rng_out[0] = REGs.r_out[6];
                rng_out[1] = REGs.r_out[10];
                rng_out[2] = REGs.r_out[26];
                rng_out[3] = REGs.r_out[14];
                rng_out[4] = REGs.r_out[2];
                rng_out[5] = REGs.r_out[30];
                rng_out[6] = REGs.r_out[0];
                rng_out[7] = REGs.r_out[22];
                rng_out[8] = REGs.r_out[24];
                rng_out[9] = REGs.r_out[27];
                rng_out[10] = REGs.r_out[29];
                rng_out[11] = REGs.r_out[13];
                rng_out[12] = REGs.r_out[8];
                rng_out[13] = REGs.r_out[28];
                rng_out[14] = REGs.r_out[7];
                rng_out[15] = REGs.r_out[3];
                rng_out[16] = REGs.r_out[23];
                rng_out[17] = REGs.r_out[9];
                rng_out[18] = REGs.r_out[21];
                rng_out[19] = REGs.r_out[19];
                rng_out[20] = REGs.r_out[20];
                rng_out[21] = REGs.r_out[25];
                rng_out[22] = REGs.r_out[18];
                rng_out[23] = REGs.r_out[17];
                rng_out[24] = REGs.r_out[31];
                rng_out[25] = REGs.r_out[1];
                rng_out[26] = REGs.r_out[4];
                rng_out[27] = REGs.r_out[16];
                rng_out[28] = REGs.r_out[12];
                rng_out[29] = REGs.r_out[5];
                rng_out[30] = REGs.r_out[11];
                rng_out[31] = REGs.r_out[15];
                //transform to ac_fixed type and then transfer to float
                #pragma unroll
                for(int i=0; i<r; i++){
                    ac_fixed_out[i] = rng_out[i];
                }
                float output = ac_fixed_out.to_float();
                Pipe_out::write(output);
                SR_extension(REGs);
            }
        }
    };

    // template <typename Pipe_out>
    // struct AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test{
    //     void operator()() const{
    //         STRUCT_of_LUT_SR_RNG_r32 REGs;
    //         STATE_regs_of_uniform_rng STATE;
    //         STRUCT_initialization(REGs);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs, STATE, 1);
    //         while(1){
    //             fixed_point_delta_0_25 ac_fixed_out = 0.0f;
    //             LUT_SR_rng_r32_XORs(REGs);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out[i] = REGs.rng_out[i];
    //             }
    //             float output = ac_fixed_out.to_float();
    //             Pipe_out::write(output);
    //             SR_extension(REGs);
    //         }
    //     }
    // };

    template <typename Pipe_out, int n_in, int idx>
    struct AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test{
        void operator()() const{
            STRUCT_of_LUT_SR_RNG_r32 REGs;
            STATE_regs_of_uniform_rng STATE;
            STRUCT_initialization(REGs);
            LUT_SR_RNG_r32_init_with_state_2nd(REGs, STATE, n_in);
            while(1){
                fixed_point_delta_0_25 ac_fixed_out = 0.0f;
                LUT_SR_rng_r32_XORs(REGs);
                //Permutation of the output
                LUT_SR_rng_r32_permutation(REGs);
                #pragma unroll
                for(int i=0; i<r; i++){
                    ac_fixed_out[i] = REGs.rng_out[i];
                }
                float output = ac_fixed_out.to_float();
                //Pipe_out::write(output);
                Pipe_out::template PipeAt<idx>::write(output);
                SR_extension(REGs);
            }
        }
    };


    template <typename Pipe_out>
    struct AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25{
        void operator()() const{
            STRUCT_of_LUT_SR_RNG_r32 REGs[2*num_p];
            STATE_regs_of_uniform_rng STATE;
            #pragma unroll
            for(int i=0; i<num_p; i++){
                STRUCT_initialization(REGs[i]);
                LUT_SR_RNG_r32_init_with_state_2nd(REGs[i], STATE, i);
            }
            for(int i=num_p; i<2*num_p; i++){
                STRUCT_initialization(REGs[i]);
                //LUT_SR_RNG_r32_init_with_state_2nd(REGs[i], STATE, (i+1)%(2*num_p));
                LUT_SR_RNG_r32_init_with_state_2nd(REGs[i], STATE, i);
            }
            while(1){
                [[intel::fpga_register]] fixed_point_delta_0_25 ac_fixed_out[2*num_p];
                [[intel::fpga_register]] float output[2*num_p];
                [[intel::fpga_register]] float tri_out_ini[num_p];
                //Initialization of ac_fixed_out
                #pragma unroll
                for(int i=0; i<2*num_p; i++){
                    ac_fixed_out[i] = 0.0f;
                    LUT_SR_rng_r32_XORs(REGs[i]);
                    //Permutation of the output
                    LUT_SR_rng_r32_permutation(REGs[i]);
                    //transform to ac_fixed type and then transfer to float
                    // (ac_fixed_out[i])[r+2-1] = REGs[i].rng_out[r-1];
                    // #pragma unroll
                    // for(int j=0; j<r-1; j++){
                    //     (ac_fixed_out[i])[j] = REGs[i].rng_out[j];
                    // }
                    #pragma unroll
                    for(int j=0; j<r; j++){
                        (ac_fixed_out[i])[j] = REGs[i].rng_out[j];
                    }
                    output[i] = (ac_fixed_out[i]).to_float();
                    SR_extension(REGs[i]);
                }
                //using U1 - U2 to represent the triangle distribution with mean = 0, and delta = 0.25
                #pragma unroll
                for(int i=0; i<num_p; i++){
                    tri_out_ini[i] = output[2*i] - output[2*i+1];
                }
                //set different mean for these triangle distributions
                #pragma unroll
                for(int i=0; i<num_p; i++){
                    tri_out_ini[i] = tri_out_ini[i] + delta*(i-num_p/2);
                }
                //pipe out
                size_t index = 0;
                fpga_tools::UnrolledLoop<num_p>([&tri_out_ini, &index](auto idx){
                    Pipe_out::template PipeAt<idx>::write(tri_out_ini[index++]);
                });
                
            }
        }
    };

    template <typename Pipe_in, typename Pipe_out>
    struct AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd{
        void operator()() const{
            while(1){
                [[intel::fpga_register]] float output[2*num_p];
                 [[intel::fpga_register]] float tri_out_ini[num_p];
                size_t index_in = 0;
                fpga_tools::UnrolledLoop<2*num_p>([&output, &index_in](auto idx_in){
                    output[index_in++] = Pipe_in::template PipeAt<idx_in>::read();
                });
                 //using U1 - U2 to represent the triangle distribution with mean = 0, and delta = 0.25
                #pragma unroll
                for(int i=0; i<num_p; i++){
                    tri_out_ini[i] = output[2*i] - output[2*i+1];
                }
                //set different mean for these triangle distributions
                #pragma unroll
                for(int i=0; i<num_p; i++){
                    tri_out_ini[i] = tri_out_ini[i] + delta*(i-num_p/2);
                }
                //pipe out
                size_t index_out = 0;
                fpga_tools::UnrolledLoop<num_p>([&tri_out_ini, &index_out](auto idx){
                    Pipe_out::template PipeAt<idx>::write(tri_out_ini[index_out++]);
                });
            }
        }
    };

    template <typename Pipe_out>
    struct AR_LUT_OP_r6_rng{
        void operator()() const{
            STRUCT_of_LUT_OP_r6 REGs;
            STRUCT_initialization_r6(REGs);
            while(1){
                int_6_bit out = 0;
                LUT_OP_rng_r6(REGs);
                #pragma unroll
                for(int i=0; i<5; i++){
                    out[i] = REGs.r_out[i];
                }
                Pipe_out::write(out);
            }
        }
    };

    // template <typename Pipe_out>
    // struct AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_2nd{
    //     void operator()() const{
    //         STRUCT_of_LUT_SR_RNG_r32 REGs1;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs2;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs3;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs4;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs5;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs6;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs7;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs8;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs9;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs10;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs11;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs12;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs13;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs14;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs15;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs16;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs17;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs18;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs19;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs20;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs21;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs22;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs23;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs24;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs25;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs26;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs27;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs28;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs29;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs30;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs31;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs32;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs33;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs34;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs35;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs36;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs37;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs38;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs39;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs40;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs41;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs42;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs43;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs44;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs45;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs46;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs47;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs48;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs49;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs50;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs51;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs52;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs53;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs54;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs55;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs56;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs57;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs58;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs59;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs60;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs61;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs62;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs63;
    //         STRUCT_of_LUT_SR_RNG_r32 REGs64;

    //         STATE_regs_of_uniform_rng STATE;
    //         // #pragma unroll
    //         // for(int i=0; i<num_p; i++){
    //         //     STRUCT_initialization(REGs[i]);
    //         //     LUT_SR_RNG_r32_init_with_state_2nd(REGs[i], STATE, i);
    //         // }
    //         // for(int i=num_p; i<2*num_p; i++){
    //         //     STRUCT_initialization(REGs[i]);
    //         //     //LUT_SR_RNG_r32_init_with_state_2nd(REGs[i], STATE, (i+1)%(2*num_p));
    //         //     LUT_SR_RNG_r32_init_with_state_2nd(REGs[i], STATE, i);
    //         // }
    //         STRUCT_initialization(REGs1);
    //         STRUCT_initialization(REGs2);
    //         STRUCT_initialization(REGs3);
    //         STRUCT_initialization(REGs4);
    //         STRUCT_initialization(REGs5);
    //         STRUCT_initialization(REGs6);
    //         STRUCT_initialization(REGs7);
    //         STRUCT_initialization(REGs8);
    //         STRUCT_initialization(REGs9);
    //         STRUCT_initialization(REGs10);
    //         STRUCT_initialization(REGs11);
    //         STRUCT_initialization(REGs12);
    //         STRUCT_initialization(REGs13);
    //         STRUCT_initialization(REGs14);
    //         STRUCT_initialization(REGs15);
    //         STRUCT_initialization(REGs16);
    //         STRUCT_initialization(REGs17);
    //         STRUCT_initialization(REGs18);
    //         STRUCT_initialization(REGs19);
    //         STRUCT_initialization(REGs20);
    //         STRUCT_initialization(REGs21);
    //         STRUCT_initialization(REGs22);
    //         STRUCT_initialization(REGs23);
    //         STRUCT_initialization(REGs24);
    //         STRUCT_initialization(REGs25);
    //         STRUCT_initialization(REGs26);
    //         STRUCT_initialization(REGs27);
    //         STRUCT_initialization(REGs28);
    //         STRUCT_initialization(REGs29);
    //         STRUCT_initialization(REGs30);
    //         STRUCT_initialization(REGs31);
    //         STRUCT_initialization(REGs32);
    //         STRUCT_initialization(REGs33);
    //         STRUCT_initialization(REGs34);
    //         STRUCT_initialization(REGs35);
    //         STRUCT_initialization(REGs36);
    //         STRUCT_initialization(REGs37);
    //         STRUCT_initialization(REGs38);
    //         STRUCT_initialization(REGs39);
    //         STRUCT_initialization(REGs40);
    //         STRUCT_initialization(REGs41);
    //         STRUCT_initialization(REGs42);
    //         STRUCT_initialization(REGs43);
    //         STRUCT_initialization(REGs44);
    //         STRUCT_initialization(REGs45);
    //         STRUCT_initialization(REGs46);
    //         STRUCT_initialization(REGs47);
    //         STRUCT_initialization(REGs48);
    //         STRUCT_initialization(REGs49);
    //         STRUCT_initialization(REGs50);
    //         STRUCT_initialization(REGs51);
    //         STRUCT_initialization(REGs52);
    //         STRUCT_initialization(REGs53);
    //         STRUCT_initialization(REGs54);
    //         STRUCT_initialization(REGs55);
    //         STRUCT_initialization(REGs56);
    //         STRUCT_initialization(REGs57);
    //         STRUCT_initialization(REGs58);
    //         STRUCT_initialization(REGs59);
    //         STRUCT_initialization(REGs60);
    //         STRUCT_initialization(REGs61);
    //         STRUCT_initialization(REGs62);
    //         STRUCT_initialization(REGs63);
    //         STRUCT_initialization(REGs64);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs1, STATE, 1);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs2, STATE, 2);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs3, STATE, 3);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs4, STATE, 4);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs5, STATE, 5);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs6, STATE, 6);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs7, STATE, 7);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs8, STATE, 8);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs9, STATE, 9);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs10, STATE, 10);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs11, STATE, 11);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs12, STATE, 12);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs13, STATE, 13);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs14, STATE, 14);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs15, STATE, 15);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs16, STATE, 16);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs17, STATE, 17);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs18, STATE, 18);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs19, STATE, 19);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs20, STATE, 20);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs21, STATE, 21);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs22, STATE, 22);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs23, STATE, 23);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs24, STATE, 24);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs25, STATE, 25);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs26, STATE, 26);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs27, STATE, 27);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs28, STATE, 28);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs29, STATE, 29);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs30, STATE, 30);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs31, STATE, 31);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs32, STATE, 32);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs33, STATE, 0);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs34, STATE, 1);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs35, STATE, 2);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs36, STATE, 3);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs37, STATE, 4);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs38, STATE, 5);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs39, STATE, 6);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs40, STATE, 7);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs41, STATE, 8);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs42, STATE, 9);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs43, STATE, 10);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs44, STATE, 11);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs45, STATE, 12);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs46, STATE, 13);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs47, STATE, 14);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs48, STATE, 15);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs49, STATE, 16);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs50, STATE, 17);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs51, STATE, 18);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs52, STATE, 19);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs53, STATE, 20);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs54, STATE, 21);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs55, STATE, 22);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs56, STATE, 23);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs57, STATE, 24);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs58, STATE, 25);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs59, STATE, 26);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs60, STATE, 27);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs61, STATE, 28);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs62, STATE, 29);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs63, STATE, 30);
    //         LUT_SR_RNG_r32_init_with_state_2nd(REGs64, STATE, 31);
    //         while(1){
    //             //[[intel::fpga_register]] fixed_point_delta_0_25 ac_fixed_out[2*num_p];
    //             fixed_point_delta_0_25 ac_fixed_out1 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out2 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out3 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out4 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out5 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out6 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out7 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out8 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out9 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out10 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out11 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out12 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out13 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out14 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out15 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out16 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out17 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out18 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out19 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out20 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out21 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out22 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out23 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out24 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out25 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out26 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out27 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out28 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out29 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out30 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out31 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out32 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out33 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out34 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out35 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out36 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out37 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out38 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out39 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out40 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out41 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out42 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out43 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out44 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out45 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out46 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out47 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out48 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out49 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out50 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out51 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out52 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out53 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out54 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out55 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out56 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out57 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out58 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out59 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out60 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out61 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out62 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out63 = 0.0f;
    //             fixed_point_delta_0_25 ac_fixed_out64 = 0.0f;

    //             [[intel::fpga_register]] float output[2*num_p];
    //             [[intel::fpga_register]] float tri_out_ini[num_p];

    //             //for REGs1
    //             LUT_SR_rng_r32_XORs(REGs1);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs1);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out1[i] = REGs1.rng_out[i];
    //             }
    //             output[0] = ac_fixed_out1.to_float();
    //             SR_extension(REGs1);
    //             //for REGs2
    //             LUT_SR_rng_r32_XORs(REGs2);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs2);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out2[i] = REGs2.rng_out[i];
    //             }
    //             output[1] = ac_fixed_out2.to_float();
    //             SR_extension(REGs2);
    //             //for REGs3
    //             LUT_SR_rng_r32_XORs(REGs3);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs3);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out3[i] = REGs3.rng_out[i];
    //             }
    //             output[2] = ac_fixed_out3.to_float();
    //             SR_extension(REGs3);
    //             //for REGs4
    //             LUT_SR_rng_r32_XORs(REGs4);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs4);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out4[i] = REGs4.rng_out[i];
    //             }
    //             output[3] = ac_fixed_out4.to_float();
    //             SR_extension(REGs4);
    //             //for REGs5
    //             LUT_SR_rng_r32_XORs(REGs5);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs5);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out5[i] = REGs5.rng_out[i];
    //             }
    //             output[4] = ac_fixed_out5.to_float();
    //             SR_extension(REGs5);
    //             //for REGs6
    //             LUT_SR_rng_r32_XORs(REGs6);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs6);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out6[i] = REGs6.rng_out[i];
    //             }
    //             output[5] = ac_fixed_out6.to_float();
    //             SR_extension(REGs6);
    //             //for REGs7
    //             LUT_SR_rng_r32_XORs(REGs7);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs7);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out7[i] = REGs7.rng_out[i];
    //             }
    //             output[6] = ac_fixed_out7.to_float();
    //             SR_extension(REGs7);
    //             //for REGs8
    //             LUT_SR_rng_r32_XORs(REGs8);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs8);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out8[i] = REGs8.rng_out[i];
    //             }
    //             output[7] = ac_fixed_out8.to_float();
    //             SR_extension(REGs8);
    //             //for REGs9
    //             LUT_SR_rng_r32_XORs(REGs9);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs9);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out9[i] = REGs9.rng_out[i];
    //             }
    //             output[8] = ac_fixed_out9.to_float();
    //             SR_extension(REGs9);
    //             //for REGs10
    //             LUT_SR_rng_r32_XORs(REGs10);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs10);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out10[i] = REGs10.rng_out[i];
    //             }
    //             output[9] = ac_fixed_out10.to_float();
    //             SR_extension(REGs10);
    //             //for REGs11
    //             LUT_SR_rng_r32_XORs(REGs11);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs11);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out11[i] = REGs11.rng_out[i];
    //             }
    //             output[10] = ac_fixed_out11.to_float();
    //             SR_extension(REGs11);
    //             //for REGs12
    //             LUT_SR_rng_r32_XORs(REGs12);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs12);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out12[i] = REGs12.rng_out[i];
    //             }
    //             output[11] = ac_fixed_out12.to_float();
    //             SR_extension(REGs12);
    //             //for REGs13
    //             LUT_SR_rng_r32_XORs(REGs13);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs13);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out13[i] = REGs13.rng_out[i];
    //             }
    //             output[12] = ac_fixed_out13.to_float();
    //             SR_extension(REGs13);
    //             //for REGs14
    //             LUT_SR_rng_r32_XORs(REGs14);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs14);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out14[i] = REGs14.rng_out[i];
    //             }
    //             output[13] = ac_fixed_out14.to_float();
    //             SR_extension(REGs14);
    //             //for REGs15
    //             LUT_SR_rng_r32_XORs(REGs15);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs15);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out15[i] = REGs15.rng_out[i];
    //             }
    //             output[14] = ac_fixed_out15.to_float();
    //             SR_extension(REGs15);
    //             //for REGs16
    //             LUT_SR_rng_r32_XORs(REGs16);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs16);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out16[i] = REGs16.rng_out[i];
    //             }
    //             output[15] = ac_fixed_out16.to_float();
    //             SR_extension(REGs16);
    //             //for REGs17
    //             LUT_SR_rng_r32_XORs(REGs17);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs17);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out17[i] = REGs17.rng_out[i];
    //             }
    //             output[16] = ac_fixed_out17.to_float();
    //             SR_extension(REGs17);
    //             //for REGs18
    //             LUT_SR_rng_r32_XORs(REGs18);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs18);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out18[i] = REGs18.rng_out[i];
    //             }
    //             output[17] = ac_fixed_out18.to_float();
    //             SR_extension(REGs18);
    //             //for REGs19
    //             LUT_SR_rng_r32_XORs(REGs19);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs19);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out19[i] = REGs19.rng_out[i];
    //             }
    //             output[18] = ac_fixed_out19.to_float();
    //             SR_extension(REGs19);
    //             //for REGs20
    //             LUT_SR_rng_r32_XORs(REGs20);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs20);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out20[i] = REGs20.rng_out[i];
    //             }
    //             output[19] = ac_fixed_out20.to_float();
    //             SR_extension(REGs20);
    //             //for REGs21
    //             LUT_SR_rng_r32_XORs(REGs21);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs21);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out21[i] = REGs21.rng_out[i];
    //             }
    //             output[20] = ac_fixed_out21.to_float();
    //             SR_extension(REGs21);
    //             //for REGs22
    //             LUT_SR_rng_r32_XORs(REGs22);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs22);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out22[i] = REGs22.rng_out[i];
    //             }
    //             output[21] = ac_fixed_out22.to_float();
    //             SR_extension(REGs22);
    //             //for REGs23
    //             LUT_SR_rng_r32_XORs(REGs23);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs23);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out23[i] = REGs23.rng_out[i];
    //             }
    //             output[22] = ac_fixed_out23.to_float();
    //             SR_extension(REGs23);
    //             //for REGs24
    //             LUT_SR_rng_r32_XORs(REGs24);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs24);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out24[i] = REGs24.rng_out[i];
    //             }
    //             output[23] = ac_fixed_out24.to_float();
    //             SR_extension(REGs24);
    //             //for REGs25
    //             LUT_SR_rng_r32_XORs(REGs25);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs25);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out25[i] = REGs25.rng_out[i];
    //             }
    //             output[24] = ac_fixed_out25.to_float();
    //             SR_extension(REGs25);
    //             //for REGs26
    //             LUT_SR_rng_r32_XORs(REGs26);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs26);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out26[i] = REGs26.rng_out[i];
    //             }
    //             output[25] = ac_fixed_out26.to_float();
    //             SR_extension(REGs26);
    //             //for REGs27
    //             LUT_SR_rng_r32_XORs(REGs27);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs27);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out27[i] = REGs27.rng_out[i];
    //             }
    //             output[26] = ac_fixed_out27.to_float();
    //             SR_extension(REGs27);
    //             //for REGs28
    //             LUT_SR_rng_r32_XORs(REGs28);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs28);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out28[i] = REGs28.rng_out[i];
    //             }
    //             output[27] = ac_fixed_out28.to_float();
    //             SR_extension(REGs28);
    //             //for REGs29
    //             LUT_SR_rng_r32_XORs(REGs29);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs29);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out29[i] = REGs29.rng_out[i];
    //             }
    //             output[28] = ac_fixed_out29.to_float();
    //             SR_extension(REGs29);
    //             //for REGs30
    //             LUT_SR_rng_r32_XORs(REGs30);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs30);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out30[i] = REGs30.rng_out[i];
    //             }
    //             output[29] = ac_fixed_out30.to_float();
    //             SR_extension(REGs30);
    //             //for REGs31
    //             LUT_SR_rng_r32_XORs(REGs31);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs31);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out31[i] = REGs31.rng_out[i];
    //             }
    //             output[30] = ac_fixed_out31.to_float();
    //             SR_extension(REGs31);
    //             //for REGs32
    //             LUT_SR_rng_r32_XORs(REGs32);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs32);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out32[i] = REGs32.rng_out[i];
    //             }
    //             output[31] = ac_fixed_out32.to_float();
    //             SR_extension(REGs32);
    //             //for REGs33
    //             LUT_SR_rng_r32_XORs(REGs33);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs33);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out33[i] = REGs33.rng_out[i];
    //             }
    //             output[32] = ac_fixed_out33.to_float();
    //             SR_extension(REGs33);
    //             //for REGs34
    //             LUT_SR_rng_r32_XORs(REGs34);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs34);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out34[i] = REGs34.rng_out[i];
    //             }
    //             output[33] = ac_fixed_out34.to_float();
    //             SR_extension(REGs34);
    //             //for REGs35
    //             LUT_SR_rng_r32_XORs(REGs35);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs35);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out35[i] = REGs35.rng_out[i];
    //             }
    //             output[34] = ac_fixed_out35.to_float();
    //             SR_extension(REGs35);
    //             //for REGs36
    //             LUT_SR_rng_r32_XORs(REGs36);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs36);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out36[i] = REGs36.rng_out[i];
    //             }
    //             output[35] = ac_fixed_out36.to_float();
    //             SR_extension(REGs36);
    //             //for REGs37
    //             LUT_SR_rng_r32_XORs(REGs37);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs37);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out37[i] = REGs37.rng_out[i];
    //             }
    //             output[36] = ac_fixed_out37.to_float();
    //             SR_extension(REGs37);
    //             //for REGs38
    //             LUT_SR_rng_r32_XORs(REGs38);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs38);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out38[i] = REGs38.rng_out[i];
    //             }
    //             output[37] = ac_fixed_out38.to_float();
    //             SR_extension(REGs38);
    //             //for REGs39
    //             LUT_SR_rng_r32_XORs(REGs39);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs39);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out39[i] = REGs39.rng_out[i];
    //             }
    //             output[38] = ac_fixed_out39.to_float();
    //             SR_extension(REGs39);
    //             //for REGs40
    //             LUT_SR_rng_r32_XORs(REGs40);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs40);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out40[i] = REGs40.rng_out[i];
    //             }
    //             output[39] = ac_fixed_out40.to_float();
    //             SR_extension(REGs40);
    //             //for REGs41
    //             LUT_SR_rng_r32_XORs(REGs41);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs41);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out41[i] = REGs41.rng_out[i];
    //             }
    //             output[40] = ac_fixed_out41.to_float();
    //             SR_extension(REGs41);
    //             //for REGs42
    //             LUT_SR_rng_r32_XORs(REGs42);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs42);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out42[i] = REGs42.rng_out[i];
    //             }
    //             output[41] = ac_fixed_out42.to_float();
    //             SR_extension(REGs42);
    //             //for REGs43
    //             LUT_SR_rng_r32_XORs(REGs43);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs43);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out43[i] = REGs43.rng_out[i];
    //             }
    //             output[42] = ac_fixed_out43.to_float();
    //             SR_extension(REGs43);
    //             //for REGs44
    //             LUT_SR_rng_r32_XORs(REGs44);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs44);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out44[i] = REGs44.rng_out[i];
    //             }
    //             output[43] = ac_fixed_out44.to_float();
    //             SR_extension(REGs44);
    //             //for REGs45
    //             LUT_SR_rng_r32_XORs(REGs45);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs45);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out45[i] = REGs45.rng_out[i];
    //             }
    //             output[44] = ac_fixed_out45.to_float();
    //             SR_extension(REGs45);
    //             //for REGs46
    //             LUT_SR_rng_r32_XORs(REGs46);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs46);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out46[i] = REGs46.rng_out[i];
    //             }
    //             output[45] = ac_fixed_out46.to_float();
    //             SR_extension(REGs46);
    //             //for REGs47
    //             LUT_SR_rng_r32_XORs(REGs47);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs47);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out47[i] = REGs47.rng_out[i];
    //             }
    //             output[46] = ac_fixed_out47.to_float();
    //             SR_extension(REGs47);
    //             //for REGs48
    //             LUT_SR_rng_r32_XORs(REGs48);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs48);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out48[i] = REGs48.rng_out[i];
    //             }
    //             output[47] = ac_fixed_out48.to_float();
    //             SR_extension(REGs48);
    //             //for REGs49
    //             LUT_SR_rng_r32_XORs(REGs49);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs49);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out49[i] = REGs49.rng_out[i];
    //             }
    //             output[48] = ac_fixed_out49.to_float();
    //             SR_extension(REGs49);
    //             //for REGs50
    //             LUT_SR_rng_r32_XORs(REGs50);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs50);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out50[i] = REGs50.rng_out[i];
    //             }
    //             output[49] = ac_fixed_out50.to_float();
    //             SR_extension(REGs50);
    //             //for REGs51
    //             LUT_SR_rng_r32_XORs(REGs51);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs51);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out51[i] = REGs51.rng_out[i];
    //             }
    //             output[50] = ac_fixed_out51.to_float();
    //             SR_extension(REGs51);
    //             //for REGs52
    //             LUT_SR_rng_r32_XORs(REGs52);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs52);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out52[i] = REGs52.rng_out[i];
    //             }
    //             output[51] = ac_fixed_out52.to_float();
    //             SR_extension(REGs52);
    //             //for REGs53
    //             LUT_SR_rng_r32_XORs(REGs53);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs53);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out53[i] = REGs53.rng_out[i];
    //             }
    //             output[52] = ac_fixed_out53.to_float();
    //             SR_extension(REGs53);
    //             //for REGs54
    //             LUT_SR_rng_r32_XORs(REGs54);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs54);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out54[i] = REGs54.rng_out[i];
    //             }
    //             output[53] = ac_fixed_out54.to_float();
    //             SR_extension(REGs54);
    //             //for REGs55
    //             LUT_SR_rng_r32_XORs(REGs55);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs55);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out55[i] = REGs55.rng_out[i];
    //             }
    //             output[54] = ac_fixed_out55.to_float();
    //             SR_extension(REGs55);
    //             //for REGs56
    //             LUT_SR_rng_r32_XORs(REGs56);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs56);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out56[i] = REGs56.rng_out[i];
    //             }
    //             output[55] = ac_fixed_out56.to_float();
    //             SR_extension(REGs56);
    //             //for REGs57
    //             LUT_SR_rng_r32_XORs(REGs57);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs57);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out57[i] = REGs57.rng_out[i];
    //             }
    //             output[56] = ac_fixed_out57.to_float();
    //             SR_extension(REGs57);
    //             //for REGs58
    //             LUT_SR_rng_r32_XORs(REGs58);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs58);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out58[i] = REGs58.rng_out[i];
    //             }
    //             output[57] = ac_fixed_out58.to_float();
    //             SR_extension(REGs58);
    //             //for REGs59
    //             LUT_SR_rng_r32_XORs(REGs59);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs59);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out59[i] = REGs59.rng_out[i];
    //             }
    //             output[58] = ac_fixed_out59.to_float();
    //             SR_extension(REGs59);
    //             //for REGs60
    //             LUT_SR_rng_r32_XORs(REGs60);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs60);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out60[i] = REGs60.rng_out[i];
    //             }
    //             output[59] = ac_fixed_out60.to_float();
    //             SR_extension(REGs60);
    //             //for REGs61
    //             LUT_SR_rng_r32_XORs(REGs61);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs61);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out61[i] = REGs61.rng_out[i];
    //             }
    //             output[60] = ac_fixed_out61.to_float();
    //             SR_extension(REGs61);
    //             //for REGs62
    //             LUT_SR_rng_r32_XORs(REGs62);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs62);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out62[i] = REGs62.rng_out[i];
    //             }
    //             output[61] = ac_fixed_out62.to_float();
    //             SR_extension(REGs62);
    //             //for REGs63
    //             LUT_SR_rng_r32_XORs(REGs63);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs63);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out63[i] = REGs63.rng_out[i];
    //             }
    //             output[62] = ac_fixed_out63.to_float();
    //             SR_extension(REGs63);
    //             //for REGs64
    //             LUT_SR_rng_r32_XORs(REGs64);
    //             //Permutation of the output
    //             LUT_SR_rng_r32_permutation(REGs64);
    //             #pragma unroll
    //             for(int i=0; i<r; i++){
    //                 ac_fixed_out64[i] = REGs64.rng_out[i];
    //             }
    //             output[63] = ac_fixed_out64.to_float();
    //             SR_extension(REGs64);

    //             //using U1 - U2 to represent the triangle distribution with mean = 0, and delta = 0.25
    //             #pragma unroll
    //             for(int i=0; i<num_p; i++){
    //                 tri_out_ini[i] = output[2*i] - output[2*i+1];
    //             }
    //             //set different mean for these triangle distributions
    //             #pragma unroll
    //             for(int i=0; i<num_p; i++){
    //                 tri_out_ini[i] = tri_out_ini[i] + delta*(i-num_p/2);
    //             }
    //             //pipe out
    //             size_t index = 0;
    //             fpga_tools::UnrolledLoop<num_p>([&tri_out_ini, &index](auto idx){
    //                 Pipe_out::template PipeAt<idx>::write(tri_out_ini[index++]);
    //             });
                
    //         }
    //     }
    // };



    // Pipe_in_rng_01: single pipe, float
    // Pipe_in_rng_0n: single pipe, ac_int<5, false>
    // Pipe_out: single pipe, ac_int<5, false>
    template <typename Pipe_in_rng_01, typename Pipe_in_rng_0n, typename Pipe_out>
    struct AR_Walker_Alias_Table{
        void operator()() const{
            [[intel::fpga_register]] float p_threshold[num_p] = {
                0.00013383f, 0.000352596f, 0.000872683f, 0.00202905f,
                0.00443185f, 0.009093560f, 0.017528300f, 0.03173970f,
                0.05399100f, 0.086277300f, 0.129518000f, 0.18264900f,
                0.24197100f, 0.301137000f, 0.352065000f, 0.38666800f,
                0.39894200f, 0.386668000f, 0.352065000f, 0.30113700f,
                0.24197100f, 0.182649000f, 0.129518000f, 0.08627730f,
                0.05399100f, 0.031739700f, 0.017528300f, 0.00909356f,
                0.00443185f, 0.002029050f, 0.000872683f, 0.000352596f
            };
            // [[intel::fpga_register]] float p_threshold[num_p] = {
            //     1.48515e-10, 2.19e-09, 2.71469e-08, 2.82878e-07, 2.47787e-06, 1.82456e-05, 0.000112938, 0.000587659, 0.00257046, 0.00945147, 0.0292138, 0.0759066, 0.165795, 0.304415, 0.469853, 0.609621, 0.664904, 0.609621, 0.469853, 0.304415, 0.165795, 0.0759066, 0.0292138, 0.00945147, 0.00257046, 0.000587659, 0.000112938, 1.82456e-05, 2.47787e-06, 2.82878e-07, 2.71469e-08, 2.19e-09
            // };
            [[intel::fpga_register]] int_5_bit a[num_p] = {
                0,  1,  2,  3, 
                4,  5,  6,  7,
                8,  9,  10, 11,
                12, 13, 14, 15, 
                16, 17, 18, 19,
                20, 21, 22, 23,
                24, 25, 26, 27,
                28, 29, 30, 31,
            };
            //Initialize the two look up tables
            while(1){
                int_5_bit idx = Pipe_in_rng_0n::read();
                int_5_bit out = 0;
                float p_from_rng = Pipe_in_rng_01::read();
                out = (p_from_rng > p_threshold[idx]) ? a[idx] : idx;
                Pipe_out::write(out);
            }
        }
    };

    template <typename Pipe_in_alias_table, typename Pipe_in_triang_distr, typename Pipe_out>
    struct AR_Gaussian_RNG_single{
        void operator()() const{
            while(1){
                [[intel::fpga_register]] float trian_p_in[num_p];
                int_5_bit idx = Pipe_in_alias_table::read();
                size_t index = 0;
                fpga_tools::UnrolledLoop<num_p>([&trian_p_in, &index](auto idx_1){
                    trian_p_in[index++] = Pipe_in_triang_distr::template PipeAt<idx_1>::read();
                });
                float output = trian_p_in[idx];
                Pipe_out::write(output);
            }
        }

    };

    template <typename Pipe_out1, typename Pipe_out2>
    struct AR_RNG{
        void operator()() const{
            while(1){
                
            }
        }
    };
}

#endif