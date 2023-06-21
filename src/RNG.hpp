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
    #define sigma    1
    #define N_stat 32

    using int_1_bit = ac_int<1, false>;
    using int_6_bit = ac_int<5, false>;
    using int_5_bit = ac_int<5, false>;
    using fixed_point_4_28 = ac_fixed<r, INT, true>;
    using fixed_point_delta_0_25 = ac_fixed<r+2, -2, true>;

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

    template <typename Pipe_out>
    struct AR_rng_n1024_r32_t5_k32_s1c48_delta_0_25_test{
        void operator()() const{
            STRUCT_of_LUT_SR_RNG_r32 REGs;
            STATE_regs_of_uniform_rng STATE;
            STRUCT_initialization(REGs);
            LUT_SR_RNG_r32_init_with_state_2nd(REGs, STATE, 31);
            while(1){
                fixed_point_delta_0_25 ac_fixed_out = 0.0f;
                LUT_SR_rng_r32_XORs(REGs);
                //Permutation of the output
                LUT_SR_rng_r32_permutation(REGs);
                //transform to ac_fixed type and then transfer to float
                ac_fixed_out[r+2-1] = REGs.rng_out[r-1];
                #pragma unroll
                for(int i=0; i<r-1; i++){
                    ac_fixed_out[i] = REGs.rng_out[i];
                }
                float output = ac_fixed_out.to_float();
                Pipe_out::write(output);
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
            for(int i=0; i<2*num_p; i++){
                STRUCT_initialization(REGs[i]);
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
                    (ac_fixed_out[i])[r+2-1] = REGs[i].rng_out[r-1];
                    #pragma unroll
                    for(int j=0; j<r-1; j++){
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