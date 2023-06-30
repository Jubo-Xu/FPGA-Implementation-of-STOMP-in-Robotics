#ifndef __DETERMINATION_HPP__
#define __DETERMINATION_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

#define N_determine     16
#define DoF_determine   3
#define k_determine     4
#define Threshold       0.5
#define Num_end_sig     2
#define end_sig_idx     0
#define itr_determine   50

namespace Determine_and_Connections{
    
    // pipe_in_cost_obstacle: single pipe
    // pipe_in_cost_smooth: single pipe
    // pipe_in_theta: pipearray(DoF)
    // pipe_out_new_theta: pipearray(DoF)
    // pipe_out_final_theta: single pipe
    // pipe_out_end: pipearray(number of end signal needed)
    template <typename Pipe_in_cost_obstacle, typename Pipe_in_cost_smooth, typename Pipe_in_theta, typename Pipe_out_new_theta, typename Pipe_out_final_theta, typename Pipe_out_end>
    struct AR_Determine_Block{
        void operator()() const{
            int COUNT = 0;
            while(1){
                [[intel::fpga_register]] float IN_theta[DoF_determine][N_determine];
                [[intel::fpga_register]] float sum_obs_cost = 0.0f;
                [[intel::fpga_register]] float total_cost = 0.0f;

                // calculate the sum of q for all i belongs to N
                for(int i=0; i<N_determine; i++){
                    sum_obs_cost += Pipe_in_cost_obstacle::read();
                    // now read the input theta
                    size_t index_in_theta = 0;
                    fpga_tools::UnrolledLoop<DoF_determine>([&index_in_theta, &IN_theta, &i](auto idx_1){
                        IN_theta[index_in_theta++][i] = Pipe_in_theta::template PipeAt<idx_1>::read();
                    });
                }

                // find the total cost
                total_cost = sum_obs_cost + Pipe_in_cost_smooth::read();
            
                // determine whether the iteration is finished
                //(total_cost < Threshold) || (total_cost == Threshold)
                //COUNT == itr_determine
                if(COUNT == itr_determine){
                    fpga_tools::UnrolledLoop<Num_end_sig>([&](auto idx_3){
                        Pipe_out_end::template PipeAt<idx_3>::write(0);
                    });
                    for(int i=0; i<DoF_determine*N_determine; i++){
                        Pipe_out_final_theta::write(IN_theta[i%DoF_determine][i/DoF_determine]);
                    }
                    COUNT = 0;
                }
                else{
                    fpga_tools::UnrolledLoop<Num_end_sig>([&](auto idx_4){
                        Pipe_out_end::template PipeAt<idx_4>::write(1);
                    });
                    for(int i=0; i<N_determine; i++){
                        // output the new theta to the first block
                        size_t index_out_theta1 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_out_theta1, &IN_theta, &i](auto idx_2){
                            Pipe_out_new_theta::template PipeAt<idx_2>::write(IN_theta[index_out_theta1++][i]);
                        });
                    }
                    COUNT++;
                }

                
            }
        }

    };

    // Pipe_in_theta_initial: single pipe
    // Pipe_in_theta_loop: pipe_array(DoF)
    // Pipe_in_end: pipe_array(Number of end signal needed)
    // Pipe_out: pipe_array(DoF)
    template <typename Pipe_in_theta_initial, typename Pipe_in_theta_loop, typename Pipe_in_end, typename Pipe_out1, typename Pipe_out2, typename Pipe_out3, typename Pipe_out4>
    struct AR_input_block{
        void operator()() const {
            [[intel::fpga_register]] int COUNT = 0;
            while(1){
                if(COUNT == 0){
                    for(int i=0; i<N_determine; i++){
                        [[intel::fpga_register]] float INPUT_theta_initial[DoF_determine];
                        for(int j=0; j<DoF_determine; j++){
                            INPUT_theta_initial[j] = Pipe_in_theta_initial::read();
                        }
                        size_t index_1 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_1, &INPUT_theta_initial](auto idx_1){
                            Pipe_out1::template PipeAt<idx_1>::write(INPUT_theta_initial[index_1++]);
                        });
                        size_t index_2 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_2, &INPUT_theta_initial](auto idx_4){
                            Pipe_out2::template PipeAt<idx_4>::write(INPUT_theta_initial[index_2++]);
                        });
                        size_t index_3 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_3, &INPUT_theta_initial](auto idx_6){
                            Pipe_out3::template PipeAt<idx_6>::write(INPUT_theta_initial[index_3++]);
                        });
                        size_t index_4 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_4, &INPUT_theta_initial](auto idx_8){
                            Pipe_out4::template PipeAt<idx_8>::write(INPUT_theta_initial[index_4++]);
                        });
                    }
                    COUNT++;
                }
                else{
                    for(int i=0; i<N_determine; i++){
                        [[intel::fpga_register]] float INPUT_theta_loop[DoF_determine];
                        size_t index_in = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_in, &INPUT_theta_loop](auto idx_2){
                            INPUT_theta_loop[index_in++] = Pipe_in_theta_loop::template PipeAt<idx_2>::read();
                        });
                        size_t index_out_1 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_out_1, &INPUT_theta_loop](auto idx_3){
                            Pipe_out1::template PipeAt<idx_3>::write(INPUT_theta_loop[index_out_1++]);
                        });
                        size_t index_out_2 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_out_2, &INPUT_theta_loop](auto idx_5){
                            Pipe_out2::template PipeAt<idx_5>::write(INPUT_theta_loop[index_out_2++]);
                        });
                        size_t index_out_3 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_out_3, &INPUT_theta_loop](auto idx_7){
                            Pipe_out3::template PipeAt<idx_7>::write(INPUT_theta_loop[index_out_3++]);
                        });
                        size_t index_out_4 = 0;
                        fpga_tools::UnrolledLoop<DoF_determine>([&index_out_4, &INPUT_theta_loop](auto idx_9){
                            Pipe_out4::template PipeAt<idx_9>::write(INPUT_theta_loop[index_out_4++]);
                        });
                    }
                    COUNT++;
                }
                if(Pipe_in_end::template PipeAt<end_sig_idx>::read() == 0){
                    COUNT = 0;
                }
            }
        }
    };

    template <typename Pipe_in_theta, typename Pipe_in_RNG, typename Pipe_out>
    struct AR_Input_2_kNoisy{
        void operator()() const {
            while(1){
                [[intel::fpga_register]] float Input_theta[DoF_determine];
                [[intel::fpga_register]] float Input_RNG[DoF_determine*k_determine];
                [[intel::fpga_register]] float OUT[DoF_determine*k_determine];
                size_t index_1 = 0;
                fpga_tools::UnrolledLoop<DoF_determine>([&index_1, &Input_theta](auto idx_1){
                    Input_theta[index_1++] = Pipe_in_theta::template PipeAt<idx_1>::read();
                });
                size_t index_2 = 0;
                fpga_tools::UnrolledLoop<DoF_determine*k_determine>([&index_2, &Input_RNG](auto idx_2){
                    Input_RNG[index_2++] = Pipe_in_RNG::template PipeAt<idx_2>::read();
                });

                #pragma unroll
                for(int i=0; i<k_determine; i++){
                    #pragma unroll
                    for(int j=0; j<DoF_determine; j++){
                        OUT[j*k_determine + i] = Input_theta[j] + Input_RNG[j*k_determine + i];
                    }
                }
                size_t index_3 = 0;
                fpga_tools::UnrolledLoop<DoF_determine*k_determine>([&index_3, &OUT](auto idx_3){
                    Pipe_out::template PipeAt<idx_3>::write(OUT[index_3++]);
                });
            }
        }
    };
}

#endif