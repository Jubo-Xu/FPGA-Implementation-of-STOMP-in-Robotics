#ifndef __COSTSMOOTH_HPP__
#define __COSTSMOOTH_HPP__

#include <sycl/sycl.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed.hpp>
#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ac_fixed_math.hpp>
//#include </opt/intel/oneapi/compiler/2023.1.0/linux/lib/oclfpga/include/sycl/ext/intel/ac_types/ap_float_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

#define kSize_smooth    5
#define DoF_smooth      3
#define end_sig_idx_smooth 1


namespace smooth{
    
template<typename PipeIn, typename PipeOutMMul_1, typename PipeOutMMul_2>
struct ARMMul{
    void operator()() const {
        
    [[intel::fpga_register]]float F_matrix[kSize_smooth*kSize_smooth] = 
    {0.5357f, 0.7143f, 0.6429f, 0.4286f, 0.1786f,
     0.7143f, 1.4286f, 1.4286f, 1.0000f, 0.4286f,
     0.6429f, 1.4286f, 1.8571f, 1.4286f, 0.6429f,
     0.4286f, 1.0000f, 1.4286f, 1.4286f, 0.7143f,
     0.1786f, 0.4286f, 0.6429f, 0.7143f, 0.5357f}; 
    // [[intel::fpga_register]]float F_matrix[kSize_smooth*kSize_smooth] = {
    //   0.0016, 0.0029, 0.0038, 0.0044, 0.0047, 0.0048, 0.0046, 0.0043, 0.0039, 0.0034, 0.0028, 0.0022, 0.0016, 0.0010, 0.0005, 0.0002,
    //   0.0029, 0.0067, 0.0095, 0.0114, 0.0125, 0.0129, 0.0127, 0.0119, 0.0108, 0.0094, 0.0078, 0.0061, 0.0044, 0.0028, 0.0015, 0.0005,
    //   0.0038, 0.0095, 0.0154, 0.0194, 0.0218, 0.0229, 0.0227, 0.0216, 0.0197, 0.0172, 0.0143, 0.0112, 0.0081, 0.0053, 0.0028, 0.0010,
    //   0.0044, 0.0114, 0.0194, 0.0265, 0.0310, 0.0333, 0.0336, 0.0323, 0.0297, 0.0260, 0.0218, 0.0171, 0.0125, 0.0081, 0.0044, 0.0016,
    //   0.0047, 0.0125, 0.0218, 0.0310, 0.0385, 0.0426, 0.0439, 0.0428, 0.0397, 0.0352, 0.0296, 0.0234, 0.0171, 0.0112, 0.0061, 0.0022,
    //   0.0048, 0.0129, 0.0229, 0.0333, 0.0426, 0.0495, 0.0525, 0.0521, 0.0490, 0.0438, 0.0372, 0.0296, 0.0218, 0.0143, 0.0078, 0.0028,
    //   0.0046, 0.0127, 0.0227, 0.0336, 0.0439, 0.0525, 0.0580, 0.0590, 0.0565, 0.0512, 0.0438, 0.0352, 0.0260, 0.0172, 0.0094, 0.0034,
    //   0.0043, 0.0119, 0.0216, 0.0323, 0.0428, 0.0521, 0.0590, 0.0625, 0.0613, 0.0565, 0.0490, 0.0397, 0.0297, 0.0197, 0.0108, 0.0039,
    //   0.0039, 0.0108, 0.0197, 0.0297, 0.0397, 0.0490, 0.0565, 0.0613, 0.0625, 0.0590, 0.0521, 0.0428, 0.0323, 0.0216, 0.0119, 0.0043,
    //   0.0034, 0.0094, 0.0172, 0.0260, 0.0352, 0.0438, 0.0512, 0.0565, 0.0590, 0.0580, 0.0525, 0.0439, 0.0336, 0.0227, 0.0127, 0.0046,
    //   0.0028, 0.0078, 0.0143, 0.0218, 0.0296, 0.0372, 0.0438, 0.0490, 0.0521, 0.0525, 0.0495, 0.0426, 0.0333, 0.0229, 0.0129, 0.0048,
    //   0.0022, 0.0061, 0.0112, 0.0171, 0.0234, 0.0296, 0.0352, 0.0397, 0.0428, 0.0439, 0.0426, 0.0385, 0.0310, 0.0218, 0.0125, 0.0047,
    //   0.0016, 0.0044, 0.0081, 0.0125, 0.0171, 0.0218, 0.0260, 0.0297, 0.0323, 0.0336, 0.0333, 0.0310, 0.0265, 0.0194, 0.0114, 0.0044,
    //   0.0010, 0.0028, 0.0053, 0.0081, 0.0112, 0.0143, 0.0172, 0.0197, 0.0216, 0.0227, 0.0229, 0.0218, 0.0194, 0.0154, 0.0095, 0.0038,
    //   0.0005, 0.0015, 0.0028, 0.0044, 0.0061, 0.0078, 0.0094, 0.0108, 0.0119, 0.0127, 0.0129, 0.0125, 0.0114, 0.0095, 0.0067, 0.0029,
    //   0.0002, 0.0005, 0.0010, 0.0016, 0.0022, 0.0028, 0.0034, 0.0039, 0.0043, 0.0046, 0.0048, 0.0047, 0.0044, 0.0038, 0.0029, 0.0016
    // };
      
        while(1){ 
          // [[intel::fpga_register]] float A_out[DoF_smooth][kSize_smooth] = {
          //   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
          //   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
          //   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
          // }; 
          [[intel::fpga_register]] float A_out[DoF_smooth][kSize_smooth];
          [[intel::fpga_register]] float B_out[DoF_smooth][kSize_smooth];
          [[intel::fpga_register]] float Delta_theta_in[DoF_smooth][kSize_smooth];
          [[intel::fpga_register]] float Out_add[DoF_smooth];
          #pragma unroll
          for(int i=0; i<DoF_smooth; i++){
            Out_add[i] = 0.0f;
            #pragma unroll
            for(int j=0; j<kSize_smooth; j++){
              A_out[i][j] = 0.0f;
              B_out[i][j] = 0.0f;
              Delta_theta_in[i][j] = 0.0f;
            }
          }
          [[intel::fpga_register]] float val[DoF_smooth];
          for(int i = 0; i < kSize_smooth; i++){
              //PRINTF("ARMMUL flag 1 \n");
              size_t index_in_val = 0;
              fpga_tools::UnrolledLoop<DoF_smooth>([&index_in_val, &val](auto idx_1) {
                    val[index_in_val++] = PipeIn::template PipeAt<idx_1>::read();
                });
              #pragma unroll
              for(int k=0; k<DoF_smooth; k++){
                Delta_theta_in[k][i] = val[k];
              }
              //PRINTF("ARMMUL flag 2 \n");
              #pragma unroll
              for(int k=0; k<DoF_smooth; k++){
                  #pragma unroll 
                  for(int j = 0; j < kSize_smooth; j++){
                      val[k] = ext::intel::fpga_reg(val[k]); 
                      A_out[k][j] += val[k]*F_matrix[i*kSize_smooth+j]; 
                      //B_out[k][j] = A_out[k][j]*val[k];
                  }
              }
          }

          #pragma unroll
          for(int i=0; i<DoF_smooth; i++){
            #pragma unroll
            for(int j=0; j<kSize_smooth; j++){
              B_out[i][j] = A_out[i][j]*Delta_theta_in[i][j];
            }
          }
          #pragma unroll
          for(int i=0; i<DoF_smooth; i++){
            #pragma unroll
            for(int j=0; j<kSize_smooth; j++){
              Out_add[i] = ext::intel::fpga_reg(Out_add[i]) + ext::intel::fpga_reg(B_out[i][j]); 
            }
          }

          size_t index_out_2 = 0;
          fpga_tools::UnrolledLoop<DoF_smooth>([&Out_add, &index_out_2](auto idx_3){
              PipeOutMMul_2::template PipeAt<idx_3>::write(Out_add[index_out_2++]/(kSize_smooth*kSize_smooth));
          }); 

          for(int i = 0; i < kSize_smooth; i++){
            size_t index_out_1 = 0;
            
            fpga_tools::UnrolledLoop<DoF_smooth>([&i, & A_out, &index_out_1](auto idx_2){
                 PipeOutMMul_1::template PipeAt<idx_2>::write(A_out[index_out_1++][i]);
                 //PRINTF("ARMMUL flag 3 in the loop  \n"); 
              }); 
            // fpga_tools::UnrolledLoop<DoF_smooth>([&i, & B_out, &index_out_2](auto idx_3){
            //      PipeOutMMul_2::template PipeAt<idx_3>::write(B_out[index_out_2++][i]);
            //      //PRINTF("ARMMUL flag 3 in the loop  \n"); 
            //   }); 
                //  TUpdatePipe::PipeAt<0>::write(A_out[i]);
                //  PRINTF("ARMMUL flag 3 in the loop  \n"); 
              //A_out[i]  = 0; 
          }
        }
    }
};

template<typename PipeInInit, typename PipeInMMul, typename PipeOut1> 

//This block update the previous trajectorie base on the result output of the smoothness matrix multiplication delta = theta_previous + M*delta , refer to the README 
struct ARTUpdate{
  void operator()() const{

    while(1){
      [[intel::fpga_register]] float T_current[DoF_smooth][kSize_smooth];
      #pragma unroll
      for(int i=0; i<DoF_smooth; i++){
        #pragma unroll
        for(int j=0; j<kSize_smooth; j++){
          T_current[i][j] = 0.0f;
        }
      }
      //bool state_iter = PipeInState::template PipeAt<0>::read();
      //bool state_iter = PipeInState::read();
      //PRINTF("ARTUpdate flag 1 \n");
      // The state_iter condition indicate if it is the first iteration in which case the theta_previous is actually theta_initial and needs to be get from an external control block
      //if(state_iter == 1){
        for(int i = 0; i < kSize_smooth; i++){
           //PRINTF("ARTUpdate flag 3 \n");
          size_t index_in_theta_org = 0;
          fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_in_theta_org](auto idx_1){
             T_current[index_in_theta_org++][i] =  PipeInInit::template PipeAt<idx_1>::read(); //get the theta_initial
          });
        }
        for(int i = 0 ; i < kSize_smooth; i++){
          size_t index_in_matrix_out = 0;
          fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_in_matrix_out](auto idx_2){
             T_current[index_in_matrix_out++][i] += PipeInMMul::template PipeAt<idx_2>::read(); // update the trajectory 
          });
          //PRINTF("ARTUpdate flag 4 \n");
          size_t index_out_1 = 0;
          size_t index_out_2 = 0;
          fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_1](auto idx_3){
             PipeOut1::template PipeAt<idx_3>::write(T_current[index_out_1++][i]); //write the result to the pipeout1
          }); 
          // fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_2](auto idx_4){
          //    PipeOut2::template PipeAt<idx_4>::write(T_current[index_out_2++][i]); //write the result to the pipeout2
          // });
        }
      //}   
      //If it is not the first trajectory then theta_previous is stored form the previous iteration   
      // else{
      //   for(int i = 0 ; i < kSize_smooth ; i++){
      //      //PRINTF("ARTUpdate flag 5 \n");
      //     size_t index_in_matrix_out = 0;
      //     fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_in_matrix_out](auto idx_1){
      //        T_current[index_in_matrix_out++][i] += PipeInMMul::template PipeAt<idx_1>::read();//update base on the previous iteration
      //     });
      //     // ARConsumePipe::write(T_current[i]); 
      //     // PRINTF("ARTUpdate flag 6 \n");
      //     size_t index_out_1 = 0;
      //     size_t index_out_2 = 0;
      //     fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_1](auto idx_2){
      //         PipeOut1::template PipeAt<idx_2>::write(T_current[index_out_1++][i]);// write the result to the pipeout1
      //     }); 
      //     fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_2](auto idx_3){
      //         PipeOut2::template PipeAt<idx_3>::write(T_current[index_out_2++][i]);// write the result to the pipeout1
      //     }); 
        
      //   }
      // }    
    }
  }
};

template<typename PipeInInit, typename PipeOut >
struct ARSparseMul{
    void operator()() const
      {
        [[intel::fpga_register]] float A[DoF_smooth][5] = {
          {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
          {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
          {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
        };
        //float Out; 
        [[intel::fpga_register]] float Out[DoF_smooth];
        #pragma unroll
        for(int i=0; i<DoF_smooth; i++){
          Out[i] = 0.0f;
        }
        int COUNT = 0;
        
        
        while(1){
          //PRINTF("SPARSE flag 1 \n");
          int do_it = 1;
          if(do_it == 1){ //this block is only require once for the first iteration it is desable the rest of the time 

            for(int i=0; i<kSize_smooth+2; i++){
              if(COUNT<kSize_smooth){
                size_t index_1 = 0;
                fpga_tools::UnrolledLoop<DoF_smooth>([&A, &index_1](auto idx_1){
                    A[index_1++][0] =  PipeInInit::template PipeAt<idx_1>::read(); //get the theta_initial
                });
                #pragma unroll
                for(int k=0; k<DoF_smooth; k++){
                  Out[k] += A[k][2]*(A[k][0] +(-4) * A[k][1] + 6*A[k][2] + (-4)* A[k][3] + A[k][4]); // perfom the operation as explain in the README 
                  #pragma unroll 
                  for(int i = 0; i < 4 ; i++ ){ 
                    A[k][4-i] = ext::intel::fpga_reg(A[k][4-i-1]); // shift register operation    
                  }
                }
              }
              else{
                #pragma unroll
                for(int k=0; k<DoF_smooth; k++){
                  A[k][0] = 0.0f;
                  Out[k] += A[k][2]*(A[k][0] +(-4) * A[k][1] + 6*A[k][2] + (-4)* A[k][3] + A[k][4]); // perfom the operation as explain in the README 
                  #pragma unroll 
                  for(int i = 0; i < 4 ; i++ ){ 
                    A[k][4-i] = ext::intel::fpga_reg(A[k][4-i-1]); // shift register operation    
                  }
                }
              }
              COUNT++;
            }
            
          }
          
          size_t index_3 = 0;
          fpga_tools::UnrolledLoop<DoF_smooth>([&Out, &index_3](auto idx_4){
                    //A[k][0] =  PipeInInit::template PipeAt<idx_2>::read(); //get the theta_initial
              PipeOut::template PipeAt<idx_4>::write(Out[index_3++]/2.0f);
          });

          COUNT = 0;
          #pragma unroll
          for(int i=0; i<DoF_smooth; i++){
            Out[i] = 0.0f;
          }
        }
    }
};

template <typename Pipe_in_state, typename Pipe_in_Mul, typename Pipe_in_sparse, typename Pipe_theta_initial, typename Pipe_delta_theta, typename Pipe_out>
struct ARSmooth{
  void operator()() const{
    [[intel::fpga_register]] float Buf_of_sparse = 0.0f;
    [[intel::fpga_register]] int iterations = 0;
    while(1){
      [[intel::fpga_register]] float theta_0_T_times_delta_theta[DoF_smooth];
      [[intel::fpga_register]] float adder_out_DoF = 0.0f;
      [[intel::fpga_register]] float adder_out_MMUL = 0.0f;
      [[intel::fpga_register]] float adder_out_Sparse = 0.0f;
      [[intel::fpga_register]] float MMUL_in[DoF_smooth];
      [[intel::fpga_register]] float Sparse_in[DoF_smooth];

      // Initialize theta0_T_times_delta_theta
      #pragma unroll
      for(int i=0; i<DoF_smooth; i++){
        theta_0_T_times_delta_theta[i] = 0.0f;
      }

      for(int i=0; i<kSize_smooth; i++){
        [[intel::fpga_register]] float theta0_in[DoF_smooth];
        [[intel::fpga_register]] float delta_theta_in[DoF_smooth];
        size_t index_1 = 0;
        size_t index_2 = 0;
        fpga_tools::UnrolledLoop<DoF_smooth>([&delta_theta_in, &index_2](auto idx_2){
            delta_theta_in[index_2++] = Pipe_delta_theta::template PipeAt<idx_2>::read();
        });
        
        fpga_tools::UnrolledLoop<DoF_smooth>([&theta0_in, &index_1](auto idx_1){
            theta0_in[index_1++] =  Pipe_theta_initial::template PipeAt<idx_1>::read(); //get the theta_initial
        });
        
        #pragma unroll
        for(int j=0; j<DoF_smooth; j++){
          theta_0_T_times_delta_theta[j] += theta0_in[j]*delta_theta_in[j];
        }
      }

      // Read the value comes from MMUL second pipeout, read here is because try to make previous calculations work in parallel for two kernels
      size_t index_3 = 0;
      fpga_tools::UnrolledLoop<DoF_smooth>([&MMUL_in, &Sparse_in, &index_3](auto idx_3){
        MMUL_in[index_3] = Pipe_in_Mul::template PipeAt<idx_3>::read();
        Sparse_in[index_3] = Pipe_in_sparse::template PipeAt<idx_3>::read();
        index_3++;
      });
      //find the addition of inner products between theta0 and delta theta for all DoF
      #pragma unroll
      for(int i=0; i<DoF_smooth; i++){
        adder_out_DoF = ext::intel::fpga_reg(adder_out_DoF) + ext::intel::fpga_reg(theta_0_T_times_delta_theta[i]);
        adder_out_MMUL = ext::intel::fpga_reg(adder_out_MMUL) + ext::intel::fpga_reg(MMUL_in[i]);
        adder_out_Sparse = ext::intel::fpga_reg(adder_out_Sparse) + ext::intel::fpga_reg(Sparse_in[i]);
      }

      //find the addition of three adder outputs
      float add_DoF_MMUL = adder_out_DoF/kSize_smooth + adder_out_MMUL;
      float out = (iterations == 0) ? (add_DoF_MMUL+adder_out_Sparse) : (add_DoF_MMUL+Buf_of_sparse);
      Pipe_out::write(out);
      Buf_of_sparse = out;
      iterations++;

      int end_check = Pipe_in_state::template PipeAt<end_sig_idx_smooth>::read();
      if(!end_check){
        Buf_of_sparse = 0.0f;
        iterations = 0;
      }
      else{
        Buf_of_sparse = Buf_of_sparse;
        iterations = iterations;
      }

    }
  }
};
// template<typename PipeInState, typename PipeInInit, typename PipeInMMul, typename PipeOut1, typename PipeOut2> 

// //This block update the previous trajectorie base on the result output of the smoothness matrix multiplication delta = theta_previous + M*delta , refer to the README 
// struct ARTUpdate{
//   void operator()() const{
//     [[intel::fpga_register]] float T_current[DoF_smooth][kSize_smooth];
//     #pragma unroll
//     for(int i=0; i<DoF_smooth; i++){
//       #pragma unroll
//       for(int j=0; j<kSize_smooth; j++){
//         T_current[i][j] = 0.0f;
//       }
//     }
//     while(1){
      
//       bool state_iter = PipeInState::template PipeAt<0>::read();
//       //bool state_iter = PipeInState::read();
//       //PRINTF("ARTUpdate flag 1 \n");
//       // The state_iter condition indicate if it is the first iteration in which case the theta_previous is actually theta_initial and needs to be get from an external control block
//       if(state_iter == 1){
//         for(int i = 0; i < kSize_smooth; i++){
//            //PRINTF("ARTUpdate flag 3 \n");
//           size_t index_in_theta_org = 0;
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_in_theta_org](auto idx_1){
//              T_current[index_in_theta_org++][i] =  PipeInInit::template PipeAt<idx_1>::read(); //get the theta_initial
//           });
//         }
//         for(int i = 0 ; i < kSize_smooth; i++){
//           size_t index_in_matrix_out = 0;
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_in_matrix_out](auto idx_2){
//              T_current[index_in_matrix_out++][i] += PipeInMMul::template PipeAt<idx_2>::read(); // update the trajectory 
//           });
//           //PRINTF("ARTUpdate flag 4 \n");
//           size_t index_out_1 = 0;
//           size_t index_out_2 = 0;
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_1](auto idx_3){
//              PipeOut1::template PipeAt<idx_3>::write(T_current[index_out_1++][i]); //write the result to the pipeout1
//           }); 
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_2](auto idx_4){
//              PipeOut2::template PipeAt<idx_4>::write(T_current[index_out_2++][i]); //write the result to the pipeout2
//           });
//         }
//       }   
//       //If it is not the first trajectory then theta_previous is stored form the previous iteration   
//       else{
//         for(int i = 0 ; i < kSize_smooth ; i++){
//            //PRINTF("ARTUpdate flag 5 \n");
//           size_t index_in_matrix_out = 0;
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_in_matrix_out](auto idx_1){
//              T_current[index_in_matrix_out++][i] += PipeInMMul::template PipeAt<idx_1>::read();//update base on the previous iteration
//           });
//           // ARConsumePipe::write(T_current[i]); 
//           // PRINTF("ARTUpdate flag 6 \n");
//           size_t index_out_1 = 0;
//           size_t index_out_2 = 0;
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_1](auto idx_2){
//               PipeOut1::template PipeAt<idx_2>::write(T_current[index_out_1++][i]);// write the result to the pipeout1
//           }); 
//           fpga_tools::UnrolledLoop<DoF_smooth>([&i, &T_current, &index_out_2](auto idx_3){
//               PipeOut2::template PipeAt<idx_3>::write(T_current[index_out_2++][i]);// write the result to the pipeout1
//           }); 
        
//         }
//       }    
//     }
//   }
// };
        
// template<typename PipeInState,typename PipeInInit,typename PipeInMMul, typename PipeInSparse, typename PipeInDelta, typename PipeInNewTraj, typename PipeOut>
// struct ARSmooth{
//     void operator()() const {
//             [[intel::fpga_register]] float smoothRes = 0.0f; 
//         while(1){
          
//             float Vec_Mul1 = 0;
//             float Vec_Mul2 = 0; 
//             float delta; 
//             float MulRes ; 
//             float  initial_traj;
      
             
//             // If it the first iteration of the block we received the initial_traj from the control block. 
//             if( PipeInState::template PipeAt<1>::read() == 1){  
//               for(int i = 0; i < kSize_smooth; i++){   
//                 //PRINTF("SMOOTH COST flag 0 \n"); 
//                 delta = PipeInDelta::template PipeAt<1>::read(); 
//                 //PRINTF("SMOOTH COST flag 1 \n"); 
//                 MulRes = PipeInMMul::template PipeAt<1>::read(); 
//                  //PRINTF("SMOOTH COST flag 2 \n");
//                 initial_traj = PipeInInit::template PipeAt<1>::read(); 
//                 Vec_Mul1 += delta * MulRes; 
//                 Vec_Mul2 += delta * initial_traj; 
//               }   
                       
//                 smoothRes = PipeInSparse::read() + Vec_Mul2 + Vec_Mul1 ;  
//                 PRINTF("SMOOTH COST flag 3 \n");
//                 PipeOut::write(smoothRes); 
//             }
//             //if it is not the first iteration the initial trajectorie is received from the update block. 
//             //Other than that the function performing the vector multiplication operation express in the README. 
//             else{
//               for(int i = 0; i < kSize; i++){   
//                   PRINTF("SMOOTH COST flag 1 0 \n"); 
//                     delta = PipeInDelta::template PipeAt<1>::read(); 
//                   PRINTF("SMOOTH COST flag 1 1 \n"); 
//                    MulRes = PipeInMMul::template PipeAt<1>::read(); 
//                   PRINTF("SMOOTH COST flag 1 2 \n"); 
//                   initial_traj = PipeInNewTraj::template PipeAt<1>::read(); 
//                   Vec_Mul1 += delta * MulRes; 
//                   Vec_Mul2 += delta * initial_traj; 
//               }   
//                 smoothRes =  Vec_Mul2 + Vec_Mul1 +   0.5 * smoothRes; 
//                 PipeOut::write(smoothRes);

                 
//             }     

         
                       
//         }
//     }
// };     
                   


  
}


#endif