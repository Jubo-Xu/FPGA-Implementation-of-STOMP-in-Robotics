// #include <sycl/sycl.hpp>
// #include <sycl/ext/intel/fpga_extensions.hpp>
// #include "autorun.hpp"

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
//#include <sycl/ext/intel/fpga_extensions.hpp>
#include "autorun.hpp"
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include "CostSmooth.hpp"


#define kSize  5 

#define  count 

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl; 

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }




#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif


//KernelID
class ARConsumeID; 
class ARProduceID; 
class KsparseMulID; 
class ARMMULID; 
class ARsmoothCostID;
class ARTUpdateID; 


//PipeID
class ARProducePipeID; //ok
class ARConsumePipeID; 
class ARInitThetaPipeID; 
class StateIterPipeID;
class TInitPipeID;
class newTrajPipeID;
class ARCostSCPipeID; 
class Init2SparsePipeID; 
class Sparse2SmoothCPipeID;
class ARMMulPipeID; 

//Pipes
using ARConsumePipe =sycl::ext::intel::pipe<ARConsumePipeID,float>; 
using ARProducePipe =  fpga_tools::PipeArray<ARProducePipeID, float, 5,2>; 
using StateIterPipe = fpga_tools::PipeArray<StateIterPipeID, float,0,3>; 
using ARMMulPipe = fpga_tools::PipeArray<ARMMulPipeID, float, 5, 2>; 
using TInitPipe =  fpga_tools::PipeArray<TInitPipeID, float,5, 2>; 
using Init2SparsePipe = sycl::ext::intel::pipe<Init2SparsePipeID, float>; 
using Sparse2SmoothCPipe = sycl::ext::intel::pipe<Sparse2SmoothCPipeID, float>; 
using newTrajPipe =fpga_tools::PipeArray<newTrajPipeID,float, 5, 2>; 
using ARCostSCPipe = sycl::ext::intel::pipe< ARCostSCPipeID, float>; 

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};


fpga_tools::Autorun<ARTUpdateID> ar_kernelUpdate{selector,smooth::ARTUpdate<StateIterPipe, TInitPipe, ARMMulPipe, newTrajPipe >{}}; 
fpga_tools::Autorun<ARMMULID> ar_kernel{selector, smooth::ARMMul<ARProducePipe, ARMMulPipe>{}};
fpga_tools::Autorun<ARsmoothCostID> ar_SmoothCost{selector, smooth::ARSmooth<StateIterPipe,TInitPipe , ARMMulPipe, Sparse2SmoothCPipe ,ARProducePipe, newTrajPipe,ARCostSCPipe >{}}; 
fpga_tools::Autorun<KsparseMulID> ar_SparseMul{selector, smooth::ARSparseMul<StateIterPipe, Init2SparsePipe, Sparse2SmoothCPipe>{}}; 

template<typename KernelID, typename Pipe, typename Pipestate, typename TinitPipe, typename Init2SparsePipe> 
event SubmitProducerKernel(queue& q, buffer<float, 1>& in_buf, buffer<float, 1>& Tinit){
    return q.submit([&](sycl::handler& h){
        sycl::accessor in(in_buf, h); 
        sycl::accessor A_Tinit(Tinit, h); 
        h.single_task<KernelID>([=](){

            for(int i = 0; i < 2 ; i++ ){
              if(i == 0){
                fpga_tools::UnrolledLoop<3>([](auto k){
                Pipestate::template PipeAt<k>::write(1);
                });
                
                 PRINTF("Producer flag 1 \n");
               for(int j = 0; j < kSize; j++){
                  fpga_tools::UnrolledLoop<2>([&in, &j, &A_Tinit](auto k){
                  Pipe::template PipeAt<k>::write(in[j]);
                  TinitPipe::template PipeAt<k>::write(A_Tinit[j]);
                }); 
                  Init2SparsePipe::write(A_Tinit[j]); 
                // Pipe::template PipeAt<0>::write(in[j]);
                PRINTF("Producer flag 2 \n");
                  
               }
              }

              else{
                fpga_tools::UnrolledLoop<3>([](auto k){
                  Pipestate::template PipeAt<k>::write(0);
                });
                
                for(int j = 0 ; j < kSize; j++){
                   PRINTF("Producer flag 2 \n");
                  fpga_tools::UnrolledLoop<2>([&in, &j, &i ](auto k) {
                        Pipe::template PipeAt<k>::write(in[i*5+j]);
                 });
                  // Pipe::template PipeAt<0>::write(in[i*5+j]);
                    
                }
              }

            } 

        });
    });
}

template<typename KernelID, typename Pipe, typename ARCostSCPipe> 
event SubmitConsumerKernel(queue& q, buffer<float, 1>& out_buf, buffer<float, 1> &OutSmoothCost){
    return q.submit([&](sycl::handler& h){
        sycl::accessor out(out_buf, h, write_only, no_init ); 
       
       sycl::accessor OutSmooth(OutSmoothCost, h, write_only, no_init ); 
        h.single_task<KernelID>([=](){
            for(int i = 0 ; i < 5 ; i++){
                out[i] = Pipe::template PipeAt<0>::read(); 
            }
            OutSmooth[0] = ARCostSCPipe::read(); 
            for(int i = 5 ; i < 10 ; i++){
                out[i] = Pipe::template PipeAt<0>::read(); 
            }
            OutSmooth[1] = ARCostSCPipe::read(); 

        });
    });
}

int main(){
    std::vector<bool> state; 
    std::vector<float > in_data;
    std::vector<float> T_init(5); 


    //fill vector
    in_data.push_back(14); 
    in_data.push_back(5); 
    in_data.push_back(15); 
    in_data.push_back(2); 
    in_data.push_back(9); 

    for(int i = 5; i < 5+5; i++){
       in_data.push_back(i);
      
    }

    /////

    std::fill(T_init.begin(),T_init.end(), 6); 
    
    /////
    
    std::cout<<"input data \n";
    for(int i = 0; i< 10; i++){
      std::cout<<"data in" <<in_data[i]<<std::endl; 
    }
    for(int i = 0; i < 5; i++){
      std::cout<<"intitial condition: "<<T_init[i]<<std::endl;
    }
    
    std::cout<<"input data finished printing \n";

    sycl::buffer<float, 1> in_buf{in_data.begin(), in_data.end()};  
    sycl::buffer<float, 1> out_buf{10}; 
    sycl::buffer<float, 1> out_costSmooth{2}; 
    sycl::buffer<float, 1> B_Tinit{T_init.begin(), T_init.end()}; 
    queue q(selector, exception_handler);
    SubmitProducerKernel<ARProduceID, ARProducePipe, StateIterPipe, TInitPipe, Init2SparsePipe>(q, in_buf, B_Tinit);
    SubmitConsumerKernel<ARConsumeID, newTrajPipe, ARCostSCPipe>(q, out_buf, out_costSmooth);
    sycl::host_accessor result{out_buf}; 
    for(int i = 0; i < 10 ; i++){
      std::cout<<"result:"<<result[i]<<std::endl; 
    }
    sycl::host_accessor resultsmooth{out_costSmooth}; 
    for(int i = 0; i < 2 ; i++){
      std::cout<<"result:"<<resultsmooth[i]<<std::endl; 
    }

    
}


// smooth::ARMMUl<ARProducePipe, ARMMulPipe>; //done
// smooth::ARSmooth<StateIterPipe,TInitPipe , ARMMulPipe, Sparse2SmoothCPipe ,ARProducePipe, newTrajPipe,ARCostSCPipe >; //done
// smooth::SparseMul<StateIterPipe,TInitPipe, Sparse2SmoothCPipe>; //done
// smooth::ARTUpdate<StateIterPipe, TInitPipe, ARMMulPipe, newTrajPipe >//done




// struct ARSparseMul{
//     void operator()() const
//       {
//         [[intel::fpga_register]] float A[5] = {0, 0, 0, 0, 0};
//         float Out; 
//         int COUNT = 0;    
//         while(1){
//           PRINTF("SPARSE flag 1 \n");
//           if(StateIterPipe::PipeAt<2>::read() == 1){
//             COUNT = 0; 
//             PRINTF("SPARSE flag 2 \n");
//              Out = 0; 
//             A[0] = Init2SparsePipe::read(); 
//             for(int j = 0; j < 7; j++){
//                 Out += A[2]*(A[0] +(-4) * A[1] + 6*A[2] + (-4)* A[3] + A[4]); 
//                 PRINTF("SPARSE flag 3 \n");
//                 #pragma unroll 
//                 for(int i = 0; i < 4 ; i++ ){ 
//                     A[4-i] = ext::intel::fpga_reg(A[4-i-1]);                  
//                 } 
//                 if(COUNT < 4){
//                 PRINTF("SPARSE flag 4\n");
//                 A[0] = Init2SparsePipe::read();
//                 COUNT++; 
//                 }
//                 else{
//                   PRINTF("SPARSE flag 4\n");
//                  A[0] = 0.0f; 
//                 }
//             }
//           }
//             //count = 0; 
//             Sparse2SmoothCPipe::write(Out); 
//         }
//     //     while(1){
//     //       float A = ARProducePipe::read(); 
//     //       ARConsumePipe::write(A); 
//     //     } 
//     }
// };

// struct ARSmooth{
//     void operator()() const {
//             [[intel::fpga_register]] float smoothRes = 0.0; 
//         while(1){      
//             float Vec_Mul1 = 0;
//             float Vec_Mul2 = 0; 
//             float delta; 
//             float MulRes ; 
//             float  initial_traj;
//             // for(int i = 0 ; i < kSize; i++){
//             //   PRINTF("SMOOTH COST flag 0 \n"); 
//             //      delta = ARProducePipe::PipeAt<1>::read(); 
//             // }
//             // for(int i = 0; i < kSize; i++){   
//             //     PRINTF("SMOOTH COST flag 0 \n"); 
//             //       delta = ARProducePipe::PipeAt<1>::read(); 
//             //     PRINTF("SMOOTH COST flag 1 \n"); 
//             //     float MulRes = ARMMulPipe::PipeAt<1>::read(); 
//             //     Vec_Mul1 += delta * MulRes; 
//             //     Vec_Mul2 += delta * newTrajPipe::PipeAt<1>::read();                 
//             // }
//             if( StateIterPipe::template PipeAt<1>::read() == 1){  
//               for(int i = 0; i < kSize; i++){   
//                 PRINTF("SMOOTH COST flag 0 \n"); 
//                   delta = ARProducePipe::PipeAt<1>::read(); 
//                 PRINTF("SMOOTH COST flag 1 \n"); 
//                 MulRes = ARMMulPipe::PipeAt<1>::read(); 
//                  PRINTF("SMOOTH COST flag 2 \n");
//                 initial_traj = TInitPipe::PipeAt<1>::read(); 
//                 Vec_Mul1 += delta * MulRes; 
//                 Vec_Mul2 += delta * initial_traj; 
//               }                         
//                 smoothRes = Sparse2SmoothCPipe::read() + Vec_Mul2 + Vec_Mul1 ;  
//                 PRINTF("SMOOTH COST flag 3 \n");
//                 ARCostSCPipe::write(smoothRes); 
//             }
//             else{
//               for(int i = 0; i < kSize; i++){   
//                   PRINTF("SMOOTH COST flag 1 0 \n"); 
//                     delta = ARProducePipe::PipeAt<1>::read(); 
//                   PRINTF("SMOOTH COST flag 1 1 \n"); 
//                   float MulRes = ARMMulPipe::PipeAt<1>::read(); 
//                   PRINTF("SMOOTH COST flag 1 2 \n"); 
//                   initial_traj = newTrajPipe::PipeAt<1>::read(); 
//                   Vec_Mul1 += delta * MulRes; 
//                   Vec_Mul2 += delta * initial_traj; 
//               }   
//                 smoothRes =  Vec_Mul2 + Vec_Mul1 +   0.5 * smoothRes; 
//                 ARCostSCPipe::write(smoothRes);            
//             }                         
//         }
//     }
// };

// struct ARMMul{
//     void operator()() const {    
//       [[intel::fpga_register]]float F_matrix[25] = 
//     {0.535714285714286,0.714285714285715,0.642857142857144,0.428571428571429,0.178571428571429,
// 0.714285714285715,1.42857142857143,1.42857142857143,1,0.428571428571429,
// 0.642857142857144,1.42857142857143,1.85714285714286,1.42857142857143,0.642857142857144,
// 0.428571428571429,1,1.42857142857143,1.42857142857143,0.714285714285715,
// 0.178571428571429,0.428571428571429,0.642857142857143,0.714285714285715,0.535714285714286}; 
//       float A_out[kSize]; 
//         while(1){ 
//           for(int i = 0; i < kSize; i++){
//               PRINTF("ARMMUL flag 1 \n");
//               float  val = ARProducePipe::PipeAt<0>::read(); 
//               PRINTF("ARMMUL flag 2 \n");
//               #pragma unroll 
//               for(int j = 0; j < kSize; j++){
//                   val = ext::intel::fpga_reg(val); 
//                   A_out[j] += val*F_matrix[i*kSize+j]; 
//               }
//           }
//           for(int i = 0; i < kSize; i++){
//             fpga_tools::UnrolledLoop<2>([&i, & A_out](auto j){
//                  ARMMulPipe::PipeAt<j>::write(A_out[i]);
//                  PRINTF("ARMMUL flag 3 in the loop  \n"); 
//              }); 
//                 //  ARMMulPipe::PipeAt<0>::write(A_out[i]);
//                 //  PRINTF("ARMMUL flag 3 in the loop  \n"); 
//             A_out[i]  = 0; 
//           }     
//         }
//     }
// };
// // start the AutoRun Kernel

// struct ARTUpdate{
//   void operator()() const{
//     while(1){
//       [[intel::fpga_register]] float T_current[5]; 
//       bool state_iter = StateIterPipe::PipeAt<0>::read();
//       PRINTF("ARTUpdate flag 1 \n");
//       if(state_iter == 1){
//         for(int i = 0; i < kSize; i++){
//            PRINTF("ARTUpdate flag 3 \n");
//           T_current[i] =  TInitPipe::PipeAt<0>::read();       
//         }
//         for(int i = 0 ; i < kSize ; i++){
//           T_current[i] += ARMMulPipe::PipeAt<0>::read(); 
//           PRINTF("ARTUpdate flag 4 \n");
//           fpga_tools::UnrolledLoop<2>([&i,&T_current](auto j){
//               newTrajPipe::PipeAt<j>::write(T_current[i]);
//           });         
//         }
//       }     
//       else{
//         for(int i = 0 ; i < kSize ; i++){
//            PRINTF("ARTUpdate flag 5 \n");
//           T_current[i] =   T_current[i] + TUpdatePipe::PipeAt<0>::read(); 
//           // ARConsumePipe::write(T_current[i]); 
//            PRINTF("ARTUpdate flag 6 \n");
//           fpga_tools::UnrolledLoop<2>([&i,&T_current](auto j){
//               newTrajPipe::PipeAt<j>::write(T_current[i]);
//           });      
//         }
//       }    
//     }
//   }
// };

