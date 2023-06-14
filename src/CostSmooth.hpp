#ifndef _COSTSMOOTH_HPP
#define _COSTSMOOTH_HPP
#define kSize 5 

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


namespace smooth{

    
template<typename PipeInDelta, typename PipeOutMMul>
struct ARMMul{
    void operator()() const {
        // this is the matrix M that is store in register 
      [[intel::fpga_register]]float M_matrix[25] = 
    {0.535714285714286,0.714285714285715,0.642857142857144,0.428571428571429,0.178571428571429,
0.714285714285715,1.42857142857143,1.42857142857143,1,0.428571428571429,
0.642857142857144,1.42857142857143,1.85714285714286,1.42857142857143,0.642857142857144,
0.428571428571429,1,1.42857142857143,1.42857142857143,0.714285714285715,
0.178571428571429,0.428571428571429,0.642857142857143,0.714285714285715,0.535714285714286}; 
      float A_out[kSize]; 
        
        while(1){ 

          for(int i = 0; i < kSize; i++){
              PRINTF("ARMMUL flag 1 \n");
              float  val = PipeInDelta::template PipeAt<0>::read(); 
              PRINTF("ARMMUL flag 2 \n");
              #pragma unroll 
              for(int j = 0; j < kSize; j++){
                  val = ext::intel::fpga_reg(val); 
                  A_out[j] += val*M_matrix[i*kSize+j]; //processing k element in parallel the output is serial
              }
          }

          for(int i = 0; i < kSize; i++){
            fpga_tools::UnrolledLoop<2>([&i, & A_out](auto j){
                 PipeOutMMul::template PipeAt<j>::write(A_out[i]);
                 PRINTF("ARMMUL flag 3 in the loop  \n"); 
             }); 
                //  TUpdatePipe::PipeAt<0>::write(A_out[i]);
                //  PRINTF("ARMMUL flag 3 in the loop  \n"); 
            A_out[i]  = 0; 
          }     
        }
    }
};

template<typename PipeInState, typename PipeInInit, typename PipeInMMul, typename PipeOut> 

//This block update the previous trajectorie base on the result output of the smoothness matrix multiplication delta = theta_previous + M*delta , refer to the README 
struct ARTUpdate{
  void operator()() const{
    while(1){
      [[intel::fpga_register]] float T_current[5]; 
      bool state_iter = PipeInState::template PipeAt<0>::read();
      PRINTF("ARTUpdate flag 1 \n");
      // The state_iter condition indicate if it is the first iteration in which case the theta_previous is actually theta_initial and needs to be get from an external control block
      if(state_iter == 1){
        for(int i = 0; i < kSize; i++){
           PRINTF("ARTUpdate flag 3 \n");
          
          T_current[i] =  PipeInInit::template PipeAt<0>::read(); //get the theta_initial
         
        }
        for(int i = 0 ; i < kSize ; i++){
          T_current[i] += PipeInMMul::template PipeAt<0>::read(); // update the trajectory 
          PRINTF("ARTUpdate flag 4 \n");
          fpga_tools::UnrolledLoop<2>([&i,&T_current](auto j){
             PipeOut::template PipeAt<j>::write(T_current[i]); //write the result 
          }); 
           
          
        }
      }   
      //If it is not the first trajectory then theta_previous is stored form the previous iteration   
      else{
        for(int i = 0 ; i < kSize ; i++){
           PRINTF("ARTUpdate flag 5 \n");
          T_current[i] =   T_current[i] + PipeInMMul::template PipeAt<0>::read(); //update base on the previous iteration
          // ARConsumePipe::write(T_current[i]); 
           PRINTF("ARTUpdate flag 6 \n");
          fpga_tools::UnrolledLoop<2>([&i,&T_current](auto j){
              PipeOut::template PipeAt<j>::write(T_current[i]);// write the result
          }); 
          
        
        }
      }    
    }
  }
};

template<typename PipeInState,typename PipeInInit,typename PipeInMMul, typename PipeInSparse, typename PipeInDelta, typename PipeInNewTraj, typename PipeOut>


struct ARSmooth{

    void operator()() const {
            [[intel::fpga_register]] float smoothRes = 0.0; 
        while(1){
          
            float Vec_Mul1 = 0;
            float Vec_Mul2 = 0; 
            float delta; 
            float MulRes ; 
            float  initial_traj;
      
             
            // If it the first iteration of the block we received the initial_traj from the control block. 
            if( PipeInState::template PipeAt<1>::read() == 1){  
              for(int i = 0; i < kSize; i++){   
                PRINTF("SMOOTH COST flag 0 \n"); 
                delta = PipeInDelta::template PipeAt<1>::read(); 
                PRINTF("SMOOTH COST flag 1 \n"); 
                MulRes = PipeInMMul::template PipeAt<1>::read(); 
                 PRINTF("SMOOTH COST flag 2 \n");
                initial_traj = PipeInInit::template PipeAt<1>::read(); 
                Vec_Mul1 += delta * MulRes; 
                Vec_Mul2 += delta * initial_traj; 
              }   
                       
                smoothRes = PipeInSparse::read() + Vec_Mul2 + Vec_Mul1 ;  
                PRINTF("SMOOTH COST flag 3 \n");
                PipeOut::write(smoothRes); 
            }
            //if it is not the first iteration the initial trajectorie is received from the update block. 
            //Other than that the function performing the vector multiplication operation express in the README. 
            else{
              for(int i = 0; i < kSize; i++){   
                  PRINTF("SMOOTH COST flag 1 0 \n"); 
                    delta = PipeInDelta::template PipeAt<1>::read(); 
                  PRINTF("SMOOTH COST flag 1 1 \n"); 
                   MulRes = PipeInMMul::template PipeAt<1>::read(); 
                  PRINTF("SMOOTH COST flag 1 2 \n"); 
                  initial_traj = PipeInNewTraj::template PipeAt<1>::read(); 
                  Vec_Mul1 += delta * MulRes; 
                  Vec_Mul2 += delta * initial_traj; 
              }   
                smoothRes =  Vec_Mul2 + Vec_Mul1 +   0.5 * smoothRes; 
                PipeOut::write(smoothRes);

                 
            }     

         
                       
        }
    }
};



 
template<typename PipeInState, typename PipeInInit, typename PipeOut >
struct ARSparseMul{
    void operator()() const
      {
        [[intel::fpga_register]] float A[5] = {0, 0, 0, 0, 0};
        float Out; 
        int COUNT = 0;
        
        
        while(1){
          PRINTF("SPARSE flag 1 \n");
          if(PipeInState::template PipeAt<2>::read() == 1){ //this block is only require once for the first iteration it is desable the rest of the time 
            COUNT = 0; 
            PRINTF("SPARSE flag 2 \n");
             Out = 0; 
            A[0] = PipeInInit::read(); 
            for(int j = 0; j < kSize + 2; j++){ //Mind that kSize + 2 cycle to be perform as it take 2 cycle to instantiate
                Out += A[2]*(A[0] +(-4) * A[1] + 6*A[2] + (-4)* A[3] + A[4]); // perfom the operation as explain in the README 
                PRINTF("SPARSE flag 3 \n");
                #pragma unroll 
                for(int i = 0; i < 4 ; i++ ){ 
                    A[4-i] = ext::intel::fpga_reg(A[4-i-1]); // shift register operation 
                   
                } 
                if(COUNT < kSize-1){ //As mention the operation takes kSize+2 cycle for the last 2 cycle the new input needs to be 0 this what the condition enables. 
                PRINTF("SPARSE flag 4\n");
                A[0] = PipeInInit::read();
                COUNT++; 
                }
                else{
                  PRINTF("SPARSE flag 4\n");
                 A[0] = 0.0f; 
                }
            }
          }
            //count = 0; 
            PipeOut::write(Out); 
        }
    //     while(1){
    //       float A = ARProducePipe::read(); 
    //       ARConsumePipe::write(A); 
    //     } 
    }
};
  
}
#endif