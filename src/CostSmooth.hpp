#ifndef _COSTSMOOTH_HPP
#define _COSTSMOOTH_HPP

namespace smooth{
    
template<typename PipeIn, typename PipeOutMMul>
struct ARMMul{
    void operator()() const {
        
      [[intel::fpga_register]]float F_matrix[25] = 
    {0.535714285714286,0.714285714285715,0.642857142857144,0.428571428571429,0.178571428571429,
0.714285714285715,1.42857142857143,1.42857142857143,1,0.428571428571429,
0.642857142857144,1.42857142857143,1.85714285714286,1.42857142857143,0.642857142857144,
0.428571428571429,1,1.42857142857143,1.42857142857143,0.714285714285715,
0.178571428571429,0.428571428571429,0.642857142857143,0.714285714285715,0.535714285714286}; 
      float A_out[kSize]; 
        while(1){ 
          

          for(int i = 0; i < kSize; i++){
              PRINTF("ARMMUL flag 1 \n");
              float  val = PipeIn::template PipeAt<0>::read(); 
              PRINTF("ARMMUL flag 2 \n");
              #pragma unroll 
              for(int j = 0; j < kSize; j++){
                  val = ext::intel::fpga_reg(val); 
                  A_out[j] += val*F_matrix[i*kSize+j]; 
              }
          }

          for(int i = 0; i < kSize; i++){
            fpga_tools::UnrolledLoop<2>([&i, & A_out](auto j){
                 PipeOut::template PipeAt<j>::write(A_out[i]);
                 PRINTF("ARMMUL flag 3 in the loop  \n"); 
             }); 
                //  TUpdatePipe::PipeAt<0>::write(A_out[i]);
                //  PRINTF("ARMMUL flag 3 in the loop  \n"); 
            A_out[i]  = 0; 
          }     
        }
    }
};

template<typename PipeInState, typename PipeInInit, typename PipeinMMul, typename PipeOut> 
struct ARTUpdate{
  void operator()() const{
    while(1){

      [[intel::fpga_register]] float T_current[5]; 
      bool state_iter = PipeInState::template PipeAt<0>::read();
      PRINTF("ARTUpdate flag 1 \n");
  
      if(state_iter == 1){
        for(int i = 0; i < kSize; i++){
           PRINTF("ARTUpdate flag 3 \n");
          T_current[i] =  PipeInInit::template PipeAt<0>::read(); 
         
        }
        for(int i = 0 ; i < kSize ; i++){
          T_current[i] += PipeinDelta::template PipeAt<0>::read(); 
          PRINTF("ARTUpdate flag 4 \n");
          fpga_tools::UnrolledLoop<2>([&i,&T_current](auto j){
             PipeOut::template PipeAt<j>::write(T_current[i]);
          }); 
           
          
        }
      }     
      else{
        for(int i = 0 ; i < kSize ; i++){
           PRINTF("ARTUpdate flag 5 \n");
          T_current[i] =   T_current[i] + PipeinDelta::template PipeAt<0>::read(); 
          // ARConsumePipe::write(T_current[i]); 
           PRINTF("ARTUpdate flag 6 \n");
          fpga_tools::UnrolledLoop<2>([&i,&T_current](auto j){
              PipeOut::template PipeAt<j>::write(T_current[i]);
          }); 
          
        
        }
      }    
    }
  }
};

template<typename PipeInSparse, typename PipeInDelta, typename PipeInState, typename PipeInNewTraj, typename PipeInMMul,typename PipeInInit, typename PipeOut>
struct ARSmooth{

    void operator()() const {
            [[intel::fpga_register]] float smoothRes = 0.0; 
        while(1){
          
            float Vec_Mul1 = 0;
            float Vec_Mul2 = 0; 
            float delta; 
            float MulRes ; 
            float  initial_traj;
            // for(int i = 0 ; i < kSize; i++){
            //   PRINTF("SMOOTH COST flag 0 \n"); 
            //      delta = ARProducePipe::PipeAt<1>::read(); 
            // }

            // for(int i = 0; i < kSize; i++){   
            //     PRINTF("SMOOTH COST flag 0 \n"); 
            //       delta = ARProducePipe::PipeAt<1>::read(); 
            //     PRINTF("SMOOTH COST flag 1 \n"); 
            //     float MulRes = TUpdatePipe::PipeAt<1>::read(); 
            //     Vec_Mul1 += delta * MulRes; 
            //     Vec_Mul2 += delta * newTrajPipe::PipeAt<1>::read(); 
                 
            // }
            if( PipeInState::template template PipeAt<1>::read() == 1){  
              for(int i = 0; i < kSize; i++){   
                PRINTF("SMOOTH COST flag 0 \n"); 
                  delta = PipeInDelta::PipeAt<1>::read(); 
                PRINTF("SMOOTH COST flag 1 \n"); 
                MulRes = PipeInMMul::template PipeAt<1>::read(); 
                 PRINTF("SMOOTH COST flag 2 \n");
                initial_traj = PipeInSparse::PipeAt<1>::read(); 
                Vec_Mul1 += delta * MulRes; 
                Vec_Mul2 += delta * initial_traj; 
              }   
                       
                smoothRes = PipeInSparse::read() + Vec_Mul2 + Vec_Mul1 ;  
                PRINTF("SMOOTH COST flag 3 \n");
                PipeOut::write(smoothRes); 
            }
            else{

              for(int i = 0; i < kSize; i++){   
                  PRINTF("SMOOTH COST flag 1 0 \n"); 
                    delta = PipeInDelta::PipeAt<1>::read(); 
                  PRINTF("SMOOTH COST flag 1 1 \n"); 
                   MulRes = PipeInMMul::template PipeAt<1>::read(); 
                  PRINTF("SMOOTH COST flag 1 2 \n"); 
                  initial_traj = PipeInNewTraj::PipeAt<1>::read(); 
                  Vec_Mul1 += delta * MulRes; 
                  Vec_Mul2 += delta * initial_traj; 
              }   
                smoothRes =  Vec_Mul2 + Vec_Mul1 +   0.5 * smoothRes; 
                PipeOut::write(smoothRes);

                 
            }     

         
                       
        }
    }
};

template<typename PipeInState, typename PipeInit, typename PipeOut >
struct ARSparseMul{
    void operator()() const
      {
        [[intel::fpga_register]] float A[5] = {0, 0, 0, 0, 0};
        float Out; 
        int COUNT = 0;
        
        
        while(1){
          PRINTF("SPARSE flag 1 \n");
          if(PipeInState::template ::PipeAt<2>::read() == 1){
            COUNT = 0; 
            PRINTF("SPARSE flag 2 \n");
             Out = 0; 
            A[0] = PipeInInit::read(); 
            for(int j = 0; j < 7; j++){
                Out += A[2]*(A[0] +(-4) * A[1] + 6*A[2] + (-4)* A[3] + A[4]); 
                PRINTF("SPARSE flag 3 \n");
                #pragma unroll 
                for(int i = 0; i < 4 ; i++ ){ 
                    A[4-i] = ext::intel::fpga_reg(A[4-i-1]); 
                   
                } 
                if(COUNT < 4){
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