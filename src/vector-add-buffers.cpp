//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
/*#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// num_repetitions: How many times to repeat the kernel invocation
size_t num_repetitions = 1;
// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<int> IntVector; 

// Create an exception handler for asynchronous SYCL exceptions
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

//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum_parallel".
//************************************
void VectorAdd(queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{a_vector.size()};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf(a_vector);
  buffer b_buf(b_vector);
  buffer sum_buf(sum_parallel.data(), num_items);

  for (size_t i = 0; i < num_repetitions; i++ ) {

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor for each buffer with access permission: read, write or
      // read/write. The accessor is a mean to access the memory in the buffer.
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);
  
      // The sum_accessor is used to store (with write permission) the sum data.
      accessor sum(sum_buf, h, write_only, no_init);
  
      // Use parallel_for to run vector addition in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // SYCL supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
    });
  };
  // Wait until compute tasks on GPU done
  q.wait();
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
void InitializeVector(IntVector &a) {
  for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change num_repetitions if it was passed as argument
  if (argc > 2) num_repetitions = std::stoi(argv[2]);
  // Change vector_size if it was passed as argument
  if (argc > 1) vector_size = std::stoi(argv[1]);
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // Intel extension: FPGA emulator selector on systems without FPGA card.
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
  // Intel extension: FPGA simulator selector on systems without FPGA card.
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  // Intel extension: FPGA selector on systems with FPGA card.
  auto selector = sycl::ext::intel::fpga_selector_v;
#else
  // The default device selector will select the most performant device.
  auto selector = default_selector_v;
#endif

  // Create vector objects with "vector_size" to store the input and output data.
  IntVector a, b, sum_sequential, sum_parallel;
  a.resize(vector_size);
  b.resize(vector_size);
  sum_sequential.resize(vector_size);
  sum_parallel.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(a);
  InitializeVector(b);

  try {
    queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << a.size() << "\n";

    // Vector addition in SYCL
    VectorAdd(q, a, b, sum_parallel);
  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }

  // Compute the sum of two vectors in sequential for validation.
  for (size_t i = 0; i < sum_sequential.size(); i++)
    sum_sequential.at(i) = a.at(i) + b.at(i);

  // Verify that the two vectors are equal.  
  for (size_t i = 0; i < sum_sequential.size(); i++) {
    if (sum_parallel.at(i) != sum_sequential.at(i)) {
      std::cout << "Vector add failed on device.\n";
      return -1;
    }
  }

  int indices[]{0, 1, 2, (static_cast<int>(a.size()) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out the result of vector add.
  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "]: " << a[j] << " + " << b[j] << " = "
              << sum_parallel[j] << "\n";
  }

  a.clear();
  b.clear();
  sum_sequential.clear();
  sum_parallel.clear();

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}*/


// Test for LOU
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

//#include "exception_handler.hpp"






using namespace sycl;





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

//PipeID

class ARProducePipeID;

class ARConsumeID;

class ARMMulPipeOutID;

class ARSparsePipeID;

class ARInitThetaPipeID;




using ARConsumePipe = sycl::ext::intel::pipe<ARConsumeID,float>;

using ARProducePipe = sycl::ext::intel::pipe<ARProducePipeID, float>;

using SparsePipe = sycl::ext::intel::pipe<ARSparsePipeID, float>;

using InitThetaPipe = sycl::ext::intel::pipe<ARInitThetaPipeID, float>;


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


// struct ARSmooth{

//     private:

//         float Vec_Mul1 = 0 ;

//         float Vec_Mul2 = 0 ;

//         float smoothRes = 0;

//         constexpr int Iter = 5 ;

//     public:

//     void operator()() {

//         while(1){

//             for(int i = 0; i < kSize; i++){

//                 float delta = ARProducePipe::read();

//                 float MulRes = ARMMulPipeOutID::read();

//                  Vec_Mul1 += delta * MulRes;

//                  Vec_Mul2 += delta * InitThethaPipe;

//             }          

//             if(counter == 0){              

//                 smoothRes = SparsePipe::read() + Vec_Mul2 + Vec_Mul1 ;  

//             }

//             else{

//                 smoothRes = 0.5 * smoothRes + Vec_Mul2 + Vec_Mul1;

//             }          

//             if(counter == Iter){

//                 counter = 0;

//             }            

//             Vec_Mul1 = 0;

//             Vec_Mul2 = 0;

//             smoothRes = 0;            

//         }

//     }

// }




struct ARsparseMul{

    void operator()() const

      {

     

        [[intel::fpga_register]] float A[5] = {0, 0, 0, 0, 0};;

       

        while(1){

            [[intel::fpga_register]] uint8_t count = 0 ;

            float Out = 0;

            A[0] = ARProducePipe::read();

            for(int j = 0; j < 7; j++){

                Out += A[2]*(A[0] +(-4) * A[1] + 6*A[2] + (-4)* A[3] + A[4]);

                #pragma unroll

                for(int i = 0; i < 4 ; i++ ){

                    A[4-i] = ext::intel::fpga_reg(A[4-i-1]);

                }

                if(count < 4){

                A[0] = ARProducePipe::read();

                count++;

                }

                else{

                 A[0] = 0.0f;

                }

            }

            count = 0;

            SparsePipe::write(Out);

        }

   

    }

};






// struct ARMMul{

//     private:

//     constexpr float M_matrix[kSize*kSize] =

//     {0.535714285714286,0.714285714285715,0.642857142857144,0.428571428571429,0.178571428571429,

// 0.714285714285715,1.42857142857143,1.42857142857143,1,0.428571428571429,

// 0.642857142857144,1.42857142857143,1.85714285714286,1.42857142857143,0.642857142857144,

// 0.428571428571429,1,1.42857142857143,1.42857142857143,0.714285714285715,

// 0.178571428571429,0.428571428571429,0.642857142857143,0.714285714285715,0.535714285714286};

//     kSize = 5;

//     public:

//     void operator()() const {

//         while(1){      

//         [[intel::fpga_register]] float F_matrix[25];      

//         for (size_t i = 0; i < kSize*kSize; i++) {

//         F_matrix[i] = M_matrix[i];

//         }

//         for(int i = 0; i < kSize; i++){

//             float  val = Pipe_in::read();

//             #pragma unroll

//             for(int j = 0; j < N; j++){

//                 val = ext::intel::fpga_reg(val);

//                 A_out[j] += val*F_matrix[i*5+j];

//             }

//         }      

//         for(int i = 0; i < 5; i++){

//             Pipe_out::write(A_out[i]);

//             }

//         }

//     }

// }

// start the AutoRun Kernel

fpga_tools::Autorun<KsparseMulID> ar_kernel{selector, ARsparseMul{}};




template<typename KernelID, typename Pipe>

event SubmitProducerKernel(queue& q, buffer<float, 1>& in_buf){

    return q.submit([&](sycl::handler& h){

        sycl::accessor in(in_buf, h, read_only );

        int size = in_buf.size();

        h.single_task<KernelID>([=](){

            for(int i = 0 ; i < size; i++){

                ARProducePipe::write(in[i]);

            }




        });

    });

}




template<typename KernelID, typename Pipe>

event SubmitConsumerKernel(queue& q, buffer<float, 1>& out_buf){

    return q.submit([&](sycl::handler& h){

        sycl::accessor out(out_buf, h, write_only, no_init );

        int size = out_buf.size();

        h.single_task<KernelID>([=](){

            for(int i = 0 ; i < size; i++){

                out[i] =  ARConsumePipe::read();

            }




        });

    });

}




int main(){

    std::vector<float> in_data(10);

   

    for(int i = 0; i < 10; i++){

        in_data.push_back(i);

    }

   

    in_data.push_back(14);

    in_data.push_back(5);

    in_data.push_back(15);

    in_data.push_back(2);

    in_data.push_back(9);  

    sycl::buffer in_buf(in_data);

    sycl::buffer<float, 1> out_buf{1};

    queue q(selector, exception_handler);

    SubmitProducerKernel<ARProduceID, ARProducePipe>(q, in_buf);

    SubmitConsumerKernel<ARConsumeID, ARConsumePipe>(q, out_buf);

    // sycl::host_allocator result{out_buf};

    // std::cout<<result[0]<<std::endl;

}
