#include "Adder_Tree.h"

Adder_Tree::Adder_Tree(){
    //BF{v};
    //sycl::buffer<float, 2> BF{sycl::range<2>{LOG_N, N}};
}

sycl::buffer<float, 1> Adder_Tree::Adder_Tree_Execute(sycl::queue &q, sycl::buffer<float, 2> &BF, int N, int LOG_N)
{
    //float output = 0.0f;
    //sycl::buffer BF{datain};
    sycl::buffer<float, 1> BF_out{sycl::range{1}};
    
    for(int j=0; j<LOG_N-1; j++){
      q.submit([&] (sycl::handler &h) {
      sycl::accessor bf{BF, h};
      h.parallel_for(N/(1<<(j+1)), [=](sycl::id<1> i) {
        bf[j+1][i] = bf[j][2*i] + bf[j][2*i+1];
      });
    });}
    q.submit([&] (sycl::handler &h) {
      sycl::accessor bf{BF, h};
      sycl::accessor bf_out{BF_out, h};
      h.single_task([=]() {
        bf_out[0] = bf[LOG_N-1][0] + bf[LOG_N-1][1];
      });
    });

    return BF_out;

    
}