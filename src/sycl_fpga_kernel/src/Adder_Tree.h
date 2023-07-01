#ifndef __ADDER_TREE_H__
#define __ADDER_TREE_H__

#include <sycl/sycl.hpp>

//constexpr int N = 4;
//constexpr int LOG_N = 2;
class Adder_Tree{
    private:
        
    public:
        //sycl::buffer<float> BF_in{sycl::range{N}};
        //sycl::buffer<float, 2> BF{sycl::range<2>{LOG_N, N}};
        //int N = 8;
        //int LOG_N = 3;
        Adder_Tree();
        sycl::buffer<float, 1> Adder_Tree_Execute(sycl::queue &q, sycl::buffer<float, 2> &BF, int N, int LOG_N);
        

};

#endif