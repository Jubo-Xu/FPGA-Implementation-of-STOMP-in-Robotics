
# FPGA-Implementation-of-STOMP-in-Robotics
## Introduction

Welcome to our FPGA implementation of [**STOMP**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5980280&tag=1)(Stochastic Trajectory Optimization for Motion Planning), 
This repositary is a part of consultancy Project with  intel  PSG Robotics Technology Group and Imperial College London. The goal of the project was to study the suitability of motion planning algorithm like STOMP or CHOMP when implemented into FPGA. This repositary contains a complete Implementation of STOMP on FPGA based on [**OneAPI**](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html) of Intel, which is based on [SYCL](https://www.khronos.org/sycl/). 

FPGA has lots of potential use in robotics field, like real-time and locally sensor fusion and data processing, as well as algorithm acceleration. This project mainly provides an example of this application. The full FPGA development testing  


For this project the  repositary a full FPGA devellopement testing the algorithm using ROS to 
<center>
<img src="image/STOMP_FPGA_architecture.png" " />
<center>




Stochastic Trajectory Optimization for Motion Planning (STOMP) is a probabilistic optimization framework (Kalakrishnan et al. 2011). STOMP produces smooth well behaved collision free paths within reasonable times. The approach relies on generating noisy trajectories to explore the space around an initial (possibly infeasible) trajectory which are then combined to produce an updated trajectory with lower cost. A cost function based on a combination of obstacle and smoothness cost is optimized in each iteration. No gradient information is required for the particular optimization algorithm that we use and so general costs for which derivatives may not be available (e.g. costs corresponding to constraints and motor torques) can be included in the cost function.

## implementation 
<center>
<img src="image/STOMP_FPGA_architecture.png" " />
<center>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


## How to run the code 

Here is an example LaTeX equation: \(x^2 + y^2 = z^2\).

## 1. Smoothness cost function 
Initially the smoothness cost function is given as $\frac{1}{2}\theta^TR\theta$ this matrix multiplication can be decompose to accelerate the computation.
The equations can be represented as:


\begin{align*}
\theta^TR\theta &= ( \theta_0 + \delta\theta)^TR( \theta_0 + \delta\theta) \\
&= \theta_o^TR\theta_o + \theta_o^TR\delta\theta \\
&= (\theta_o^{T}+\delta\theta^T)R(\theta_o+\delta\theta) \\
&= \theta_o^TR\theta_o+\theta_o^TR\delta\theta+\delta\theta^TR\theta_o+\delta\theta^T\delta\theta \\
\delta\theta &= M\delta\tilde{\theta} = \frac{1}{N}R^-\delta\tilde{\theta} \\
\theta_o^TR\delta\theta &= \theta_o^TR(\frac{1}{N}R^-\delta\tilde{\theta})= \frac{1}{N}\theta_o^T\delta\tilde{\theta} \\
\delta\theta^TR\theta &= \frac{1}{N}\delta\tilde{\theta}^T(R^-)^TR\theta_o =\frac{1}{N}\theta_o^T\delta\tilde{\theta} \\
\delta\theta^TR\delta\theta &= \frac{1}{N^2}(R^-\delta\tilde{\theta})^TR(R^-\delta\tilde{\theta}) = \frac{1}{N^2}\delta\tilde{\theta}^TR^-\delta\tilde{\theta}
\end{align*}


and can be implemented in hardware as: 


<center>
<img src="smoothness.svg" style="width:400px;" />
<center>
  
The sycl code is


the FPGA implemenation 

![test](path/to/image.png)

The repositary is made of 3 branch This repors
<pre><code class="language-bash">

</code></pre>
https://github.com/Jubo-Xu/FPGA-Implementation-of-STOMP-in-Robotics/blob/master/README.md
test



test
