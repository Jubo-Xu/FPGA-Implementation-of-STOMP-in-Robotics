
# FPGA-Implementation-of-STOMP-in-Robotics

Welcome to our FPGA implementation of STOMP, 
This repositary is part of Imperial College London and intel PSG Robotics Technology Group in which the goal was to study the suitability of 
This repors


# FPGA-Implementation-of-STOMP-in-Robotics
Welcome to our FPGA implementation of STOMP, 
This repository is part of consultancy project made with  intel PSG Robotics Technology Group in which the goal was to study the suitability of motion planning algorithm into FPGA.

<center>
<img src="image/STOMP" style="width:400px;" />
<center>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

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
