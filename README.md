
# FPGA-Implementation-of-STOMP-in-Robotics(Not Finished Yet)
test
not finished yet, RNG needs to be done, Chelosky Decomposition might be done later,
also need to try to connect to ROS

## Set Up
1. create a folder for build
```bash
mkdir build
cd build
```
2. generate the makefile for the main file, which contains the general structure
```bash
cmake ..
```
3. if want to generate makefile for the file that contains test for each hardware block(kernel)
```bash
cmake -DDOTEST=1 ..
```
4. do the emulation at this stage
```bash
make fpga_emu
```
5. execute the file
```bash
./stomp.fpga_emu
```
or
```bash
./do-test.fpga_emu
```


