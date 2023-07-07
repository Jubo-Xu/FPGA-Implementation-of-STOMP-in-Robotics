
# FPGA-Implementation-of-STOMP-in-Robotics
## Introduction 
This branch mainly contains the code for testing each individual block of FPGA implementation of STOMP in SYCL. For detailed information of this implementation please refer to [master branch](https://github.com/Jubo-Xu/FPGA-Implementation-of-STOMP-in-Robotics).

## How to Run it
This testing is mainly used to test the functionality of the code as well as having a basic estimation of RTL desgin based on the static report. Therefore, only FPGA emulation and FPGA static report of the oneAPI workflow are considered. For our testing, we have several test cases for testing each individual block and the combination of them, the list below shows these testcases:
1. delta theta block individually
2. obstacle cost function block individually
3. obstacle cost block and the delta theta block together
4. smooth cost block individually
5. determination block individually
6. rng block individually
7. all blocks are connected but only test for one iteration, no loop back

### Build
>**Note**: if we only consider about the fpga emulation, then we can just create one build folder and the make files will be replaced each time. But if we want to generate the report, although report is in one html file, it seems it cannot run individually, we have to have the whole build files for it to run, so it's better to have different build folder for different testing.

1. git clone the code of this branch into a folder
   ```bash
   git clone -b test https://github.com/Jubo-Xu/FPGA-Implementation-of-STOMP-in-Robotics.git
   ```
2. move into this folder and source the environment variables of oneapi
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
3. create a build folder for the testcase
   ```bash
   mkdir build_<variant>
   ```
   the ```<variant>``` is the name defined by user, for example, if want to test the delta theta block, we can create a build folder for it like this ```mkdir build_deltatheta```.
4. direct to the build folder created and cmake for the corresponding testcase
   ```bash
   cd build_<variant>
   cmake -DDOTEST=<index of test case> ..
   ```
### Run the code
1. under the build folder, generate the execution files
   ```bash
   # if want to do the fpga emulation
   make fpga_emu
   # if want to generate the report
   make report
   ```
2. run the execution file(for fpga emulation only)
   ```bash
   # test case 1
   ./do-test-deltatheta.fpga_emu
   # test case 2
   ./do-test-obstaclecost.fpga_emu
   # test case 3
   ./do-test-obscostanddeltatheta.fpga_emu
   # test case 4
   ./do-test-smoothcost.fpga_emu
   # test case 5
   ./do-test-determination.fpga_emu
   # test case 6
   ./do-test-rng.fpga_emu
   # test case 7
   ./do-test-whole.fpga_emu
   ```
## License

Code samples are licensed under the MIT license. See [LICENSE](https://github.com/Jubo-Xu/FPGA-Implementation-of-STOMP-in-Robotics/blob/master/LICENSE) for details.
