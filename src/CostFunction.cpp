#include "CostFunction.h"

CostFunction::CostFunction(){

}

template <typename T_out, typename T_in>
sycl::buffer<T_out, 1> CostFunction::Distance_Calc_fixed(sycl::queue &q, sycl::buffer<float, Dimension> &BF, int N, T_in ob_x, T_in ob_y, T_in a, T_in b)
{
    sycl::buffer<T_out, 1> BF_out{sycl::range{1}};
    sycl::buffer<T_in, Dimension> BF_in_fixed{sycl::range{Dimension}};
    auto out = q.submit([&](sycl::handler &h) {
        sycl::accessor float_in{BF, h};
        sycl::accessor fixed_in{BF_in_fixed, h, sycl::write_only, sycl::no_init};
        h.single_task<ConstructFromFloat>([=] {
            #pragma unroll
            for(int i=0; i<N; i++){
                #pragma unroll
                for(int j=0; j<Dimension; j++){
                    fixed_in[j][i] = T_in(float_in[j][i]);
                }
            }
        });
    });

    q.submit([&] (sycl::handler &h) {
        h.depends_on(out);
        sycl::accessor fixed_in{BF_in_fixed, h};
        sycl::accessor fixed_out{BF_out, h};
        h.single_task([=] {
            #pragma unroll
            for(int i=0; i<N; i++){
                T_in Px = 0.0f;
                T_in Py = 0.0f;
                T_out Px_sqr = 0.0f;
                T_out Py_sqr = 0.0f;
                //square for Px
                if(fixed_in[0][i] > ob_x){
                    Px = (fixed_in[0][i] - ob_x - a > 0) ? (fixed_in[0][i] - ob_x - a) : 0;
                }
                else{
                    Px = (ob_x - fixed_in[0][i] - a > 0) ? (ob_x - fixed_in[0][i] - a) : 0;
                }
                Px_sqr = Px*Px;
                //square for Py
                if(fixed_in[1][i] > ob_x){
                    Py = (fixed_in[1][i] - ob_y - b > 0) ? (fixed_in[1][i] - ob_y - b) : 0;
                }
                else{
                    Py = (ob_y - fixed_in[1][i] - b > 0) ? (ob_y - fixed_in[1][i] - b) : 0;
                }
                Py_sqr = Py*Py;

                //find the sqrt of Px^2 + Py^2
                fixed_out[i] = sqrt_fixed(Px_sqr + Py_sqr);
                
            }
        });
    });

    return BF_out;

}

// sycl::event CostFunction::Distance_Calc_float(sycl::queue &q, sycl::buffer<float, Dimension> &BF, sycl::buffer<float, 1> &BF_out, int N)
// {
//     float Obstacle_x_n = Obstacle_x;
//     float Obstacle_y_n = Obstacle_y;
//     float Obstacle_a_n = Obstacle_a;
//     float Obstacle_b_n = Obstacle_b;
//     sycl::event out = q.submit([&] (sycl::handler &h) {
//         sycl::accessor float_in{BF, h};
//         sycl::accessor float_out{BF_out, h};
//         h.single_task([=] {
//             #pragma unroll
//             for(int i=0; i<N; i++){
//                 //calculate square of px
//                 float Px = ((sycl::fdim(float_in[0][i], Obstacle_x_n) - Obstacle_a_n) > 0) ? (sycl::fdim(float_in[0][i], Obstacle_x_n) - Obstacle_a_n) : 0;
//                 //calculate square of py
//                 float Py = ((sycl::fdim(float_in[1][i], Obstacle_y_n) - Obstacle_b_n) > 0) ? (sycl::fdim(float_in[1][i], Obstacle_y_n) - Obstacle_b_n) : 0;

//                 //calculate sqrt(px^2 + py^2)
//                 float_out[i] = hypot(Px, Py);
//             }
//         });
//     });
//     return out;
// }

// sycl::event CostFunction::Velocity_Calc_float(sycl::queue &q, sycl::buffer<float, Dimension> &BF, sycl::buffer<float, 1> &BF_out, int N)
// {
//     sycl::event out = q.submit([&] (sycl::handler &h) {
//         sycl::accessor float_in{BF, h};
//         sycl::accessor float_out{BF_out, h};
//         h.single_task([=] {
//             #pragma unroll
//             for(int i=0; i<N; i++){
//                 //for x
//                 float x_dot = (sycl::ext::intel::fpga_reg(1) == 0) ? 0 : (float_in[0][i] - sycl::ext::intel::fpga_reg(float_in[0][i]));
//                 //for y
//                 float y_dot = (sycl::ext::intel::fpga_reg(1) == 0) ? 0 : (float_in[1][i] - sycl::ext::intel::fpga_reg(float_in[1][i]));

//                 float_out[i] = hypot(x_dot, y_dot);
//             }
//         });
//     });

//     return out;
// }


// This would be changed later for more complex cases
void CostFunction::Forward_Kinematics(void)
{
    // Define the basic cos and sin terms
    float COS_theta1 = sycl::cos(INPUT[0]);
    float SIN_theta1 = sycl::sin(INPUT[0]);
    float COS_theta2 = sycl::cos(INPUT[1]);
    float SIN_theta2 = sycl::sin(INPUT[1]);
    float COS_theta23 = sycl::cos(INPUT[1]+INPUT[2]);
    float SIN_theta23 = sycl::sin(INPUT[1]+INPUT[2]);

    // Find the positions
    // point1
    Cartesian_Pos[0][0] = L2 * COS_theta2 * COS_theta1;
    Cartesian_Pos[1][0] = L2 * COS_theta2 * SIN_theta1;
    Cartesian_Pos[2][0] = L1 + L2 * SIN_theta2;
    // point 2
    Cartesian_Pos[0][1] = (L2*COS_theta2 + L3*COS_theta23) * COS_theta1;
    Cartesian_Pos[1][1] = (L2*COS_theta2 + L3*COS_theta23) * SIN_theta1;
    Cartesian_Pos[2][1] = L1 + L2 * SIN_theta2 + L3 * SIN_theta23;

}

void CostFunction::Find_Distance_AR_float(void)
{
    #pragma unroll
    for(int i=0; i<N_B; i++){
        [[intel::fpga_register]] float square_term[Dimension];
        #pragma unroll
        for(int j=0; j<Dimension; j++){
            square_term[j] = (Cartesian_Pos[j][i]-Sphere_center[j])*(Cartesian_Pos[j][i]-Sphere_center[j]);
        }
        float square_add = square_term[0] + square_term[1] + square_term[2];
        Distance[i] = sycl::sqrt(square_add) - Sphere_radius;
    }
}

void CostFunction::Find_Velocity_AR_float(void)
{
    #pragma unroll
    for(int i=0; i<N_B; i++){
        [[intel::fpga_register]] float Diff_1st[Dimension];
        #pragma unroll
        for(int j=0; j<Dimension; j++){
            Diff_1st[j] = Cartesian_Pos[j][i] - sycl::ext::intel::fpga_reg(Cartesian_Pos[j][i]);
        }
        Velocity[i] = sycl::sqrt(Diff_1st[0]*Diff_1st[0] + Diff_1st[1]*Diff_1st[1] + Diff_1st[2]*Diff_1st[2]);
    }
}

void CostFunction::Find_Obstacle_cost_AR(void)
{
    #pragma unroll
    for(int i=0; i<Num_k; i++){
        #pragma unroll
        for(int j=0; j<DoF; j++){
            INPUT[j] = Input_to_Kernel[j][i];
        }
        CostFunction::Forward_Kinematics();
        CostFunction::Find_Distance_AR_float();
        CostFunction::Find_Velocity_AR_float();
        #pragma unroll
        for(size_t k=0; k<N_B; k++){
            if(epsilon_float+r_b_float-sycl::ext::intel::fpga_reg(Distance[k]) > 0){
                Out_of_Kernel[i] = sycl::ext::intel::fpga_reg(Out_of_Kernel[i]) + (epsilon_float+r_b_float-sycl::ext::intel::fpga_reg(Distance[k]))*sycl::ext::intel::fpga_reg(Velocity[k]);
            }
            else{
                Out_of_Kernel[i] = sycl::ext::intel::fpga_reg(Out_of_Kernel[i]);
            }
        }
    }
}