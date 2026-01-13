


#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Eigen/src/Core/util/Constants.h"
#include "fftw3.h"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMap.h"
#include <numbers>


//#define C_SPEED 299792458.0
#define C_SPEED 2.99792e8

//FOR CHANGING PARAMETERS SUCH AS LENGTH OF CRYSTAL GOTO LINE 370
//FOR CHANGING INTEGRATION PARAMETERS NUM_samples and k_max GOTO LINE 440

//Calculates V_pump function at specified distance d, and beam waist omega_0 with k_p 
std::complex<double> calc_V_pump_at_dist(double_t qx, double_t qy , double omega_0, double d, double k_p){
    const std::complex<double> i(0.0, 1.0);
    double_t q_p_l2norm_squared = std::pow(qx,2)+std::pow(qy,2);
    std::complex<double> V_pump = std::exp(-q_p_l2norm_squared * std::pow(omega_0, 2.0)/4.0) * 
                                    std::exp(-1.0* i *q_p_l2norm_squared*d / (2.0* k_p));
    // exp(1i*k_p*d); constant factor left out here since it does not contribute to the coincidence rate later (p.8r)
    return V_pump;
}
 // for degenerate n2o_s and n2o_i are the same so instead of n2o_i n2o_s is used in this line
// prints frequency constants for type-1 phase matching is just used for debugging purposes.
void print_frequency_constants_t1(double_t bbo_length,double_t alpha_p, double_t eta_p, double_t gamma_p, double_t omega_p, double_t omega_i, double_t omega_s, double_t beta_p,
                                  double_t n_o_s, double_t n_o_i){
    double_t qsx_constant = bbo_length / 2.0 *(C_SPEED/(2.0*eta_p*omega_p) * std::pow(beta_p,2) - C_SPEED/(2.0*n_o_s*omega_s));
    double_t qix_constant = bbo_length / 2.0 *(C_SPEED/(2.0*eta_p*omega_p) * std::pow(beta_p,2) - C_SPEED/(2.0*n_o_i*omega_i));
    double_t qsy_constant = bbo_length / 2.0 *(C_SPEED/(2.0*eta_p*omega_p) * std::pow(gamma_p,2) - C_SPEED/(2.0*n_o_s*omega_s));
    double_t qiy_constant = bbo_length / 2.0 *(C_SPEED/(2.0*eta_p*omega_p) * std::pow(gamma_p,2) - C_SPEED/(2.0*n_o_i*omega_i));

    double_t qsx_linear = bbo_length*alpha_p /2.0;
    double_t qix_linear = bbo_length*alpha_p /2.0;
    
    double_t qx_mixed = bbo_length/2.0 *(C_SPEED/(eta_p*omega_p) *std::pow(beta_p, 2));
    double_t qy_mixed = bbo_length/2.0 *(C_SPEED/(eta_p*omega_p) *std::pow(gamma_p, 2));

    std::cout << "Printing Frequency constants for Type-1 phase matching (delta_k_z) \n";
    std::cout << "qsx_constant (Quadratic Terms): " << qsx_constant <<"\t qix_constant: " << qix_constant <<"\t qsy_constant: " << qsy_constant <<"\t qiy_constant: " << qiy_constant <<"\n";
    std::cout << "qsx_linear: " << qsx_linear << "\t qix_linear: " << qix_linear << "\n";
    std::cout << "qx_mixed: " << qx_mixed << "\t qy_mixed: " << qy_mixed << "\n";

}
void print_frequency_constants_t2(double_t bbo_length, double_t alpha_p, double_t alpha_s, double_t eta_p, double_t eta_s, double_t gamma_p, double_t gamma_s, double_t omega_p, 
                                  double_t omega_s, double_t omega_i, double_t beta_p, double_t beta_s, double_t n_o_i){
    double_t qsx_linear = bbo_length*(alpha_p - alpha_s)/2.0;
    double_t qix_linear = bbo_length*alpha_p/2.0;

    double_t qsx_quad = bbo_length/2.0 *(-C_SPEED/(2.0*eta_s*omega_s) * std::pow(beta_s,2) + C_SPEED/(2.0*eta_p*omega_p) * std::pow(beta_p,2));
    double_t qsy_quad = bbo_length/2.0 *(-C_SPEED/(2.0*eta_s*omega_s) * std::pow(gamma_s,2) + C_SPEED/(2.0*eta_p*omega_p) * std::pow(gamma_p,2));
    double_t qix_quad = bbo_length/2.0 *(-C_SPEED/(2.0*n_o_i*omega_i) + C_SPEED/(2.0*eta_p*omega_p) * std::pow(beta_p,2));
    double_t qiy_quad = bbo_length/2.0 *(-C_SPEED/(2.0*n_o_i*omega_i) + C_SPEED/(2.0*eta_p*omega_p) * std::pow(gamma_p,2));

    double_t qx_mixed = bbo_length/2.0 *(C_SPEED/(eta_p*omega_p) * std::pow(beta_p,2));
    double_t qy_mixed = bbo_length/2.0 *(C_SPEED/(eta_p*omega_p) * std::pow(gamma_p,2));

    std::cout << "Printing Frequency constants for Type-2 phase matching (delta_k_z) \n";
    std::cout << "qsx_constant (Quadratic Terms): " << qsx_quad <<"\t qix_constant: " << qix_quad <<"\t qsy_constant: " << qsy_quad <<"\t qiy_constant: " << qiy_quad <<"\n";
    std::cout << "qsx_linear: " << qsx_linear << "\t qix_linear: " << qix_linear << "\n";
    std::cout << "qx_mixed: " << qx_mixed << "\t qy_mixed: " << qy_mixed << "\n";

}
//these are runtime constants for faster computation of the phase_matching k-term, which will be used in the calc_k_term_fast version 
//if you just want to look at the k-term implementation you should look at calc_k_term 
double_t calc_linear_constant(double_t n_o_s, double_t n_o_i, double_t omega_s, double_t omega_i, double_t eta_p, double_t omega_p){
    return n_o_s*omega_s/C_SPEED + n_o_i*omega_i/C_SPEED - eta_p*omega_p/C_SPEED;
}
double_t calc_qpx_sq_constant(double_t eta_p, double_t omega_p, double_t beta_p){
    return C_SPEED/(2*eta_p*omega_p)*(std::pow(beta_p,2));
}
double_t calc_qpy_sq_constant(double_t eta_p, double_t omega_p, double_t gamma_p){
    return C_SPEED/(2*eta_p*omega_p)*(std::pow(gamma_p,2));
}
double_t calc_qs_constant(double_t n_o_s, double_t omega_s){
return C_SPEED/(2*n_o_s*omega_s);
}
double_t calc_qi_constant(double_t n_o_i, double_t omega_i){
    return C_SPEED/(2*n_o_i*omega_i);
}
//runtime constants for k2-kterm calculations speed up is about ~3x so you should use it. 
//
double_t calc_linear_constants_k2(double_t eta_s, double_t eta_p, double_t n_o_i, double_t omega_s, double_t omega_i, double_t omega_p ){
return eta_s*omega_s/C_SPEED + n_o_i*omega_i/C_SPEED - eta_p * omega_p/C_SPEED;
}
double_t calc_qpx_sq_constant_k2(double_t eta_p, double_t omega_p, double_t beta_p){
    return C_SPEED/(2.0*eta_p*omega_p) * (std::pow(beta_p, 2));
}
double_t calc_qpy_sq_constant_k2(double_t eta_p, double_t omega_p, double_t gamma_p){
return C_SPEED/(2.0*eta_p*omega_p) * (std::pow(gamma_p, 2));
}
double_t calc_qi_constant_k2(double_t n_o_i, double_t omega_i){
    return - C_SPEED / (2.0*n_o_i*omega_i);
}
double_t calc_qsx_sq_constant_k2(double_t eta_s, double_t omega_s, double_t beta_s){
return - C_SPEED/(2.0*eta_s*omega_s) * (std::pow(beta_s,2));
}
double_t calc_qsy_sq_constant_k2(double_t eta_s, double_t omega_s, double_t gamma_s){
return - C_SPEED/(2.0*eta_s*omega_s) * (std::pow(gamma_s,2));
}

//Normal k-term calculation for type-1 phase matching 
double_t calc_k_term(double_t qsx, double_t qix, double_t qsy, double_t qiy, double_t theta_p, double_t omega_s, double_t omega_i, double_t n_o_i, double_t n_o_s, 
                     double_t alpha_p, double_t beta_p, double_t eta_p, double_t gamma_p){
    double_t omega_p = omega_s + omega_i;

    double_t qpx = qsx + qix;
    double_t qpy = qsy + qiy;
    double_t qpx_sq = std::pow(qpx, 2);
    double_t qpy_sq = std::pow(qpy, 2);
    double_t qs_norm_sq = std::pow(qsx,2) + std::pow(qsy,2);
    double_t qi_norm_sq = std::pow(qix,2) + std::pow(qiy,2);

    

    double_t delta_k_z = n_o_s*omega_s/C_SPEED + n_o_i*omega_i/C_SPEED - eta_p*omega_p/C_SPEED + C_SPEED/(2*eta_p*omega_p)*(std::pow(beta_p,2)*qpx_sq 
        + std::pow(gamma_p,2)*qpy_sq) + 
        alpha_p*(qsx+qix) - C_SPEED/(2*n_o_s*omega_s)*qs_norm_sq - C_SPEED/(2*n_o_i*omega_i)*qi_norm_sq;
    return delta_k_z;
}
//Fast implementation of k-term type-1 phase matching
double_t calc_k_term_fast(double_t linear_constant, double_t qpx_sq_constant, double_t qpy_sq_constant, double_t alpha_p, double_t qs_constant, double_t qi_constant,
                          double_t qsx, double_t qix, double_t qsy, double_t qiy){
    double_t qpx = qsx + qix;
    double_t qpy = qsy + qiy;
    double_t qpx_sq = std::pow(qpx, 2);
    double_t qpy_sq = std::pow(qpy, 2);
    double_t qs_norm_sq = std::pow(qsx,2) + std::pow(qsy,2);
    double_t qi_norm_sq = std::pow(qix,2) + std::pow(qiy,2);



    double_t delta_k_z = linear_constant + qpx_sq_constant*qpx_sq + qpy_sq_constant*qpy_sq + alpha_p*(qsx+qix) - qs_constant*qs_norm_sq - qi_constant*qi_norm_sq;
    return delta_k_z;
}
//type-2 k-term phase_matching 
double_t calc_k_term2(double_t qsx, double_t qix, double_t qsy, double_t qiy, double_t theta_p, double_t omega_s, double_t omega_i, double_t n_o_i, double_t n_o_s, 
                      double_t alpha_p, double_t beta_p, double_t eta_p, double_t gamma_p, double_t alpha_s, double_t beta_s, double_t eta_s, double_t gamma_s){
    double_t omega_p = omega_s + omega_i; 

    double_t qpx = qsx + qix;
    double_t qpy = qsy + qiy;
    double_t qpx_sq = std::pow(qpx, 2);
    double_t qpy_sq = std::pow(qpy, 2);
    double_t qs_norm_sq = std::pow(qsx,2) + std::pow(qsy,2);
    double_t qi_norm_sq = std::pow(qix,2) + std::pow(qiy,2);

    double_t delta_k_z = - alpha_s*qsx + eta_s*omega_s/C_SPEED  
        - C_SPEED/(2.0*eta_s*omega_s) * (std::pow(beta_s,2)*std::pow(qsx,2) + std::pow(gamma_s,2)*std::pow(qsy,2))
        + n_o_i*omega_i/C_SPEED - C_SPEED*qi_norm_sq / (2.0*n_o_i*omega_i)
        + alpha_p * (qix+qsx) - eta_p * omega_p/C_SPEED
        + C_SPEED/(2.0*eta_p*omega_p) * (std::pow(beta_p, 2)*qpx_sq + std::pow(gamma_p,2)*qpy_sq);


    return delta_k_z;

}
//fast type-2 k-term phase matching 
double_t calc_k_term2_fast(double_t linear_constant, double_t qpx_sq_constant, double_t qpy_sq_constant, double_t qi_constant, double_t qsx_sq_constant, double_t qsy_sq_constant, 
                        double_t qsx, double_t qix, double_t qsy, double_t qiy,  
                        double_t alpha_p,  double_t alpha_s ){
    
    double_t qpx = qsx + qix;
    double_t qpy = qsy + qiy;
    double_t qpx_sq = std::pow(qpx, 2);
    double_t qpy_sq = std::pow(qpy, 2);
    //double_t qs_norm_sq = std::pow(qsx,2) + std::pow(qsy,2);
    double_t qi_norm_sq = std::pow(qix,2) + std::pow(qiy,2);


    double_t delta_k_z_fast = linear_constant + qpx_sq_constant*qpx_sq + qpy_sq_constant*qpy_sq - alpha_s*qsx + alpha_p*(qsx+qix) +
        qi_constant*qi_norm_sq + qsx_sq_constant*std::pow(qsx,2) + qsy_sq_constant*std::pow(qsy,2);

    return delta_k_z_fast;

}

//sinc function can also import boost math, which is faster
std::double_t calc_sinc(double_t x){
    if(x == 0) return 1;
    return sin(x)/x;
}
//actual phase matching function is the same for type-1 and type-2 however the delta_k_z Term is different.
std::complex<double> calc_phase_match_t1(double delta_k_z, double bbo_length){
    const std::complex<double> i(0.0, 1.0);
    //for phi to be less than e^-5 (at max because of sinc) -> delta_k_z must be higher than (2/(e^-5)) -> 296.8
    std::complex<double> phi = bbo_length * 
        calc_sinc(std::numbers::pi*delta_k_z*bbo_length/2.0) * //pi added here to make it into the normalized version of sinc. Also using boos::math::sinc is faster than sinc used here. So uses that if you want it to be faster.
        std::exp(1.0 * i * delta_k_z * bbo_length/2.0);
    return phi;

}

std::complex<double> calc_fft_integrand(double qix, double qiy, double delta_qx, double delta_qy, double omega_0, double bbo_length, double d,  double k_p, double k_s,
                                        double k_i, double z, const std::vector<double> abcd_constants, const std::vector<double> abcd_constants_s, double_t omega_s, double_t omega_i, double no_s, double_t theta_p,
                                        uint32_t type_select){
    const std::complex<double> i(0.0, 1.0);
    double_t qsx = -qix + delta_qx;
    double_t qsy = -qiy + delta_qy;
    std::complex<double> fft_integrand = {0,0};
    //std::vector<double> q_p = {qsx+qix, qsy+qiy};
    /*fft_integrand = calc_V_pump_at_dist(q_p, omega_0, d, k_p) * std::exp(1.0*i *(k_s+k_i)*z) * calc_phi(sx, sy, ix, iy, bbo_length, abcd_constants, omegas, no_s)
                    * std::exp(1.0*i *(- (std::pow(sx, 2.0)+std::pow(sy, 2.0))*z / (2.0* k_s) - (std::pow(ix,2.0)+std::pow(iy,2.0))*z /(2.0*k_i) ));
                    */
    double_t n_o_s = no_s;
    double_t n_o_i = no_s;
    double_t alpha_p = abcd_constants[0];
    double_t beta_p = abcd_constants[1];
    double_t gamma_p = abcd_constants[2];
    double_t eta_p = abcd_constants[3];

    double_t alpha_s = abcd_constants_s[0];
    double_t beta_s = abcd_constants_s[1];
    double_t gamma_s = abcd_constants_s[2];
    double_t eta_s = abcd_constants_s[3];
    double_t delta_k_term;
    double_t delta_k_term_i;
    if (type_select == 2){
    //double_t delta_k_term = calc_k_term(qsx, qix, qsy, qiy, theta_p, omega_s, omega_i, n_o_i, n_o_s, alpha_p, beta_p, eta_p, gamma_p);
    delta_k_term = calc_k_term2(qsx, qix, qsy, qiy, theta_p, omega_s, omega_i, n_o_i, n_o_s, alpha_p, beta_p, eta_p, gamma_p, alpha_s, beta_s, eta_s, gamma_s);
    delta_k_term_i = calc_k_term2(qix, qsx, qiy, qsy, theta_p, omega_i, omega_s, n_o_s, n_o_i, alpha_p, beta_p, eta_p, gamma_p, alpha_s, beta_s, eta_s, gamma_s);
    fft_integrand = calc_V_pump_at_dist(qsx+qix, qsy+qiy, omega_0, d, k_p)* std::exp(1.0*i*(k_s+k_i)*z) 
        * (calc_phase_match_t1(delta_k_term, bbo_length)  + calc_phase_match_t1(delta_k_term_i, bbo_length) )
        //* delta_k_term
        * std::exp(1.0*i *(- (std::pow(qsx, 2.0)+std::pow(qsy, 2.0))*z / (2.0* k_s) - (std::pow(qix,2.0)+std::pow(qiy,2.0))*z /(2.0*k_i) ));
    }
    else {
    delta_k_term = calc_k_term(qsx, qix, qsy, qiy, theta_p, omega_s, omega_i, n_o_i, n_o_s, alpha_p, beta_p, eta_p, gamma_p);
    fft_integrand = calc_V_pump_at_dist(qsx+qix, qsy+qiy, omega_0, d, k_p)* std::exp(1.0*i*(k_s+k_i)*z) 
        * calc_phase_match_t1(delta_k_term, bbo_length)
        //* delta_k_term
        * std::exp(1.0*i *(- (std::pow(qsx, 2.0)+std::pow(qsy, 2.0))*z / (2.0* k_s) - (std::pow(qix,2.0)+std::pow(qiy,2.0))*z /(2.0*k_i) ));

    }
    //std::cout << delta_k_term << "\t";
    //std::cout << delta_qx << ", " << delta_qy << ", " << omega_0 << ", " << d << ", " << k_p << "\n";
    return fft_integrand;
}
std::complex<double> calc_fft_integrand_fast(double qix, double qiy, double delta_qx, double delta_qy, double omega_0, double bbo_length, double d,  double k_p, double k_s,
                                        double k_i, double z, double_t linear_constant, double_t qpx_sq_constant, double_t qpy_sq_constant, double_t alpha_p, 
                                        double_t qs_constant, double_t qi_constant, double_t omega_s, double_t omega_i, double no_s, double_t theta_p,
                                        double_t linear_constant_k2, double_t qpx_sq_constant_k2, double_t qpy_sq_constant_k2, double_t qi_constant_k2,
                                        double_t qsx_sq_constant_k2, double_t qsy_sq_constant_k2, double_t alpha_s,  uint32_t type_select){
    const std::complex<double> i(0.0, 1.0);
    //constructs qsx from qix and delta_qx rather than just using a linear array for both 
    double_t qsx = -qix + delta_qx;
    double_t qsy = -qiy + delta_qy;
    std::complex<double> fft_integrand = {0,0};
    //std::vector<double> q_p = {qsx+qix, qsy+qiy};
    /*fft_integrand = calc_V_pump_at_dist(q_p, omega_0, d, k_p) * std::exp(1.0*i *(k_s+k_i)*z) * calc_phi(sx, sy, ix, iy, bbo_length, abcd_constants, omegas, no_s)
                    * std::exp(1.0*i *(- (std::pow(sx, 2.0)+std::pow(sy, 2.0))*z / (2.0* k_s) - (std::pow(ix,2.0)+std::pow(iy,2.0))*z /(2.0*k_i) ));
                    */
    double_t n_o_s = no_s;
    double_t n_o_i = no_s;
    //For type-2 phase matching delta_k_term_i corresponds to the delta_k value for the idler. 
    double_t delta_k_term;
    double_t delta_k_term_i;
    if (type_select == 2){ //type == 2 -> type-2 phase matching -> computes both signal(like regular type-1) and idler. And since Fourier transform is linear we can 
        //write in_data = v_pump *(phase_match(k1) + phase_match(k2)); instead of computing FFT for phase_match(k1) and phase_match(k2) and then adding the results together.

        //double_t delta_k_term = calc_k_term(qsx, qix, qsy, qiy, theta_p, omega_s, omega_i, n_o_i, n_o_s, alpha_p, beta_p, eta_p, gamma_p);
        //above term would be the non fast implementation (for delta_k_term_i just flip qsx and qix, as well as qsy and qiy aka flip signal and idler);
        double_t delta_k_term = calc_k_term2_fast(linear_constant_k2, qpx_sq_constant_k2, qpy_sq_constant_k2,
                                                  qi_constant_k2, qsx_sq_constant_k2, qsy_sq_constant_k2,
                                                      qsx, qix, qsy, qiy,   alpha_p, alpha_s);
        double_t delta_k_term_i = calc_k_term2_fast(linear_constant_k2, qpx_sq_constant_k2, qpy_sq_constant_k2,
                                                  qi_constant_k2, qsx_sq_constant_k2, qsy_sq_constant_k2,
                                                      qix, qsx, qiy, qsy,   alpha_p, alpha_s);

        fft_integrand = calc_V_pump_at_dist(qsx+qix, qsy+qiy, omega_0, d, k_p)* std::exp(1.0*i*(k_s+k_i)*z) 
        * (calc_phase_match_t1(delta_k_term, bbo_length) + calc_phase_match_t1(delta_k_term_i, bbo_length))
        * std::exp(1.0*i *(- (std::pow(qsx, 2.0)+std::pow(qsy, 2.0))*z / (2.0* k_s) - (std::pow(qix,2.0)+std::pow(qiy,2.0))*z /(2.0*k_i) ));

        }
    else { //type-1 case (standard case)
        delta_k_term = calc_k_term_fast(linear_constant, qpx_sq_constant, qpy_sq_constant, alpha_p, qs_constant, qi_constant,
                                    qsx, qix, qsy, qiy);
        fft_integrand = calc_V_pump_at_dist(qsx+qix, qsy+qiy, omega_0, d, k_p)* std::exp(1.0*i*(k_s+k_i)*z) 
        * calc_phase_match_t1(delta_k_term, bbo_length)
        * std::exp(1.0*i *(- (std::pow(qsx, 2.0)+std::pow(qsy, 2.0))*z / (2.0* k_s) - (std::pow(qix,2.0)+std::pow(qiy,2.0))*z /(2.0*k_i) ));

    }
    return fft_integrand;
}
//class for calculating alpha, beta, gamma and eta
class calc_const {
public:
    calc_const(double theta_p, double n2o, double n2e);
    calc_const(calc_const &&) = default;
    calc_const(const calc_const &) = default;
    calc_const &operator=(calc_const &&) = default;
    calc_const &operator=(const calc_const &) = default;
    ~calc_const();
    
    const std::vector<double> return_constants() const;

private:
    double theta_p, n2o, n2e, no, ne; // Inputs local variables 
    double alpha, beta, gamma, eta;
    std::vector<double> const_vec{0, 0, 0, 0};
};

calc_const::calc_const(double theta_p, double n2o, double n2e) {
    this->theta_p = theta_p; //theta in RADIAN
    this->n2o = n2o;
    this->n2e = n2e;
    this->no = std::sqrt(n2o);
    this->ne = std::sqrt(n2e);

    this->alpha = (this->n2o-this->n2e) * std::sin(this->theta_p) * std::cos(this->theta_p) / (this->n2o * std::pow(std::sin(this->theta_p), 2) +
                                                                                               this->n2e * std::pow(cos(this->theta_p), 2));
    this->beta = (no*ne) / (this->n2o * std::pow(std::sin(this->theta_p), 2) + this->n2e * std::pow(std::cos(this->theta_p), 2));
    this->gamma = (no) / std::sqrt(this->n2o * std::pow(std::sin(theta_p),2) + this->n2e * std::pow(std::cos(theta_p),2));
    this->eta = (no*ne) / std::sqrt(this->n2o * std::pow(std::sin(this->theta_p), 2) + this->n2e * std::pow(std::cos(this->theta_p), 2));  
    this->const_vec[0]=(this->alpha);
    this->const_vec[1]=(this->beta);
    this->const_vec[2]=(this->gamma);
    this->const_vec[3]=(this->eta);
}

const std::vector<double> calc_const::return_constants() const {
    return this->const_vec;}

calc_const::~calc_const() {
}
//Calculates the refractive index for a given lambda (ordinary and extraordinary case)
void calc_n2(double lambda,double* ret_n2o, double* ret_n2e){
    double_t lambda_l = lambda * std::pow(10, 6);
    *ret_n2o = 2.7405 + 0.0184/(std::pow(lambda_l, 2.0) - 0.0179) - 0.0155*std::pow(lambda_l, 2.0);
    *ret_n2e = 2.3730 + 0.0128/(std::pow(lambda_l, 2.0) - 0.0156) - 0.0044*std::pow(lambda_l, 2.0);
}

double calc_theta_p(double theta_p_0, double alpha, double n_pe){
    double theta_p = theta_p_0 + asin(sin(alpha)/n_pe); //radian version of -> ALPHA IS IN RADIAN
    return theta_p;
}




int main (int argc, char *argv[]) {
    //std::vector<std::complex<double>> v_pump = 
    //Defining Constants
    double_t input_theta = 0;
    std::cout << "Please type in Theta_p as an angle: \n";
    std::cin >> input_theta;
    double_t input_alpha = 0;
    std::cout << "Please type in alpha as an angle (if you don't know just leave as 0.0): \n";
    std::cin >> input_alpha;
    std::cout << "Please type in File_name: \n";
    std::string file_name;
    std::cin >> file_name;
    uint32_t type_select = 0;
    std::cout << "Please choose the type of phase_matching (Input 1 for type-1; and 2 for type-2): \n";
    std::cin >> type_select;

    const double_t bbo_length = 0.002; //Length of bbo crystal in meter (Values taken from Experimental and Numreical results 5.)
    const double_t beam_distance = 1078e-3; //107,8 cm distance of beam waist to crystal
    const double_t omega_0 = 388e-6; //beam waist 388um
    const double_t lambda = 405e-9; //wavelength of incoming pump photon
    const double_t lambda_down = 2.0* lambda; //downconverted photon at twice the wavelength of pump photon (assumed here to be degenerate downconversion)
    const double_t z = 0.035; //Distance of measurement to crystal 35mm
    const double_t theta_p_0 =  input_theta * std::numbers::pi/180.0; //Optical axis angle (Figure 4.)
    const double_t alpha = input_alpha * std::numbers::pi/180.0; //Non-normal incidence angle of pump photon (Figure 4.)
    double_t n2o_p = 0; //ordinary and extraordinary index of refraction for pump, signal and idler (n2 means n^2)
    double_t n2e_p = 0;
    double_t n2o_s = 0;
    double_t n2e_s = 0;
    double_t n2o_i = 0;
    double_t n2e_i = 0;
    calc_n2(lambda, &n2o_p, &n2e_p);
    calc_n2(lambda*2.0, &n2o_s, &n2e_s);
    calc_n2(lambda*2.0, &n2o_i, &n2e_i);
    
    //Non-squared refraction index
    const double_t no_p = sqrt(n2o_p);
    const double_t ne_p = sqrt(n2e_p);
    const double_t no_s = sqrt(n2o_s);
    const double_t ne_s = sqrt(n2e_s);
    const double_t no_i = sqrt(n2o_i);
    const double_t ne_i = sqrt(n2e_i);
    
    //actual theta_p used in calculations from alpha, theta_p_0 and ne_p
    const double_t theta_p = calc_theta_p(theta_p_0, alpha, ne_p);
    
    //alpha, beta, eta and gamma constants as used in delta_k_z calculations in (89, 94 and 99). Here const_object_s denounces constants for signal instead of pump
    calc_const const_object(theta_p, n2o_p, n2e_p);
    calc_const const_object_s(theta_p, n2o_s, n2e_s);
    
    const std::vector<double> abcd_constants = const_object.return_constants();
    const std::vector<double> abcd_constants_s = const_object_s.return_constants();
    std::cout << "Constants : alpha_p: " <<abcd_constants[0] << "\tbeta_p: " << abcd_constants[1] << "\tgamma_p: " << abcd_constants[2] << "\teta_p: " << abcd_constants[3] << "\n\n";
    const double_t omega_i = (2.0* std::numbers::pi * C_SPEED) / lambda_down;
    const double_t omega_s = (2.0* std::numbers::pi * C_SPEED) / lambda_down;
    const double_t omega_p = (2.0* std::numbers::pi * C_SPEED) / lambda; //omega_p = omega_s+omega_i
    const double_t k_p = 2.0*std::numbers::pi/lambda; //k-vector magnitude for pump, signal and idler
    const double_t k_s = 2.0*std::numbers::pi /(lambda_down);
    const double_t k_i = 2.0*std::numbers::pi/(lambda_down);
    
    //const double_t omega_s = omega_p /2;
    //const double_t omega_i = omega_p /2; //for degenerate downconversion (p. 15l)
    //abcd_constants[3] is eta_p 
    std::cout << "Constant factor for q_max estimation: " << (no_s*omega_s/C_SPEED + no_i*omega_i/C_SPEED - abcd_constants[3]*omega_p/C_SPEED)*bbo_length*0.5 << "\n";
    std::cout << "omega_i debug: " << std::format("{}", omega_i) << "\n";
    std::cout << "k_p: " << std::format("{}", k_p) << "\t k_i: " << std::format("{}", k_i) << "\n";
    std::vector<double> omegas = {omega_p, omega_s, omega_i};
    //abschätzen wie viele fft schritte speicher was würde man brauchen  warum hats nicht funktioniert.
    
    print_frequency_constants_t1(bbo_length, abcd_constants[0], abcd_constants[3], abcd_constants[2], omega_p, omega_i, omega_s, abcd_constants[1], no_s,  no_i);
    print_frequency_constants_t2(bbo_length, abcd_constants[0], abcd_constants_s[0], abcd_constants[3], abcd_constants_s[3], abcd_constants[2], abcd_constants_s[2], omega_p, omega_s, omega_i, abcd_constants[1], abcd_constants_s[1], no_i);

    double_t const linear_constant = calc_linear_constant(no_s, no_i, omega_s, omega_i, abcd_constants[3], omega_p);
    double_t const qpx_sq_constant = calc_qpx_sq_constant(abcd_constants[3], omega_p, abcd_constants[1]);
    double_t const qpy_sq_constant = calc_qpy_sq_constant(abcd_constants[3], omega_p, abcd_constants[2]);
    double_t const qs_constant = calc_qs_constant(no_s, omega_s);
    double_t const qi_constant = calc_qi_constant(no_i, omega_i);
    
    //constants for sample_count estimation; multiply the square and qi_constant by your q_max 
    double_t const linear_constant_k2 = calc_linear_constants_k2(abcd_constants_s[3], abcd_constants[3], no_i, omega_s, omega_i, omega_p);
    double_t const qpx_sq_constant_k2 = calc_qpx_sq_constant_k2(abcd_constants[3], omega_p, abcd_constants[1]);
    double_t const qpy_sq_constant_k2 = calc_qpy_sq_constant_k2(abcd_constants[3], omega_p, abcd_constants[2]);
    double_t const qi_constant_k2 = calc_qi_constant_k2(no_i, omega_i);
    double_t const qsx_sq_constant_k2 = calc_qsx_sq_constant_k2(abcd_constants_s[3], omega_s, abcd_constants_s[1]);
    double_t const qsy_sq_constant_k2 = calc_qsy_sq_constant_k2(abcd_constants_s[3], omega_s, abcd_constants_s[2]);

    //IF YOU Encounter an issue where the picture seems to fade away at the edges. It will most likely be that you need to increase the max momentum. For this either only  increase
    //momentum_span_wide_x/y or momentum_span_wide and momentum_span_narrow by similar factors.
    //KEEP IN MIND that if you increase the max_momentum you will need to increase the number of samples by the square of that factor. (There is prob. something in the sample_count estimation
    // that i missed)
    //
    // Momentum span is just a prefactor of k_vector: k_max (the max you "integrate" to) = momentum_span_wide * k_i / k_s
    double_t momentum_span_wide_x = 0.045; //0.045
    double_t momentum_span_wide_y = 0.045; //0.045
    double_t momentum_span_narrow_x = 0.035; //0.035
    double_t momentum_span_narrow_y = 0.035; //0.035
    
    //For Type-2 you need to go to some larger max_momentum
    if(type_select == 2){
        momentum_span_wide_y *= 1.2; //1.5 when you increase the mommentum span by a factor you need to increase the sample_count by the square of the factor i.e momentum*=1.5 -> samples*=(1.5)^2 -> 2.25
    }
    else{
        momentum_span_wide_y *= 1.0;
        momentum_span_narrow_y *= 1.0;
        momentum_span_wide_x *= 1.0;
        momentum_span_narrow_x *= 1.0;
    }
    double_t qx = k_i * momentum_span_wide_x;
    double_t qy = k_i * momentum_span_wide_y;
    double_t dqx = k_s * momentum_span_narrow_x;
    double_t dqy = k_s * momentum_span_narrow_y;

    uint32_t num_samples_wide_x = 800; //800 for max
    uint32_t num_samples_wide_y = 800;
    uint32_t num_samples_narrow_x = 40;
    uint32_t num_samples_narrow_y = 40;

    if(type_select == 2){
        //CAN get away with only using 25 for narrow is good because then only 5 GB of Ram
        num_samples_wide_x = 800; //800
        num_samples_wide_y = 1200; //1600
        num_samples_narrow_x = 25;
        num_samples_narrow_y = 25;
    }
    uint64_t total_samples = (uint64_t) num_samples_wide_x*num_samples_wide_y*num_samples_narrow_x*num_samples_narrow_y;
    double_t memory_consumption = (double_t) total_samples / (1e09/8.0);
    
    std::cout << "Estimated Memory Consumption (Total): " << memory_consumption << " GB\n";

    Eigen::ArrayXd qx_array = Eigen::ArrayXd::LinSpaced(num_samples_wide_x, -qx,qx);
    Eigen::ArrayXd qy_array = Eigen::ArrayXd::LinSpaced(num_samples_wide_y, -qy, qy); 
    Eigen::ArrayXd dqx_array = Eigen::ArrayXd::LinSpaced(num_samples_narrow_x, -dqx, dqx); 
    Eigen::ArrayXd dqy_array = Eigen::ArrayXd::LinSpaced(num_samples_narrow_y, -dqy, dqy); 

    //pls->line(num_samples_wide_x, qx_array.data(), qx_array.data());

    /*std::cout  << "\nqx_array: \n" << qx_array << "\n\n";
    std::cout  << "\nqy_array: \n" << qy_array << "\n\n";
    std::cout  << "\ndqx_array: \n" << dqx_array << "\n\n";
    std::cout  << "\ndqy_array: \n" << dqy_array << "\n\n";
    */
    std::cout << "theta_p: " << theta_p << "\n\n";
    std::cout << "qx span: " << qx << "\n";
    std::cout << "qy span: " << qy << "\n";

    int n_pointer[4];
    n_pointer[0] = num_samples_wide_x;
    n_pointer[1] = num_samples_wide_y;
    n_pointer[2] = num_samples_narrow_x;
    n_pointer[3] = num_samples_narrow_y;
   
    //std::variant<Eigen::Tensor<std::complex<float>, 4, Eigen::RowMajor>, Eigen::Tensor<double_t, 4, Eigen::RowMajor>> tensor_variant, tensor_double;
    //tensor_variant = Eigen::Tensor<std::complex<float>, 4, Eigen::RowMajor>(num_samples_wide_x, num_samples_wide_y, num_samples_narrow_x, num_samples_narrow_y);
    //tensor_double = Eigen::Tensor<double_t, 4, Eigen::RowMajor>(num_samples_wide_x,num_samples_wide_y,num_samples_narrow_x,num_samples_narrow_y);
    Eigen::Tensor<std::complex<float>, 4, Eigen::RowMajor> in_tensor(num_samples_wide_x,num_samples_wide_y,num_samples_narrow_x,num_samples_narrow_y);
    //Eigen::Tensor<std::complex<double_t>, 4, Eigen::RowMajor> in_tensor = tensor_variant._Storage()._Get();
    fftwf_plan p;
    size_t size = num_samples_wide_x;
    //double_t k_max = 178;
    //double_t multiplier = k_max/N;
    
    printf("starting malloc and input Data creation. May take a while.\n");
    //in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N*N*N*N);
    //out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N*N*N*N);
    p = fftwf_plan_dft(4,n_pointer, reinterpret_cast<fftwf_complex*>(in_tensor.data()), reinterpret_cast<fftwf_complex*>(in_tensor.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    auto start = std::chrono::high_resolution_clock::now(); 
   // concurrency::parallel_for (size_t(0), size, [&] (size_t i) {
   
    //Progress estimation:
    double_t progress_inc = 20.0/num_samples_wide_x;
    //set background color
    printf("Current Progress: Input Data: \n");
    printf("\033[47m");
    for(int i = 0; i < 20; i++){
        printf(" ");
    }
    printf("\033[46m");
    for (int i = 0; i < num_samples_wide_x; i++){
        printf("\r");
        //progress bar
        for(int p = 0; p < i*progress_inc; p++){
            printf(" ");
        }
        fflush(stdout);
        for (int j = 0 ; j < num_samples_wide_y; j++) {
            for (int k = 0; k < num_samples_narrow_x; k++){
                for (int l = 0; l < num_samples_narrow_y; l++){
                    std::complex<double> fft_integrand = calc_fft_integrand_fast(qx_array(i), qy_array(j), dqx_array(k), dqy_array(l), omega_0, bbo_length,
                                   beam_distance, k_p, k_s, k_i, z,linear_constant, qpx_sq_constant, qpy_sq_constant, abcd_constants[0], qs_constant, qi_constant,
                                                                                 omega_s, omega_i, no_s, theta_p, linear_constant_k2, qpx_sq_constant_k2, qpy_sq_constant_k2,
                                                                                 qi_constant_k2, qsx_sq_constant_k2, qsy_sq_constant_k2, abcd_constants_s[0], type_select);
                    
                    /*std::complex<double> fft_integrand = calc_fft_integrand(qx_array(i), qy_array(j), dqx_array(k), dqy_array(l), omega_0, bbo_length,
                                 beam_distance, k_p, k_s, k_i, z, abcd_constants, abcd_constants_s, omega_s, omega_i, no_s, theta_p, type_select);
                    */
                    in_tensor(i,j,k,l) = fft_integrand*std::pow(-1, i+j+k+l); //double_t((-1)^(i+j+k+l)); //the (-1)^(i+j+k+l) is for shifting the 0th frequency to the middle of the output_tensor
                }
            }
        }
    }
    //reset background color
    printf("\033[0m\n");
    std::cout << "qx(0) : " << qx_array(0) << "\n";
    //);
    std::cout << "Parameters: theta_p: " << theta_p << ", omega_i: " << omega_i << ", no_s: " << no_s << ", k_p: " << k_p << ", z_pos: " << z << ", beam_distance(d): " << beam_distance << "\n";
    //std::cout << "fft_input_data at 0000: " << in_tensor(0,0,0,0);
    //std::cout << "In_Tensor : " << in_tensor << "\t";
    printf("Before fftw_execute\n");
    auto stop = std::chrono::high_resolution_clock::now();
    
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> abs_tensor((float*) in_tensor.data(), num_samples_wide_x, num_samples_wide_y, num_samples_narrow_x, num_samples_narrow_y);
    //Eigen::Tensor<float, 4, Eigen::RowMajor> abs_tensor(num_samples_wide_x,num_samples_wide_y,num_samples_narrow_x,num_samples_narrow_y);
    //abs_tensor = in_tensor.abs();
    //std::cout << "Min & Max in_tensor absolute (temp) Data: " << abs_tensor.minimum() << "\t" << abs_tensor.maximum() << "\n\n";
    std::cout << "Elapsed Time for calculating Tensor_data: " << std::chrono::duration_cast<std::chrono::seconds>(stop-start).count() << std::endl;
    fftwf_execute(p); /* repeat as needed */
    printf("Finished computing fftw \n");
    //std::cout << "fft_output_data at 0000: " << in_tensor(0,0,0,0);
    /*for (int i = 0; i<N; i++){
        printf("Real: %f, Imag: %f\n", in_tensor(i,0,0,0).real(), in_tensor(i,0,0,0).imag());
    }
        */
    stop = std::chrono::high_resolution_clock::now();
    


    std::cout << "Elapsed Time after fftw: " << std::chrono::duration_cast<std::chrono::seconds>(stop -start).count() << std::endl;
    fftwf_destroy_plan(p);
    abs_tensor = in_tensor.abs();
    std::cout << "Value after abs: Max: " << abs_tensor.maximum() << " Min: " << abs_tensor.minimum() << "\n\n";
    abs_tensor = abs_tensor.square();
    std::cout << "Value after square: " << abs_tensor.maximum() << "\n\n";
    
    //VERY Crude Integration method. Is fast though. Takes sum along one axis of tensor and multiplies with 2.0*dqx and then takes sum of other axis and multiplies by 2.0*dqy. 
    //-> simplified to sum along dqy and dqx and then multiplie by (2.0)*dqx * (2.0)*dqy
    //Eigen::Tensor<double_t, 3, Eigen::RowMajor> sum_tensor(num_samples_wide_x,num_samples_wide_y,num_samples_narrow_x);
    Eigen::Tensor<float, 2, Eigen::RowMajor> sum_tensor(num_samples_wide_x, num_samples_wide_y);
    Eigen::array<int,2> reduce_dims({3,2});
    //Eigen::array<int,1> reduce_dims2({2});
    sum_tensor = abs_tensor.sum(reduce_dims);
    float del = 4.0 * dqy * dqx;
    sum_tensor = sum_tensor * del;//sum_tensor.constant(4.0*dqy*dqx);
    //sum_tensor_y = sum_tensor.sum(reduce_dims2);
    //sum_tensor_y = sum_tensor_y * 2.0*dqx;
    Eigen::Tensor<float, 0, Eigen::RowMajor> max_tensor = sum_tensor.maximum();
    double_t max_value = max_tensor(0);
    Eigen::Tensor<float, 0, Eigen::RowMajor> min_tensor = sum_tensor.minimum();
    double_t min_value = min_tensor(0);
    Eigen::Tensor<float, 0, Eigen::RowMajor> max_tensor_abs = abs_tensor.maximum();
    double_t max_value_abs = max_tensor_abs(0);
    Eigen::Tensor<float, 0 ,Eigen::RowMajor> min_tensor_abs = abs_tensor.minimum();
    double_t min_value_abs = min_tensor_abs(0);
    std::cout << "Maximum Value in sum tensor: " << max_value << "\n\n";
    std::cout << "Minimum Value in sum tensor: " << min_value << "\n\n";
    
    
        //std::cout << "Computed SPDC Tensor:\n\n" << sum_tensor << "\n\n";
    std::cout << "Tensor size is: " << sum_tensor.size() << "\n\n";
    
   // std::vector<std::vector<double_t>> heatmap_data(sum_tensor.dimension(0), std::vector<double_t>(sum_tensor.dimension(1)));
    std::cout << "sum_tensor dim 0 : " << sum_tensor.dimension(0) << "\n";
    std::cout << "sum_tensor dim 1 : " << sum_tensor.dimension(1) << "\n";
    //std::cout << "Sum_tensor values : " << sum_tensor_y;
    std::cout << "Sum_tensor Min: " << min_value << "\n";
    std::cout << "Sum_tensor Max: " << max_value << "\n";
    //std::cout << "dqy and dqx: " << dqy << dqx << "\n";
    printf("dqy and dqx %f.2, %f.2\n", dqy, dqx);
    printf("Min & Max in Abs_tensor %f, %f\n", min_value_abs, max_value_abs);
    std::ofstream output_file;
    output_file.open(file_name, std::ios_base::binary | std::ios_base::out);
    int size_write = sum_tensor.size()* sizeof(float);
    output_file.write( (char*) sum_tensor.data(),size_write); 
    output_file.close();
    std::cout << "Finished computing Data. Q_increment x: " << std::numbers::pi / qx << "\t Q_increment y: " << std::numbers::pi / qy << "\n";
    std::cout << "Call: py ./visualize.py " << file_name << " -x " << sum_tensor.dimension(0) << " -y " << sum_tensor.dimension(1) << " -q_inc_x " << std::numbers::pi/qx << " -q_inc_y " << std::numbers::pi/qy << " <output_name>\n";
    std::cout << "Make sure you are in python virtual env. before that. Or just install matplotlib and numpy. Also python executable symbol may not be called py on unix but I think py3 or python";
    return 0;

}
