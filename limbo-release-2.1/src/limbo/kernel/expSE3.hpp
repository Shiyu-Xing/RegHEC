//This is an efficient kernel for BO over 3-D motion group SE(3) parameterized 
//by a 6-dimensional vector u ( lie algebra so(3) for rotation as its head and translation
//vector as its tail). For more information please refer to the RegHEC paper, BO-IA section.

#ifndef LIMBO_KERNEL_EXPSE3_HPP
#define LIMBO_KERNEL_EXPSE3_HPP

#include <limbo/kernel/kernel.hpp>
#include "sophus/types.hpp"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

namespace limbo {
    namespace defaults {
        struct kernel_expSE3 {
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 0.00001);
            BO_PARAM(double, l, 1);
            BO_PARAM(double, a, 1);
       
        };
    } // namespace defaults
    namespace kernel {
       
        template <typename Params>
        struct ExpSE3 : public BaseKernel<Params, ExpSE3<Params>> {
            ExpSE3(size_t dim = 1) : _sf2(Params::kernel_expSE3::sigma_sq()), _l(Params::kernel_expSE3::l()),_a(Params::kernel_expSE3::a())    //constructor using the parameters in the Params
            {
                _h_params = Eigen::VectorXd(3);
                _h_params << std::log(_l), std::log(std::sqrt(_sf2)), std::log(_a);
            }

            size_t params_size() const { return 3; }

            // Return the hyper parameters in log-space
            Eigen::VectorXd params() const { return _h_params; }

            // We expect the input parameters to be in log-space
            void set_params(const Eigen::VectorXd& p)
            {
                _h_params = p;
                _l = std::exp(p(0));
                _sf2 = std::exp(2.0 * p(1));
                _a = std::exp(p(2));

            }

            double kernel(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {

                Eigen::Vector3d so3_1 = (v1.head(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * 2 * 3.1415926;
                Eigen::Vector3d so3_2 = (v2.head(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * 2 * 3.1415926;


                Eigen::Matrix3d SO3_1 = Sophus::SO3d::exp(so3_1).matrix();
                Eigen::Matrix3d SO3_2 = Sophus::SO3d::exp(so3_2).matrix();
                Eigen::Matrix3d deltaR = SO3_1.inverse() * SO3_2;
                Eigen::Vector3d so3 = Sophus::SO3d::log(deltaR);

                double distance = so3.norm() + (v1.tail(3) - v2.tail(3)).norm()* _a * _a;

                 
                double l_sq = _l * _l;
                double r = distance / l_sq;
                return _sf2 * std::exp(-0.5 * r);
            }

            Eigen::VectorXd gradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {

                Eigen::Vector3d so3_1 = (x1.head(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * 2 * 3.1415926;
                Eigen::Vector3d so3_2 = (x2.head(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * 2 * 3.1415926;


                Eigen::Matrix3d SO3_1 = Sophus::SO3d::exp(so3_1).matrix();
                Eigen::Matrix3d SO3_2 = Sophus::SO3d::exp(so3_2).matrix();
                Eigen::Matrix3d deltaR = SO3_1.inverse() * SO3_2;
                Eigen::Vector3d so3 = Sophus::SO3d::log(deltaR);

                double distance = so3.norm() + (x1.tail(3) - x2.tail(3)).norm() *  _a * _a;



                Eigen::VectorXd grad(this->params_size());
                double l_sq = _l * _l;
                double r = distance / l_sq;
                double k = _sf2 * std::exp(-0.5 * r);

                double a_sq = _a * _a;
                double v = a_sq / l_sq * (x1.tail(3) - x2.tail(3)).norm();


                grad(0) = r * k;
                grad(1) = 2 * k;
                grad(2) = - k * v;


                return grad;
            }

        protected:
            double _sf2, _l, _a;

            Eigen::VectorXd _h_params;
        };
    } // namespace kernel
} // namespace limbo

#endif
