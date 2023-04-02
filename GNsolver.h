#pragma once

#include <pcl/io/pcd_io.h> 
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/types.hpp"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"



void GNcalculation(std::vector<Eigen::Vector3d>& p, std::vector<Eigen::Vector3d>& q, std::vector<int> idx, std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Vector3d>& t,
	int sum, Eigen::Matrix3d HER, Eigen::Vector3d HEt, Sophus::Vector6d& delta);
