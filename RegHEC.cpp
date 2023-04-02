#include <iostream>

#pragma   push_macro("min")  
#pragma   push_macro("max")  
#undef   min  
#undef   max  

#include <pcl/common/transforms.h>  
#include <pcl/io/pcd_io.h> 
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl\visualization\pcl_visualizer.h>
#include <pcl/filters/random_sample.h>


#define USE_NLOPT

#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/acqui/ei.hpp>
#include <limbo/serialize/text_archive.hpp>

#include <limbo/kernel/expSE3.hpp>
#include <limbo/opt/nlopt_grad.hpp>


#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/types.hpp"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

#include<algorithm>

#include "GNsolver.h"



// Two-dimensional array to store the robot poses read from controller
double RoboPose[9][6];

//Pointer to the downsampled multi-view point clouds
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud1sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud2sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud3sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud4sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud5sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud6sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud7sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud8sub(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud9sub(new pcl::PointCloud<pcl::PointXYZ>());

//Pointer to the downsampled multi-view point clouds after transformation
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud1subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud2subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud3subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud4subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud5subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud6subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud7subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud8subT(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud9subT(new pcl::PointCloud<pcl::PointXYZ>());

//Robot poses in SE(3)
Sophus::SE3d T_A1, T_A2, T_A3, T_A4, T_A5, T_A6, T_A7, T_A8, T_A9;


//search space of Bayesian optimization, eye in hand
double sideLength = 0.2;
Eigen::Vector3d offset(0, 0, 0);

//search space of Bayesian optimization, eye to hand
/*double sideLength = 0.4;
Eigen::Vector3d offset(-0.6, -0.2, 0.2)*/;

//trim ratio and convergence threshold
double trimRatio = 0.9;
double convThresh = 0.0001;

//eye in hand or eye to hand. True for eye in hand, false for eye to hand
bool eyeinhand = true;


using namespace limbo;
using namespace std;

//Parameters setting for Bayesian Optimization

struct Params {
    //kernel hyper parameters update frequency in itertions
	struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
	  BO_PARAM(int, hp_period, 10)         
	};

	// depending on which internal optimizer we use, we need to import different parameters
#ifdef USE_NLOPT
	struct opt_nloptnograd : public defaults::opt_nloptnograd {
	};
	struct opt_nloptgrad : public defaults::opt_nloptgrad {
	};

#elif defined(USE_LIBCMAES)
	struct opt_cmaes : public defaults::opt_cmaes {
	};
#else
	struct opt_gridsearch : public defaults::opt_gridsearch {
		BO_PARAM(int, bins, 2)
	};
#endif


	// enable / disable the writing of the result files
	struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
		BO_PARAM(int, stats_enabled, false);
	};


	struct kernel : public defaults::kernel {
		BO_PARAM(double, noise, 0);
	};


	// we use 50 random samples to initialize the algorithm
	struct init_randomsampling {
		BO_PARAM(int, samples, 50);
	};

	// we stop after 50 iterations
	struct stop_maxiterations {
		BO_PARAM(int, iterations, 50);
	};


	struct acqui_ei : public defaults::acqui_ei {
	};


	struct opt_rprop : public defaults::opt_rprop {
	};


	struct kernel_expSE3 : public defaults::kernel_expSE3 {
	};

};


//Optimization objective of Bayesian Optimization, which is the mean of squared distance between corresponding points, i.e. MSE or E(u) in the paper.
struct Eval {

	// number of input dimension (x.size())
	BO_PARAM(size_t, dim_in, 6);
	// number of dimensions of the result (res.size())
	BO_PARAM(size_t, dim_out, 1);


	// the function to be optimized
	Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
	{

		//Set search space. All dimensions of x are in [0,1] by default, so rescale the hyper rectangle.
		Sophus::Vector3d Rot = (x.head(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * 2 * 3.1415926;  //rotation
		Eigen::Vector3d t = (x.tail(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * sideLength + offset; // translation

		//Convert the sample point into SE(3).
		Eigen::Matrix4d HE = Eigen::Matrix4d::Identity();
		HE.block(0, 0, 3, 3) = Sophus::SO3d::exp(Rot).matrix();
		HE.block(0, 3, 3, 1) = t;


		//Transform point clouds with HE and corresponding robot poses.
		pcl::transformPointCloud(*PTRcloud1sub, *PTRcloud1subT, (T_A1.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud2sub, *PTRcloud2subT, (T_A2.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud3sub, *PTRcloud3subT, (T_A3.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud4sub, *PTRcloud4subT, (T_A4.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud5sub, *PTRcloud5subT, (T_A5.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud6sub, *PTRcloud6subT, (T_A6.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud7sub, *PTRcloud7subT, (T_A7.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud8sub, *PTRcloud8subT, (T_A8.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud9sub, *PTRcloud9subT, (T_A9.matrix() * HE).cast<float>());


		std::vector<float> pointDistance(1);
		std::vector<int> pointIdx(1);
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		std::vector<float> distances;


		//Pair point clouds captured before and after each robot motion. In each pair, for each point in the samller
		//point cloud, we find its corresponding closet point in the other point cloud.
		if (PTRcloud1sub->size() < PTRcloud2sub->size()) {

			kdtree.setInputCloud(PTRcloud2subT);

			for (size_t i = 0; i < PTRcloud1subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud1subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud1subT);

			for (size_t i = 0; i < PTRcloud2subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud2subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}



		if (PTRcloud2sub->size() < PTRcloud3sub->size()) {

			kdtree.setInputCloud(PTRcloud3subT);

			for (size_t i = 0; i < PTRcloud2subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud2subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud2subT);

			for (size_t i = 0; i < PTRcloud3subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud3subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}



		if (PTRcloud3sub->size() < PTRcloud4sub->size()) {

			kdtree.setInputCloud(PTRcloud4subT);

			for (size_t i = 0; i < PTRcloud3subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud3subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud3subT);

			for (size_t i = 0; i < PTRcloud4subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud4subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}


		}



		if (PTRcloud4sub->size() < PTRcloud5sub->size()) {

			kdtree.setInputCloud(PTRcloud5subT);

			for (size_t i = 0; i < PTRcloud4subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud4subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud4subT);

			for (size_t i = 0; i < PTRcloud5subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud5subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}



		if (PTRcloud5sub->size() < PTRcloud6sub->size()) {

			kdtree.setInputCloud(PTRcloud6subT);

			for (size_t i = 0; i < PTRcloud5subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud5subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud5subT);

			for (size_t i = 0; i < PTRcloud6subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud6subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}



		if (PTRcloud6sub->size() < PTRcloud7sub->size()) {

			kdtree.setInputCloud(PTRcloud7subT);

			for (size_t i = 0; i < PTRcloud6subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud6subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud6subT);

			for (size_t i = 0; i < PTRcloud7subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud7subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}



		if (PTRcloud7sub->size() < PTRcloud8sub->size()) {

			kdtree.setInputCloud(PTRcloud8subT);

			for (size_t i = 0; i < PTRcloud7subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud7subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud7subT);

			for (size_t i = 0; i < PTRcloud8subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud8subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}



		if (PTRcloud8sub->size() < PTRcloud9sub->size()) {

			kdtree.setInputCloud(PTRcloud9subT);

			for (size_t i = 0; i < PTRcloud8subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud8subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud8subT);

			for (size_t i = 0; i < PTRcloud9subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud9subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);


			}
		}

		std::sort(distances.begin(), distances.end());

		int numCorres = ceil(distances.size() * trimRatio);

		// Calculate MSE. Limbo always maximizes, so we take its opposite.
		double y = -1 * accumulate(distances.begin(), distances.begin() + numCorres, 0.0) / numCorres;



		// we return a 1-dimensional vector
		return tools::make_vector(y);
	}
};




int main() {

	//Pointer to multi-view point clouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud1(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud2(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud3(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud4(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud5(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud6(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud7(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud8(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud9(new pcl::PointCloud<pcl::PointXYZ>());

	//Pointer to the multi-view point clouds after transformation
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud1T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud2T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud3T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud4T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud5T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud6T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud7T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud8T(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr PTRcloud9T(new pcl::PointCloud<pcl::PointXYZ>());


	//Read point clouds
	std::string path = "./data/David";
	pcl::io::loadPCDFile(path + "/view1d.pcd", *PTRcloud1);
	pcl::io::loadPCDFile(path + "/view2d.pcd", *PTRcloud2);
	pcl::io::loadPCDFile(path + "/view3d.pcd", *PTRcloud3);
	pcl::io::loadPCDFile(path + "/view4d.pcd", *PTRcloud4);
	pcl::io::loadPCDFile(path + "/view5d.pcd", *PTRcloud5);
	pcl::io::loadPCDFile(path + "/view6d.pcd", *PTRcloud6);
	pcl::io::loadPCDFile(path + "/view7d.pcd", *PTRcloud7);
	pcl::io::loadPCDFile(path + "/view8d.pcd", *PTRcloud8);
	pcl::io::loadPCDFile(path + "/view9d.pcd", *PTRcloud9);



	//Down sample for Bayesian Optimization and trimming distance calculation
	float sampleRatio = 0.1;
	pcl::RandomSample<pcl::PointXYZ> rs;
	rs.setInputCloud(PTRcloud1);
	rs.setSample(ceil(PTRcloud1->size() * sampleRatio));
	rs.filter(*PTRcloud1sub);
	rs.setInputCloud(PTRcloud2);
	rs.setSample(ceil(PTRcloud2->size() * sampleRatio));
	rs.filter(*PTRcloud2sub);
	rs.setInputCloud(PTRcloud3);
	rs.setSample(ceil(PTRcloud3->size() * sampleRatio));
	rs.filter(*PTRcloud3sub);
	rs.setInputCloud(PTRcloud4);
	rs.setSample(ceil(PTRcloud4->size() * sampleRatio));
	rs.filter(*PTRcloud4sub);
	rs.setInputCloud(PTRcloud5);
	rs.setSample(ceil(PTRcloud5->size() * sampleRatio));
	rs.filter(*PTRcloud5sub);
	rs.setInputCloud(PTRcloud6);
	rs.setSample(ceil(PTRcloud6->size() * sampleRatio));
	rs.filter(*PTRcloud6sub);
	rs.setInputCloud(PTRcloud7);
	rs.setSample(ceil(PTRcloud7->size() * sampleRatio));
	rs.filter(*PTRcloud7sub);
	rs.setInputCloud(PTRcloud8);
	rs.setSample(ceil(PTRcloud8->size() * sampleRatio));
	rs.filter(*PTRcloud8sub);
	rs.setInputCloud(PTRcloud9);
	rs.setSample(ceil(PTRcloud9->size() * sampleRatio));
	rs.filter(*PTRcloud9sub);



	//Read robot poses. If eye-to-hand scenario then take the inverse
	fstream in;
	in.open(path + "/RobotPoses.dat", ios::in);//打开一个file
	if (!in.is_open()) {
		cout << "Can not find " << "RobotPoses.dat" << endl;
		system("pause");
	}
	std::string buff;
	int i = 0;//行数i
	while (getline(in, buff)) {
		std::vector<double> nums;
		// string->char *
		char* s_input = (char*)buff.c_str();
		const char* split = ",";
		// 以‘，’为分隔符拆分字符串
		char* p = strtok(s_input, split);
		double a;
		while (p != NULL) {
			// char * -> int
			a = atof(p);
			//cout << a << endl;
			nums.push_back(a);
			p = strtok(NULL, split);
		}//end while
		for (int b = 0; b < nums.size(); b++) {
			RoboPose[i][b] = nums[b];
		}//end for
		i++;
	}//end while
	in.close();

	std::vector<Eigen::Matrix3d> R;
	std::vector<Eigen::Vector3d> t;


	Eigen::Matrix3d R_A1;
	R_A1 = Eigen::AngleAxisd(RoboPose[0][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[0][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[0][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A1(RoboPose[0][3] / 1000, RoboPose[0][4] / 1000, RoboPose[0][5] / 1000);

	Sophus::SE3d T_1(R_A1, t_A1);
	if (eyeinhand) {
		T_A1 = T_1;
		R.push_back(R_A1);
		t.push_back(t_A1);
	}
	else {
		T_A1 = T_1.inverse();
		R.push_back(R_A1.inverse());
		t.push_back(-R_A1.inverse() * t_A1);
	}


	Eigen::Matrix3d R_A2;
	R_A2 = Eigen::AngleAxisd(RoboPose[1][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[1][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[1][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A2(RoboPose[1][3] / 1000, RoboPose[1][4] / 1000, RoboPose[1][5] / 1000);

	Sophus::SE3d T_2(R_A2, t_A2);
	if (eyeinhand) {
		T_A2 = T_2;
		R.push_back(R_A2);
		t.push_back(t_A2);
	}
	else {
		T_A2 = T_2.inverse();
		R.push_back(R_A2.inverse());
		t.push_back(-R_A2.inverse() * t_A2);
	}


	Eigen::Matrix3d R_A3;
	R_A3 = Eigen::AngleAxisd(RoboPose[2][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[2][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[2][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A3(RoboPose[2][3] / 1000, RoboPose[2][4] / 1000, RoboPose[2][5] / 1000);

	Sophus::SE3d T_3(R_A3, t_A3);
	if (eyeinhand) {
		T_A3 = T_3;
		R.push_back(R_A3);
		t.push_back(t_A3);
	}
	else {
		T_A3 = T_3.inverse();
		R.push_back(R_A3.inverse());
		t.push_back(-R_A3.inverse() * t_A3);
	}


	Eigen::Matrix3d R_A4;
	R_A4 = Eigen::AngleAxisd(RoboPose[3][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[3][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[3][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A4(RoboPose[3][3] / 1000, RoboPose[3][4] / 1000, RoboPose[3][5] / 1000);

	Sophus::SE3d T_4(R_A4, t_A4);
	if (eyeinhand) {
		T_A4 = T_4;
		R.push_back(R_A4);
		t.push_back(t_A4);
	}
	else {
		T_A4 = T_4.inverse();
		R.push_back(R_A4.inverse());
		t.push_back(-R_A4.inverse() * t_A4);
	}


	Eigen::Matrix3d R_A5;
	R_A5 = Eigen::AngleAxisd(RoboPose[4][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[4][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[4][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A5(RoboPose[4][3] / 1000, RoboPose[4][4] / 1000, RoboPose[4][5] / 1000);

	Sophus::SE3d T_5(R_A5, t_A5);
	if (eyeinhand) {
		T_A5 = T_5;
		R.push_back(R_A5);
		t.push_back(t_A5);
	}
	else {
		T_A5 = T_5.inverse();
		R.push_back(R_A5.inverse());
		t.push_back(-R_A5.inverse() * t_A5);
	}


	Eigen::Matrix3d R_A6;
	R_A6 = Eigen::AngleAxisd(RoboPose[5][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[5][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[5][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A6(RoboPose[5][3] / 1000, RoboPose[5][4] / 1000, RoboPose[5][5] / 1000);

	Sophus::SE3d T_6(R_A6, t_A6);
	if (eyeinhand) {
		T_A6 = T_6;
		R.push_back(R_A6);
		t.push_back(t_A6);
	}
	else {
		T_A6 = T_6.inverse();
		R.push_back(R_A6.inverse());
		t.push_back(-R_A6.inverse() * t_A6);
	}


	Eigen::Matrix3d R_A7;
	R_A7 = Eigen::AngleAxisd(RoboPose[6][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[6][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[6][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A7(RoboPose[6][3] / 1000, RoboPose[6][4] / 1000, RoboPose[6][5] / 1000);

	Sophus::SE3d T_7(R_A7, t_A7);
	if (eyeinhand) {
		T_A7 = T_7;
		R.push_back(R_A7);
		t.push_back(t_A7);
	}
	else {
		T_A7 = T_7.inverse();
		R.push_back(R_A7.inverse());
		t.push_back(-R_A7.inverse() * t_A7);
	}


	Eigen::Matrix3d R_A8;
	R_A8 = Eigen::AngleAxisd(RoboPose[7][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[7][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[7][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A8(RoboPose[7][3] / 1000, RoboPose[7][4] / 1000, RoboPose[7][5] / 1000);

	Sophus::SE3d T_8(R_A8, t_A8);
	if (eyeinhand) {
		T_A8 = T_8;
		R.push_back(R_A8);
		t.push_back(t_A8);
	}
	else {
		T_A8 = T_8.inverse();
		R.push_back(R_A8.inverse());
		t.push_back(-R_A8.inverse() * t_A8);
	}


	Eigen::Matrix3d R_A9;
	R_A9 = Eigen::AngleAxisd(RoboPose[8][2], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(RoboPose[8][1], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(RoboPose[8][0], Eigen::Vector3d::UnitX());

	Eigen::Vector3d t_A9(RoboPose[8][3] / 1000, RoboPose[8][4] / 1000, RoboPose[8][5] / 1000);

	Sophus::SE3d T_9(R_A9, t_A9);
	if (eyeinhand) {
		T_A9 = T_9;
		R.push_back(R_A9);
		t.push_back(t_A9);
	}
	else {
		T_A9 = T_9.inverse();
		R.push_back(R_A9.inverse());
		t.push_back(-R_A9.inverse() * t_A9);
	}

	cout << "Data is read into memory" << endl;



	cout << "Start BO-IA" << endl;
	//Use modified kernel ExpSE3
	using Kernel2_t = kernel::ExpSE3<Params>;

	//Use the mean of already sampled data as the mean of prior distribution
	using Mean_t = mean::Data<Params>;
	using GP_t = model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params, opt::Rprop<Params>>>;

	//Use Expected Improvement as acquisition function
	using Acqui_t = acqui::EI<Params, GP_t>;
	using acqui_opt_t = opt::NLOptNoGrad<Params>;
	bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>, acquiopt<acqui_opt_t>> boptimizer;
	boptimizer.optimize(Eval());

	//Take the best sample as the initial guess of hand-eye relation
	Eigen::Matrix<double, 6, 1> best = boptimizer.best_sample();
	Eigen::Matrix4d HE = Eigen::Matrix4d::Identity();
	Sophus::Vector3d Rotbest = (best.head(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * 2 * 3.1415926;
	Eigen::Vector3d tbest = (best.tail(3) - Eigen::Matrix<double, 3, 1>::Constant(0.5)) * sideLength + offset;
	HE.block(0, 0, 3, 3) = Sophus::SO3d::exp(Rotbest).matrix();
	HE.block(0, 3, 3, 1) = tbest;



	//Some AA-ICPv parameter initialization
	int CorresNum = 0;
	double SquareDistSum = 0.0;
	int counter = 0;
	double Err = 0;
	double ErrPrev = 0;
	double ratio = 0;
	double distanceThresh = 0;



	Eigen::Matrix3d HER = HE.block(0, 0, 3, 3);
	Eigen::Vector3d HEt = HE.block(0, 3, 3, 1);
	Eigen::Matrix<double, 6, 1> para;



	int m = 1;

	Eigen::Matrix<double, 6, Eigen::Dynamic> u(6, 0);
	Eigen::Matrix<double, 6, Eigen::Dynamic> g(6, 0);
	Eigen::Matrix<double, 6, Eigen::Dynamic> f(6, 0);

	Sophus::Vector6d u_k;
	Sophus::Vector6d u_next;

	Sophus::Vector6d u_0;
	u_0.head(3) = Sophus::SO3d::log(HER);
	u_0.tail(3) = HEt;

	u.conservativeResize(u.rows(), u.cols() + 1);
	u.col(0) = u_0;



	//Start AA-ICPv
	cout << "Start AA-ICPv" << endl;

	while (true) {

		counter = counter + 1;

		CorresNum = 0;
		SquareDistSum = 0;

		para << 0, 0, 0, HEt(0), HEt(1), HEt(2);



		//Calculate trimming distance threshold
		std::vector<int> pointIdx(1);
		std::vector<float> pointDistance(1);
		std::vector<float> distances;
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

		pcl::transformPointCloud(*PTRcloud1sub, *PTRcloud1subT, (T_A1.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud2sub, *PTRcloud2subT, (T_A2.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud3sub, *PTRcloud3subT, (T_A3.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud4sub, *PTRcloud4subT, (T_A4.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud5sub, *PTRcloud5subT, (T_A5.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud6sub, *PTRcloud6subT, (T_A6.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud7sub, *PTRcloud7subT, (T_A7.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud8sub, *PTRcloud8subT, (T_A8.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud9sub, *PTRcloud9subT, (T_A9.matrix() * HE).cast<float>());

		if (PTRcloud1sub->size() < PTRcloud2sub->size()) {

			kdtree.setInputCloud(PTRcloud2subT);


			for (size_t i = 0; i < PTRcloud1subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud1subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud1subT);


			for (size_t i = 0; i < PTRcloud2subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud2subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}



		if (PTRcloud2sub->size() < PTRcloud3sub->size()) {

			kdtree.setInputCloud(PTRcloud3subT);

			for (size_t i = 0; i < PTRcloud2subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud2subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud2subT);

			for (size_t i = 0; i < PTRcloud3subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud3subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}



		if (PTRcloud3sub->size() < PTRcloud4sub->size()) {

			kdtree.setInputCloud(PTRcloud4subT);

			for (size_t i = 0; i < PTRcloud3subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud3subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud3subT);

			for (size_t i = 0; i < PTRcloud4subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud4subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}


		}



		if (PTRcloud4sub->size() < PTRcloud5sub->size()) {

			kdtree.setInputCloud(PTRcloud5subT);

			for (size_t i = 0; i < PTRcloud4subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud4subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud4subT);

			for (size_t i = 0; i < PTRcloud5subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud5subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}



		if (PTRcloud5sub->size() < PTRcloud6sub->size()) {

			kdtree.setInputCloud(PTRcloud6subT);

			for (size_t i = 0; i < PTRcloud5subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud5subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}
		else {

			kdtree.setInputCloud(PTRcloud5subT);

			for (size_t i = 0; i < PTRcloud6subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud6subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}



		if (PTRcloud6sub->size() < PTRcloud7sub->size()) {

			kdtree.setInputCloud(PTRcloud7subT);

			for (size_t i = 0; i < PTRcloud6subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud6subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud6subT);

			for (size_t i = 0; i < PTRcloud7subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud7subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}



		if (PTRcloud7sub->size() < PTRcloud8sub->size()) {

			kdtree.setInputCloud(PTRcloud8subT);

			for (size_t i = 0; i < PTRcloud7subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud7subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud7subT);

			for (size_t i = 0; i < PTRcloud8subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud8subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}
		}



		if (PTRcloud8sub->size() < PTRcloud9sub->size()) {

			kdtree.setInputCloud(PTRcloud9subT);

			for (size_t i = 0; i < PTRcloud8subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud8subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);

			}

		}
		else {

			kdtree.setInputCloud(PTRcloud8subT);

			for (size_t i = 0; i < PTRcloud9subT->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud9subT->points[i], 1, pointIdx, pointDistance);

				distances.push_back(pointDistance[0]);


			}
		}

		std::sort(distances.begin(), distances.end());

		distanceThresh = distances[ceil(distances.size() * trimRatio)];



		////Pair point clouds captured before and after each robot motion. In each pair, for each point in the samller
		//point cloud, we find its corresponding closet point in the other point cloud. 
		//Then keep the correspondences with distance below trimming distance threshold

		pcl::transformPointCloud(*PTRcloud1, *PTRcloud1T, (T_A1.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud2, *PTRcloud2T, (T_A2.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud3, *PTRcloud3T, (T_A3.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud4, *PTRcloud4T, (T_A4.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud5, *PTRcloud5T, (T_A5.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud6, *PTRcloud6T, (T_A6.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud7, *PTRcloud7T, (T_A7.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud8, *PTRcloud8T, (T_A8.matrix() * HE).cast<float>());
		pcl::transformPointCloud(*PTRcloud9, *PTRcloud9T, (T_A9.matrix() * HE).cast<float>());


		std::vector<Eigen::Vector3d> p, q;
	    std::vector<int> idx;

		if (PTRcloud1->size() < PTRcloud2->size()) {

			kdtree.setInputCloud(PTRcloud2T);

			for (size_t i = 0; i < PTRcloud1T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud1T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {


					p.push_back(PTRcloud1->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud2->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}
		else {

			kdtree.setInputCloud(PTRcloud1T);

			for (size_t i = 0; i < PTRcloud2T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud2T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud1->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud2->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}



		if (PTRcloud2->size() < PTRcloud3->size()) {

			kdtree.setInputCloud(PTRcloud3T);

			for (size_t i = 0; i < PTRcloud2T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud2T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud2->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud3->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}
		else {

			kdtree.setInputCloud(PTRcloud2T);

			for (size_t i = 0; i < PTRcloud3T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud3T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud2->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud3->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}



		if (PTRcloud3->size() < PTRcloud4->size()) {

			kdtree.setInputCloud(PTRcloud4T);

			for (size_t i = 0; i < PTRcloud3T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud3T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud3->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud4->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);
		}
		else {

			kdtree.setInputCloud(PTRcloud3T);

			for (size_t i = 0; i < PTRcloud4T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud4T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud3->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud4->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}



		if (PTRcloud4->size() < PTRcloud5->size()) {

			kdtree.setInputCloud(PTRcloud5T);

			for (size_t i = 0; i < PTRcloud4T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud4T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud4->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud5->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}
		else {

			kdtree.setInputCloud(PTRcloud4T);

			for (size_t i = 0; i < PTRcloud5T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud5T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud4->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud5->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}



		if (PTRcloud5->size() < PTRcloud6->size()) {

			kdtree.setInputCloud(PTRcloud6T);

			for (size_t i = 0; i < PTRcloud5T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud5T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud5->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud6->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}
		else {

			kdtree.setInputCloud(PTRcloud5T);

			for (size_t i = 0; i < PTRcloud6T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud6T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud5->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud6->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);


		}



		if (PTRcloud6->size() < PTRcloud7->size()) {

			kdtree.setInputCloud(PTRcloud7T);

			for (size_t i = 0; i < PTRcloud6T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud6T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud6->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud7->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);


		}
		else {

			kdtree.setInputCloud(PTRcloud6T);

			for (size_t i = 0; i < PTRcloud7T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud7T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud6->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud7->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}



		if (PTRcloud7->size() < PTRcloud8->size()) {

			kdtree.setInputCloud(PTRcloud8T);

			for (size_t i = 0; i < PTRcloud7T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud7T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud7->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud8->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);


		}
		else {

			kdtree.setInputCloud(PTRcloud7T);

			for (size_t i = 0; i < PTRcloud8T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud8T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud7->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud8->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);
		}



		if (PTRcloud8->size() < PTRcloud9->size()) {

			kdtree.setInputCloud(PTRcloud9T);

			for (size_t i = 0; i < PTRcloud8T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud8T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud8->points[i].getVector3fMap().cast<double>());
					q.push_back(PTRcloud9->points[pointIdx[0]].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}
		else {

			kdtree.setInputCloud(PTRcloud8T);

			for (size_t i = 0; i < PTRcloud9T->points.size(); ++i)
			{
				kdtree.nearestKSearch(PTRcloud9T->points[i], 1, pointIdx, pointDistance);

				if (pointDistance[0] < distanceThresh) {

					p.push_back(PTRcloud8->points[pointIdx[0]].getVector3fMap().cast<double>());
					q.push_back(PTRcloud9->points[i].getVector3fMap().cast<double>());

					CorresNum = CorresNum + 1;
					SquareDistSum = SquareDistSum + pointDistance[0];

				}

			}

			idx.push_back(CorresNum);

		}


		//MSE of current iterate
		Err = SquareDistSum / CorresNum;


		if (counter == 1) { //Anderson Accleration initialization

			//Solve least square problem in (1)
			Sophus::Vector6d delta;
			GNcalculation(p, q, idx, R, t, CorresNum, HER, HEt, delta);
			para = para + delta;


			HER = Sophus::SO3d::exp(para.head(3)).matrix() * HER;
			HEt = para.tail(3);
			HE.block(0, 0, 3, 3) = HER;
			HE.block(0, 3, 3, 1) = HEt;


			Sophus::Vector6d u_1;
			u_1.head(3) = Sophus::SO3d::log(HER);
			u_1.tail(3) = HEt;
			u.conservativeResize(u.rows(), u.cols() + 1);
			u.col(1) = u_1;
			g.conservativeResize(g.rows(), g.cols() + 1);
			g.col(0) = u_1;
			f.conservativeResize(f.rows(), f.cols() + 1);
			f.col(0) = u_1 - u_0;
			u_next = u_1;
			u_k = u_1;

			m = m + 1;
			ErrPrev = Err;

		}
		else {

			ratio = Err / ErrPrev;

			// Failure handling
			if (ratio > 1.03) {

				//Use the last trustworthy G call as new iterate then reset history

				u.col(u.cols() - 1) = g.col(g.cols() - 1);
				HER = Sophus::SO3d::exp(u.col(u.cols() - 1).head(3)).matrix();
				HEt = u.col(u.cols() - 1).tail(3);
				HE.block(0, 0, 3, 3) = HER;
				HE.block(0, 3, 3, 1) = HEt;
				u_k = u.col(u.cols() - 1);

				m = 2;
				ErrPrev = 100;

			}

			// Anderson Acceleration
			else {

				//Solve least square problem in (1)
				Sophus::Vector6d delta;
				GNcalculation(p, q, idx, R, t, CorresNum, HER, HEt, delta);
				para = para + delta;


				HER = Sophus::SO3d::exp(para.head(3)).matrix() * HER;
				HEt = para.tail(3);


				Sophus::Vector6d g_k;
				g_k.head(3) = Sophus::SO3d::log(HER);
				g_k.tail(3) = HEt;
				g.conservativeResize(g.rows(), g.cols() + 1);
				g.col(g.cols() - 1) = g_k;
				Sophus::Vector6d f_k = g_k - u_k;

				ErrPrev = Err;

				//Check convergence
				if (f_k.norm() < convThresh) {
					break;
				}

				f.conservativeResize(f.rows(), f.cols() + 1);
				f.col(f.cols() - 1) = f_k;


				Eigen::Matrix<double, 6, Eigen::Dynamic> f_recent = f.rightCols(min(m, 5));
				Eigen::Matrix<double, 6, Eigen::Dynamic> A = f_recent.leftCols(f_recent.cols() - 1);
				A *= -1;
				A += f_recent.rightCols(1) * Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(f_recent.cols() - 1, 1).transpose();
				Eigen::Matrix<double, Eigen::Dynamic, 1> alphas = A.colPivHouseholderQr().solve(f_recent.rightCols(1));
				alphas.conservativeResize(alphas.size() + 1);
				alphas[alphas.size() - 1] = 0;
				alphas[alphas.size() - 1] = 1 - alphas.sum();


				u_next = g.rightCols(min(m, 5)) * alphas;
				u.conservativeResize(u.rows(), u.cols() + 1);
				u.col(u.cols() - 1) = u_next;

				u_k = u_next;


				HER = Sophus::SO3d::exp(u_next.head(3)).matrix();
				HEt = u_next.tail(3);
				HE.block(0, 0, 3, 3) = HER;
				HE.block(0, 3, 3, 1) = HEt;

				m = m + 1;

			}

		}

	}


	std::cout << "*********" << endl;
	std::cout << "Hand-eye relation is" << endl;
	std::cout << HE << endl;
	std::cout << "*********" << endl;


	// See Final registration
	pcl::transformPointCloud(*PTRcloud1, *PTRcloud1T, (T_A1.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud2, *PTRcloud2T, (T_A2.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud3, *PTRcloud3T, (T_A3.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud4, *PTRcloud4T, (T_A4.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud5, *PTRcloud5T, (T_A5.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud6, *PTRcloud6T, (T_A6.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud7, *PTRcloud7T, (T_A7.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud8, *PTRcloud8T, (T_A8.matrix() * HE).cast<float>());
	pcl::transformPointCloud(*PTRcloud9, *PTRcloud9T, (T_A9.matrix() * HE).cast<float>());


	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Visualizer_Viewer"));
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud1T, "cloud1");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud2T, "cloud2");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud3T, "cloud3");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud4T, "cloud4");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud5T, "cloud5");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud6T, "cloud6");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud7T, "cloud7");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud8T, "cloud8");
	viewer->addPointCloud<pcl::PointXYZ>(PTRcloud9T, "cloud9");
	viewer->addText("Press r to centre and zoom the viewer", 20, 380, 18, 100, 100, 100, "text");

	if (eyeinhand) {
		viewer->setCameraPosition(0.05, -0.3, 0.28, 0.2, -1, -0.1, 0, 0, 1);
	}
	else {
		viewer->setCameraPosition(0.25, -0.25, 0.4, -0.5, 0.5, -0.6, 0, 0, 1);
	}

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}


}

