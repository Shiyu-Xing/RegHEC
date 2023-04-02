# RegHEC
## About
RegHEC is a registration-based hand-eye calibration technique using multi-view point clouds of arbitrary object. It tries to align multi-view point clouds of arbitrary object by estimating the hand-eye relation, thus both point clouds registration and hand-eye calibration are achieved simultaneously, making it favourable for robotic 3-D reconstruction task, where calibration and registration processess are normally separated. 

<!--The idea is simple. Multi-view point clouds point clouds captured at different viewpoints should align, after transformation into robot base frame by left multiplying first correct hand-eye(flange-sensor) relation then corresponding robot poses(pose of flange frame w.r.t robot base, when point clouds are captured).-->

<!--RegHEC tries to align multi-view point clouds of arbitrary object by estimating the hand-eye relation, thus both point clouds registration and hand-eye calibration are achieved simultaneously, making it favourable for robotic 3-D reconstruction task, where calibration and registration processess are normally separated. -->

The cores of RegHEC are 2 novel algorithms. First, Bayesian Optimization based initial alignment(BO-IA) models the registration problem as a Gaussian Process over hand-eye relation and covariance function is modified (given in `ExpSE3.cpp`) to be compatible with distance metric in 3-D motion space SE(3). It gives the coarse point clouds registration then hand over the the proper initial guess of hand-eye relation to an ICP variant with Anderson Accleration(AA-ICPv) for later fine registration and accurate calibration. 

<p align="center">
<img src="pic/setup.png" alt="alt text" width=68% height=68%>
</p>

<!--First, point clouds initial alignment and rough hand-eye relation are obtained via Bayesian Optimization,
where registration problem is modeled as a Gaussian Process over hand-eye relation and covariance function is modified to be compatible with distance metric in 3-D motion space SE(3). Second, an ICP variant, regarded as a fixed-point problem and significantly accelerated by Anderson Acceleration, is proposed to realize fine registration and accurate hand-eye calibration. -->


As a general solution, RegHEC is applicable for most 3-D vision guided task in both eye-in-hand and eye-to-hand scenario with no need for specialized calibration rig(e.g. calibration board) but arbitrary available object. *This technique is verified feasible and effective with real robotic hand-eye system and varieties of arbitrary objects including cylinder, cone, sphere and simple plane, which can be quite challenging for correct point cloud registration and sensor motion estimation using existing method.*    

For more information, please refer to our paper.


## Dependencies
This repository is a C++ solution developed with VS2019 and the following versions of external libraries:

PCL 1.11.1
  
Limbo 2.1 with NLOPT

Eigen 3.3.9
  
Sophus 1.0.0

## Input
Multi-view point clouds and corresponding robot poses(pose of flange frame w.r.t robot base frame) where point clouds are captured. 

Data used in the paper is given in Data folder. In our experiments, point clouds were captured from 9 different viewpoints. Change the input directory where multi-view point clouds and correponding robot poses are to try different object. You can also try with your own data.  
```C++
std::string path = "./data/David";
```
  
`RobotPoses.dat` gives the robot poses in 6 dimensions. First 3 elements in each row are Euler angles for orientation and second 3 elements are positions. 
The current version is rather static, some simple modifications are needed to test with number of viewpoints other than 9. We will make the solution more dynamic in the later commit.

## Output
Calibrated hand-eye relation and multi-view point clouds registration.
<p align="center">
<img src="pic/FineReg.png" alt="alt text" width=68% height=68%>
</p>
<p align="center">
Registration of 9 David point clouds and calibration results given by RegHEC(eye in hand)
</p>

<p align="center">
<img src="pic/Eye2HandFineReg.png" alt="alt text" width=68% height=68%>
</p>
<p align="center">
Registration of 9 Gripper point clouds and calibration results given by RegHEC(eye to hand)
</p>