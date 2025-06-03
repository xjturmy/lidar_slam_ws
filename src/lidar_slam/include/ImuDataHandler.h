#ifndef IMUDATAHANDLER_H
#define IMUDATAHANDLER_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include <Eigen/Matrix>
#include <vector>

class ImuDataHandler {
public:
    ImuDataHandler();
    ~ImuDataHandler();
    Eigen::Matrix4f getMatrix001();

private:
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg);
    void processImudata(float ax, float ay, float az);

    ros::NodeHandle nh_imu_;
    ros::Subscriber imu_sub_;
    Eigen::Matrix3d rotation_matrix_to_001_;
};

#endif // IMUDATAHANDLER_H