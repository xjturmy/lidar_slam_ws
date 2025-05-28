#ifndef IMUDATAHANDLER_H
#define IMUDATAHANDLER_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Matrix>
#include <vector>

class ImuDataHandler {
public:
    ImuDataHandler(const std::string& imu_addr);
    void publishBaseToMapTransform(const Eigen::MatrixXd& base_to_map);
    void processFormer10Imu(double ax, double ay, double az);
    void calculateVelocityDisplacement(const Eigen::Vector3d& acceleration, double dt);
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg);

private:
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;
    std::shared_ptr<tf::TransformBroadcaster> tf_broadcaster_;
    Eigen::Matrix3d rotation_matrix_to_001_;
    Eigen::Matrix3d rotation_matrix_to_world_;
    Eigen::Vector3d velocity_;
    Eigen::Vector3d displacement_;
    ros::Time last_time_;
    std::vector<Eigen::Vector3d> initial_imu_data_;
    Eigen::Quaterniond quaternion_;
    bool flag1;
    bool flag2;
};

#endif // IMUDATAHANDLER_H