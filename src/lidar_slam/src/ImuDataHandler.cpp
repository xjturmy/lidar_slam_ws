#include "ImuDataHandler.h"

ImuDataHandler::ImuDataHandler() {
    // 初始化ROS订阅器
    imu_sub_ = nh_imu_.subscribe("/imu",1,&ImuDataHandler::imuCallback, this);
    tf_broadcaster_ = std::make_shared<tf::TransformBroadcaster>();
    // 初始化变量
    last_time_ = ros::Time::now();
    rotation_matrix_to_001_ = Eigen::Matrix3d::Identity();
}

void ImuDataHandler::processImudata(float ax, float ay, float az) {

    Eigen::Vector3d init_imu_data_accel(ax, ay, az);
    init_imu_data_accel.normalize();
    Eigen::Vector3d rotation_axis = Eigen::Vector3d(0.0, 0.0, 1.0).cross(init_imu_data_accel);
    double cos_theta = Eigen::Vector3d(0.0, 0.0, 1.0).dot(init_imu_data_accel);
    double theta = acos(cos_theta);

    if (rotation_axis.norm() < 1e-6) {
        ROS_WARN("Rotation axis is too small, skipping update.");
        return;
    }

    Eigen::AngleAxisd rotation(theta, rotation_axis.normalized());
    rotation_matrix_to_001_ = rotation.toRotationMatrix();
}

void ImuDataHandler::calculateVelocityDisplacement(const Eigen::Vector3d& acceleration, double dt) {
    velocity_ += acceleration * dt;
    displacement_ += velocity_ * dt;
}

Eigen::Matrix3d ImuDataHandler::getMatrix001(){
    return rotation_matrix_to_001_; 
}

Eigen::Vector3d ImuDataHandler::getAcclerationXYZ() {
    return Eigen::Vector3d(ax, ay, az);
}

void ImuDataHandler::imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
    try {

         ax = imu_msg->linear_acceleration.x;
         ay = imu_msg->linear_acceleration.y;
         az = imu_msg->linear_acceleration.z;
        // double gx = imu_msg->angular_velocity.x;
        // double gy = imu_msg->angular_velocity.y;
        // double gz = imu_msg->angular_velocity.z;

        if (ax == 0.0 && ay == 0.0 && az == 0.0) {
            return ;
        }

        // if (initial_imu_data_.size() < 10) {
        //     processFormer10Imu(ax, ay, az);
        //     initial_imu_data_.push_back(Eigen::Vector3d(ax, ay, az));
        //     return;
        // } else if (initial_imu_data_.size() == 10) {
        //     Eigen::Matrix3d rotation = rotation_matrix_to_001_;
        //     Eigen::Quaterniond quaternion(rotation);
        // }
        
        processImudata(ax, ay, az);
        // initial_imu_data_.push_back(Eigen::Vector3d(ax, ay, az));


        // Eigen::Vector3d corrected_accel_data(ax, ay, az);
        // Eigen::Vector3d corrected_gyro_data(gx, gy, gz);

        // Eigen::Vector3d corrected_accel = rotation_matrix_to_001_ * corrected_accel_data;
        // Eigen::Vector3d corrected_gyro = rotation_matrix_to_001_ * corrected_gyro_data;

        // ros::Time current_time = ros::Time::now();
        // double dt = (current_time - last_time_).toSec();
        // last_time_ = current_time;
        // calculateVelocityDisplacement(corrected_accel, dt);

    } catch (const std::exception& e) {
        ROS_ERROR("Error processing IMU data: %s", e.what());
        last_time_ = ros::Time::now();
    }
}