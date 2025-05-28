#include "ImuDataHandler.h"

ImuDataHandler::ImuDataHandler(const std::string& imu_addr) {
    // 初始化ROS订阅器
    imu_sub_ = nh_.subscribe("/imu", 1000, &ImuDataHandler::imuCallback, this);
    tf_broadcaster_ = std::make_shared<tf::TransformBroadcaster>();
    // 初始化变量
    last_time_ = ros::Time::now();
}

void ImuDataHandler::publishBaseToMapTransform(const Eigen::MatrixXd& base_to_map) {
    // 假设 base_link 到 map 的变换矩阵是 self.base_to_map
    tf::Vector3 translation;
    translation.setValue(base_to_map(0,3), base_to_map(1,3), base_to_map(2,3));
    Eigen::Quaterniond rotation = base_to_map.block<3,3>(0,0);
    tf::Quaternion q(rotation.x(), rotation.y(), rotation.z(), rotation.w());

    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "base_link";
    transformStamped.transform.translation.x = translation.x();
    transformStamped.transform.translation.y = translation.y();
    transformStamped.transform.translation.z = translation.z();
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(transformStamped);
}

void ImuDataHandler::processFormer10Imu(double ax, double ay, double az) {
    ROS_INFO("正在处理前十帧数据");
    Eigen::Vector3d init_imu_data_accel(ax, ay, az);
    // 初始化为单位矩阵
    Eigen::Matrix3d rotation_matrix_to_001 = Eigen::Matrix3d::Identity();
    init_imu_data_accel = rotation_matrix_to_001 * init_imu_data_accel;
    init_imu_data_accel.normalize();
    Eigen::Vector3d rotation_axis = Eigen::Vector3d(0.0, 0.0, 1.0).cross(init_imu_data_accel);
    double cos_theta = Eigen::Vector3d(0.0, 0.0, 1.0).dot(init_imu_data_accel);
    double theta = acos(cos_theta);
    Eigen::AngleAxisd rotation(theta, rotation_axis);
    rotation_matrix_to_001 = rotation_matrix_to_001 * rotation.toRotationMatrix();
}

void ImuDataHandler::calculateVelocityDisplacement(const Eigen::Vector3d& acceleration, double dt) {
    velocity_ += acceleration * dt;
    displacement_ += velocity_ * dt;
}

void ImuDataHandler::imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
    try {
        double ax = imu_msg->linear_acceleration.x;
        double ay = imu_msg->linear_acceleration.y;
        double az = imu_msg->linear_acceleration.z;
        double gx = imu_msg->angular_velocity.x;
        double gy = imu_msg->angular_velocity.y;
        double gz = imu_msg->angular_velocity.z;

        if (ax == 0.0 && ay == 0.0 && az == 0.0) {
            return;
        }

        if (initial_imu_data_.size() < 10) {
            processFormer10Imu(ax, ay, az);
            initial_imu_data_.push_back(Eigen::Vector3d(ax, ay, az));
            return;
        } else if (initial_imu_data_.size() == 10) {
            Eigen::Matrix3d rotation = rotation_matrix_to_001_;
            Eigen::Quaterniond quaternion(rotation);
        }

        Eigen::Vector3d corrected_accel_data(ax, ay, az);
        Eigen::Vector3d corrected_gyro_data(gx, gy, gz);

        Eigen::Vector3d corrected_accel = rotation_matrix_to_001_ * corrected_accel_data;
        Eigen::Vector3d corrected_gyro = rotation_matrix_to_001_ * corrected_gyro_data;

        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_time_).toSec();
        last_time_ = current_time;
        calculateVelocityDisplacement(corrected_accel, dt);

    } catch (const std::exception& e) {
        ROS_ERROR("Error processing IMU data: %s", e.what());
        last_time_ = ros::Time::now();
    }
}