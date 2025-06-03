#include "ImuDataHandler.h"

ImuDataHandler::ImuDataHandler()
{
    // 初始化ROS订阅器
    imu_sub_ = nh_imu_.subscribe("/imu", 1, &ImuDataHandler::imuCallback, this);
    rotation_matrix_to_001_ = Eigen::Matrix3d::Identity();
}

ImuDataHandler::~ImuDataHandler()
{
    // 执行清理操作
    ROS_INFO("ImuDataHandler is being destroyed.");
}
void ImuDataHandler::imuCallback(const sensor_msgs::Imu::ConstPtr &imu_msg)
{
    try
    {

        float ax = imu_msg->linear_acceleration.x;
        float ay = imu_msg->linear_acceleration.y;
        float az = imu_msg->linear_acceleration.z;
        if (ax == 0.0 && ay == 0.0 && az == 0.0)
        {
            return;
        }
        processImudata(ax, ay, az);
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Error processing IMU data: %s", e.what());
    }
}

void ImuDataHandler::processImudata(float ax, float ay, float az)
{

    Eigen::Vector3d init_imu_data_accel(ax, ay, az);
    init_imu_data_accel.normalize();
    Eigen::Vector3d rotation_axis = Eigen::Vector3d(0.0, 0.0, 1.0).cross(init_imu_data_accel);
    double cos_theta = Eigen::Vector3d(0.0, 0.0, 1.0).dot(init_imu_data_accel);
    double theta = acos(cos_theta);

    if (rotation_axis.norm() < 1e-6)
    {
        ROS_WARN("Rotation axis is too small, skipping update.");
        return;
    }

    Eigen::AngleAxisd rotation(theta, rotation_axis.normalized());
    rotation_matrix_to_001_ = rotation.toRotationMatrix();
}

Eigen::Matrix4f ImuDataHandler::getMatrix001()
{

    Eigen::Matrix4f transform_matrix;
    transform_matrix.setIdentity();                                             // 初始化为单位矩阵
    transform_matrix.block<3, 3>(0, 0) = rotation_matrix_to_001_.cast<float>(); // 将 3x3 旋转矩阵转换为 float 类型并赋值
    return transform_matrix;
}
