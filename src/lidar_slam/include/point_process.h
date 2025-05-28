#ifndef POINTPROCESS_H
#define POINTPROCESS_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>
#include <sstream>

// PCL 点云处理相关头文件
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>                  // 用于下采样
#include <pcl/filters/statistical_outlier_removal.h> // 用于移除离群点
#include <pcl/filters/passthrough.h>                 // 用于直通滤波
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> // 用于计算点云的中心
// Eigen 相关头文件
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// ROS 和 tf2 相关头文件
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

// G2O 相关头文件
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <thread> // 包含线程库

#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>

// imu 相关头文件
#include "ImuDataHandler.h"

class PointCloudProcessor
{
public:
    PointCloudProcessor();
    void start();

private:
    void process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
    void callback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
    void publish_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points,
                            const std::string &frame_id,
                            ros::Publisher &pub);
    float calculateCorrespondenceDistances(const pcl::PointCloud<pcl::PointXYZ>::Ptr &Final,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt);

    Eigen::Matrix4f icp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr &Final);
    Eigen::Matrix4f ndt_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr &Final);
    Eigen::Matrix4f gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr &Final);
    Eigen::Matrix4f ndt_registration_test(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                          pcl::PointCloud<pcl::PointXYZ>::Ptr &Final,
                                          float &resolution,
                                          float &step_size,
                                          int &max_iterations);
    void compareRegistrationAlgorithms();
    void publishTransform(const Eigen::Matrix4f &transformation_total);
    void filterPointCloudByField(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud);
    void projectPointCloudToXYPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void recordStatisticsToCSV(const std::string &filename, const std::string &algorithm,
                           float distance, float mean, float stddev);
    void optimizeTrajectory(std::vector<Eigen::Matrix4f>& transformations) ;
    void publishMarker(const Eigen::Matrix4f &transformation_total);

    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pc_pub;
    ros::Publisher pc_pub_target;
    ros::Publisher pc_icp_pub;
    ros::Publisher pc_ndt_pub;
    ros::Publisher pc_gicp_pub;
    ros::Publisher marker_pub_;


    pcl::PointCloud<pcl::PointXYZ>::Ptr map_points;
    // 存储所有帧的点云数据
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_buffer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame_points;
    Eigen::Matrix4f base_to_map;
    int frame_count;
    std::ofstream csv_file;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    Eigen::Matrix4f transformation_total_ = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> transformations; // 存储变换矩阵
    std::vector<geometry_msgs::Point> trajectory_points_; // 用于存储轨迹点

    std::vector<float> icp_distances; // 存储每次计算的ICP对应点距离
    float icp_distance_sum = 0.0f;    // ICP对应点距离的总和
    float icp_distance_mean = 0.0f;   // ICP对应点距离的平均值
    float icp_distance_stddev = 0.0f; // ICP对应点距离的标准差   
    std::vector<float> gicp_distances; // 存储每次计算的GICP对应点距离
    float gicp_distance_sum = 0.0f;    // GICP对应点距离的总和
    float gicp_distance_mean = 0.0f;   // GICP对应点距离的平均值
    float gicp_distance_stddev = 0.0f; // GICP对应点距离的标准差   
    std::vector<float> ndt_distances; // 存储每次计算的NDT对应点距离
    float ndt_distance_sum = 0.0f;    // NDT对应点距离的总和
    float ndt_distance_mean = 0.0f;   // NDT对应点距离的平均值
    float ndt_distance_stddev = 0.0f; // NDT对应点距离的标准差
    std::vector<float> current_to_last_distances; // 存储当前帧到上一帧的距离
    float current_to_last_distance_sum = 0.0f; // 当前帧到上一帧的距离的总和
    float current_to_last_distance_mean = 0.0f; // 当前帧到上一帧的距离的平均值
    float current_to_last_distance_stddev = 0.0f; // 当前帧到上一帧的距离的标准差

    Eigen::Vector3f center_point_map; // 地图中点云箱子的中心点
    Eigen::Vector3f center_point_current; // 当前帧中点云箱子的中心点
    Eigen::Matrix4f reality_transformation = Eigen::Matrix4f::Identity();
    double reality_timestamp = 0.0; // 真实时间戳
    std::vector<double> timestamps_buffer; // 存储所有帧的时间戳
};

#endif // POINTPROCESS_H