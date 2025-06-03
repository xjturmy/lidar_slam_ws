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
    // ROS 节点句柄和订阅/发布对象
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pc_pub;
    ros::Publisher pc_icp_pub;
    ros::Publisher pc_ndt_pub;
    ros::Publisher pc_gicp_pub;
    ros::Publisher marker_pub_;

    // IMU 数据处理器
    std::unique_ptr<ImuDataHandler> imu_handler_;

    // 点云数据和变换矩阵
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_points;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_buffer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame_points;

    // 文件操作和广播对象
    std::ofstream csv_file;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // 变换矩阵和轨迹点
    Eigen::Matrix4f transformation_total_ = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> transformations;
    std::vector<geometry_msgs::Point> trajectory_points_;

    // 中心点和变换
    Eigen::Vector3f center_point_map;
    Eigen::Vector3f center_point_current;
    Eigen::Matrix4f reality_transformation = Eigen::Matrix4f::Identity();

    // 时间戳相关
    double reality_timestamp = 0.0;
    std::vector<double> timestamps_buffer;

    // 标志位
    bool first_frame_flag_ = true;

    // 点云处理相关函数
    void process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
    void callback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
    void publish_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points,
                            const std::string &frame_id,
                            ros::Publisher &pub);
    float calculateCorrespondenceDistances(const pcl::PointCloud<pcl::PointXYZ>::Ptr &Final,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt);
    void handleFirstFrame(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_001, double timestamp);
    void handleBuffers();
    void optimizeAndPublishAll();
    void updateAndPublishRecent();
    void updateMapPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const Eigen::Matrix4f &transformation);

    // 点云配准相关函数
    Eigen::Matrix4f icp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr &Final);
    Eigen::Matrix4f ndt_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr &Final);
    Eigen::Matrix4f gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt);

    // 发布和记录相关函数
    void publishTransform(const Eigen::Matrix4f &transformation_total);
    void filterPointCloudByField(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud);
    void projectPointCloudToXYPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void recordStatisticsToCSV(const std::string &filename, const std::string &algorithm,
                               float distance, float mean, float stddev);
    void recordTrajectory(const std::string &filename,
                          const Eigen::Matrix4f &transformation_total_, double timestamp_);
    void publishMarker(const Eigen::Matrix4f &transformation_total);
    void storeToBuffers(const Eigen::Matrix4f &transformation, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double timestamp);
};

#endif // POINTPROCESS_H