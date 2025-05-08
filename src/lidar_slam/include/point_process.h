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

// Eigen 相关头文件
#include <Eigen/Core>

// ROS 和 tf2 相关头文件
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

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

    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pc_pub;
    ros::Publisher pc_pub_target;
    ros::Publisher pc_icp_pub;
    ros::Publisher pc_ndt_pub;
    ros::Publisher pc_gicp_pub;

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame_points;
    Eigen::Matrix4f base_to_map;
    int frame_count;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    Eigen::Matrix4f transformation_total_ = Eigen::Matrix4f::Identity();
};

#endif // POINTPROCESS_H