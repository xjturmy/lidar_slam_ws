#ifndef POINTPROCESS_H
#define POINTPROCESS_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>

class PointCloudProcessor {
public:
    PointCloudProcessor();
    void start();

private:
    void process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr& pc_msg);
    void callback(const sensor_msgs::PointCloud2::ConstPtr& pc_msg);
    void publish_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& points, const std::string& frame_id);
    void publish_GICP_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& points, const std::string& frame_id);

    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pc_pub;
    ros::Publisher pc_gicp_pub;

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_points;
    Eigen::Matrix4f base_to_map;
    int frame_count;
};

#endif // POINTPROCESS_H