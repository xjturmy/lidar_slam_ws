#include "point_process.h"

PointCloudProcessor::PointCloudProcessor() : nh(), frame_count(0)
{
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pointcloud", 10);
    pc_gicp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Gicp_pointcloud", 10);
    map_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    base_to_map = Eigen::Matrix4f::Identity();
}

void PointCloudProcessor::publish_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points, const std::string &frame_id)
{
    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*points, pc_msg);
    pc_msg.header.stamp = ros::Time::now();
    pc_msg.header.frame_id = frame_id;
    pc_pub.publish(pc_msg);
}

void PointCloudProcessor::publish_GICP_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points, const std::string &frame_id)
{
    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*points, pc_msg);
    pc_msg.header.stamp = ros::Time::now();
    pc_msg.header.frame_id = frame_id;
    pc_gicp_pub.publish(pc_msg);
}

void PointCloudProcessor::start()
{
    std::string msg_name = "/sunny_topic/device_0A30_952B_10F9_3044/tof_frame/pointcloud_horizontal";
    sub = nh.subscribe<sensor_msgs::PointCloud2>(msg_name, 10, &PointCloudProcessor::callback, this);
    ros::spin();
}

void PointCloudProcessor::callback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg)
{
    try
    {
        process_pointcloud(pc_msg);
    }
    catch (const std::exception &e)
    {
        ROS_ERROR_STREAM("处理点云时发生错误: " << e.what());
    }
}

void PointCloudProcessor::process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pc_msg)
{
    // 示例：简单地将点云数据发布到另一个话题
    //TODO：添加主要处理流程
    publish_pointcloud(map_points, "map");
    publish_GICP_pointcloud(map_points, "map");
    frame_count++;
    std::cout << "处理了第" << frame_count << " 帧" << std::endl;
    // ROS_INFO_STREAM("处理了第 " << frame_count << " 帧");//日志显示未乱码，后面再解决
}