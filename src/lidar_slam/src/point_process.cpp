#include "point_process.h"

PointCloudProcessor::PointCloudProcessor() : nh(), frame_count(0)
{
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pointcloud", 10);
    pc_pub_target = nh.advertise<sensor_msgs::PointCloud2>("/map_pointcloud_target", 10);
    pc_icp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Icp_pointcloud", 10);
    pc_ndt_pub = nh.advertise<sensor_msgs::PointCloud2>("/Ndt_pointcloud", 10);
    pc_gicp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Gicp_pointcloud", 10);
    
    map_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    last_frame_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    current_frame_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    base_to_map = Eigen::Matrix4f::Identity();

    // 创建一个transform broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>();
}

void PointCloudProcessor::publishTransform(const Eigen::Matrix4f &transformation_total)
{
    // 创建一个TransformStamped消息
    geometry_msgs::TransformStamped transformStamped;

    // 设置消息的header
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "base_link";

    // 提取平移部分
    transformStamped.transform.translation.x = transformation_total(0, 3);
    transformStamped.transform.translation.y = transformation_total(1, 3);
    transformStamped.transform.translation.z = transformation_total(2, 3);

    // 提取旋转部分（从旋转矩阵转换为四元数）
    Eigen::Matrix3f rotation_matrix;
    rotation_matrix = transformation_total.block<3, 3>(0, 0);

    tf2::Matrix3x3 tf_rotation_matrix(rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2),
                                      rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2),
                                      rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2));

    tf2::Quaternion quaternion;
    tf_rotation_matrix.getRotation(quaternion);

    transformStamped.transform.rotation.x = quaternion.x();
    transformStamped.transform.rotation.y = quaternion.y();
    transformStamped.transform.rotation.z = quaternion.z();
    transformStamped.transform.rotation.w = quaternion.w();

    // 发布变换
    tf_broadcaster_->sendTransform(transformStamped);
}

void PointCloudProcessor::publish_pointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points,
                                             const std::string &frame_id,
                                             ros::Publisher &pub)
{
    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*points, pc_msg);
    pc_msg.header.stamp = ros::Time::now();
    pc_msg.header.frame_id = frame_id;
    pub.publish(pc_msg);
}

void PointCloudProcessor::start()
{
    // std::string msg_name = "/sunny_topic/device_0A30_952B_10F9_3044/tof_frame/pointcloud_horizontal";
    std::string msg_name = "/camera1/points2/original";

    
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

// 计算配准后的点云和目标点云对应点的距离
float PointCloudProcessor::calculateCorrespondenceDistances(const pcl::PointCloud<pcl::PointXYZ>::Ptr &Final,
                                                            const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt)
{
    // 创建 KD 树
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_tgt);

    // 存储对应点的距离
    std::vector<float> distances;

    // 遍历配准后的点云
    for (const auto &point : Final->points)
    {
        // 查找最近的对应点
        std::vector<int> indices(1);
        std::vector<float> squared_distances(1);
        kdtree.nearestKSearch(point, 1, indices, squared_distances);

        // 计算欧几里得距离
        float distance = std::sqrt(squared_distances[0]);
        distances.push_back(distance);
    }

    // 计算平均距离
    float average_distance = 0.0f;
    for (const auto &distance : distances)
    {
        average_distance += distance;
    }
    average_distance /= distances.size();

    // 输出结果
    return average_distance;
}

void PointCloudProcessor::compareRegistrationAlgorithms()
{
    // 打开 CSV 文件
    std::ofstream csv_file("/home/keyi/Documents/01_slam/lidar_slam_ws/data/registration_distances.csv");
    if (!csv_file.is_open())
    {
        std::cerr << "无法打开 CSV 文件" << std::endl;
        return;
    }

    // 写入 CSV 文件的表头
    csv_file << "位移,ICP距离,NDT距离,GICP距离\n";

    // 加载点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/gongyou/Documents/01_slam/lidar_slam_ws/data/bunny.pcd", *cloud_src) == -1)
    {
        std::cout << "Cloud_src file not found" << std::endl;
        return;
    }

    // 定义位移范围和步长
    float max_translation = 1.0; // 最大位移量
    float step_size = 0.1;       // 位移步长
    // 发布目标点云
    publish_pointcloud(cloud_src, "map", pc_pub);

    for (float translation = 0.0; translation <= max_translation; translation += step_size)
    {
        // 对目标点云进行位移和旋转
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() << translation, 0.0, 0.0;
        pcl::transformPointCloud(*cloud_src, *cloud_tgt, transform);

        // 发布目标点云
        publish_pointcloud(cloud_tgt, "map", pc_pub_target);

        // 创建配准后的点云对象
        pcl::PointCloud<pcl::PointXYZ>::Ptr Final_icp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr Final_ndt(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr Final_gicp(new pcl::PointCloud<pcl::PointXYZ>);

        // ICP 配准
        Eigen::Matrix4f transformation_icp = icp_registration(cloud_src, cloud_tgt, Final_icp);
        std::cout << "位移 " << translation << " 米时的 ICP 转换矩阵:" << std::endl
                  << transformation_icp << std::endl;

        // NDT 配准
        Eigen::Matrix4f transformation_ndt = ndt_registration(cloud_src, cloud_tgt, Final_ndt);
        std::cout << "位移 " << translation << " 米时的 NDT 转换矩阵:" << std::endl
                  << transformation_ndt << std::endl;

        // GICP 配准
        Eigen::Matrix4f transformation_gicp = gicp_registration(cloud_src, cloud_tgt, Final_gicp);
        std::cout << "位移 " << translation << " 米时的 GICP 转换矩阵:" << std::endl
                  << transformation_gicp << std::endl;
        publish_pointcloud(Final_icp, "map", pc_icp_pub);
        publish_pointcloud(Final_ndt, "map", pc_ndt_pub);
        publish_pointcloud(Final_gicp, "map", pc_gicp_pub);
        // 计算对应点的距离
        float icp_distance = calculateCorrespondenceDistances(Final_icp, cloud_tgt);
        float ndt_distance = calculateCorrespondenceDistances(Final_ndt, cloud_tgt);
        float gicp_distance = calculateCorrespondenceDistances(Final_gicp, cloud_tgt);

        // 输出对应点的距离
        std::cout << "位移 " << translation << " 米时的 ICP 对应点距离: " << icp_distance << std::endl;
        std::cout << "位移 " << translation << " 米时的 NDT 对应点距离: " << ndt_distance << std::endl;
        std::cout << "位移 " << translation << " 米时的 GICP 对应点距离: " << gicp_distance << std::endl;

        // 将结果写入 CSV 文件
        csv_file << translation << "," << icp_distance << "," << ndt_distance << "," << gicp_distance << "\n";
    }

    // 关闭 CSV 文件
    csv_file.close();
}

// 封装的直通滤波函数
void PointCloudProcessor::filterPointCloudByField(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud)
{
    if (input_cloud->points.empty())
    {
        std::cerr << "输入点云为空，无法进行滤波。" << std::endl;
        return;
    }

    // 创建直通滤波器对象
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 10);
    pass.filter(*output_cloud); // 执行滤波操作

    return;
}

// 将点云投影到X-Y平面
void PointCloudProcessor::projectPointCloudToXYPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    // 遍历输入点云中的每个点，将其 Z 值设置为 0
    for (auto &point : cloud->points)
    {
        point.z = 0; // 将 Z 值设置为 0
    }
}

Eigen::Matrix4f PointCloudProcessor::icp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                                      pcl::PointCloud<pcl::PointXYZ>::Ptr &Final)
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_src);
    icp.setInputTarget(cloud_tgt);
    icp.align(*Final);
    return icp.getFinalTransformation();
}

Eigen::Matrix4f PointCloudProcessor::ndt_registration_test(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                                           pcl::PointCloud<pcl::PointXYZ>::Ptr &Final,
                                                           float &resolution,
                                                           float &step_size,
                                                           int &max_iterations)
{
    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
    // 创建 NDT 对象
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    // std::cout << "ndt_registration_test 1 index: " << resolution << std::endl;
    // 设置 NDT 参数
    float resolution_temp = 0.02f;
    // int max_iterations = 50;
    // float transformation_epsilon = 1e-6;                  // 设置变换 epsilon
    // float step_size = 0.1; // 设置步长
    // step_size = step_size * index;
    ndt.setResolution(resolution); // 设置分辨率
    // std::cout << "ndt_registration_test 2 index: " << resolution << std::endl;
    ndt.setMaximumIterations(max_iterations); // 设置最大迭代次数
    // ndt.setTransformationEpsilon(transformation_epsilon); // 设置变换 epsilon
    ndt.setStepSize(step_size); // 设置步长

    // 设置输入点云
    ndt.setInputSource(cloud_src);
    ndt.setInputTarget(cloud_tgt);

    // 执行对齐
    ndt.align(*Final);

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算运行时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "NDT Registration took " << duration.count() * 1000 << " milliseconds (" << duration.count() << " seconds)." << std::endl;
    // 返回最终变换矩阵
    return ndt.getFinalTransformation();
}

Eigen::Matrix4f PointCloudProcessor::ndt_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                                      pcl::PointCloud<pcl::PointXYZ>::Ptr &Final)
{
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    // float resolution = 0.2f;
    // ndt.setResolution(resolution); // 设置分辨率
    ndt.setInputSource(cloud_src);
    ndt.setInputTarget(cloud_tgt);
    ndt.align(*Final);
    return ndt.getFinalTransformation();
}

Eigen::Matrix4f PointCloudProcessor::gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt,
                                                       pcl::PointCloud<pcl::PointXYZ>::Ptr &Final)
{
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(cloud_src);
    gicp.setInputTarget(cloud_tgt);
    gicp.align(*Final);
    return gicp.getFinalTransformation();
}

void PointCloudProcessor::recordStatisticsToCSV(const std::string &filename, const std::string &algorithm,
                                                float distance, float mean, float stddev)
{
    std::ofstream csv_file(filename, std::ios::app); // 以追加模式打开文件
    if (!csv_file.is_open())
    {
        std::cerr << "无法打开 CSV 文件：" << filename << std::endl;
        return;
    }

    // 如果文件是新创建的，写入表头
    if (csv_file.tellp() == 0)
    {
        csv_file << "Frame,Algorithm,Distance,Mean,StdDev\n";
    }

    // 写入当前帧的统计信息
    csv_file << frame_count << "," << algorithm << "," << distance << "," << mean << "," << stddev << "\n";

    csv_file.close();
}


void PointCloudProcessor::process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pc_msg)
{
    frame_count++;
    std::cout << "处理了第" << frame_count << " 帧" << std::endl;

    // 将得到的话题数据转换为PCL格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*pc_msg, *cloud);

    // 初始化第一帧
    if (frame_count == 1)
    {
        *last_frame_points = *cloud;
        *map_points = *cloud;
        return;
    }
    // 将点云进行体素降采样
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*cloud);

    // 直通滤波过滤掉高度小于0.1的点
    pcl::copyPointCloud(*cloud, *current_frame_points);
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame_points_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame_points_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    filterPointCloudByField(last_frame_points, last_frame_points_filtered);
    filterPointCloudByField(current_frame_points, current_frame_points_filtered);

    // 将滤波后的点云z值设置为0
    // projectPointCloudToXYPlane(last_frame_points_filtered);
    // projectPointCloudToXYPlane(current_frame_points_filtered);

    // 相邻帧匹配（ndt+icp）


    pcl::PointCloud<pcl::PointXYZ>::Ptr Final_icp(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f transformation_icp = icp_registration(current_frame_points_filtered, last_frame_points_filtered, Final_icp);


    // 更新base_link到map的变换矩阵
    Eigen::Matrix4f tf_between_frames =  transformation_icp ;

    transformation_total_ = tf_between_frames * transformation_total_;

    publishTransform(transformation_total_); // 发布坐标转换关系
    // 更新当前帧cloud转换到map并添加到地图点云
    pcl::transformPointCloud(*cloud, *map_points, transformation_total_);


    // 发布测试点云，在车体坐标系下查看相邻帧匹配效果
    publish_pointcloud(current_frame_points_filtered, "base_link", pc_pub);
    publish_pointcloud(last_frame_points_filtered, "base_link", pc_pub_target);
    // publish_pointcloud(Final_ndt, "base_link", pc_ndt_pub);
    publish_pointcloud(Final_icp, "base_link", pc_icp_pub);
    // publish_pointcloud(Final_gicp, "base_link", pc_gicp_pub);



    // 计算ICP对应点距离
    float icp_distance = calculateCorrespondenceDistances(Final_icp, last_frame_points_filtered);

    // 更新ICP统计信息
    icp_distances.push_back(icp_distance);
    icp_distance_sum += icp_distance;
    icp_distance_mean = icp_distance_sum / icp_distances.size();

    // // 计算ICP标准差
    float icp_sum_of_squares = 0.0f;
    for (float dist : icp_distances)
    {
        icp_sum_of_squares += (dist - icp_distance_mean) * (dist - icp_distance_mean);
    }
    icp_distance_stddev = std::sqrt(icp_sum_of_squares / icp_distances.size());

    //     // 记录统计信息到CSV文件
    // recordStatisticsToCSV("statistics_ICP.csv", "ICP", icp_distance, icp_distance_mean, icp_distance_stddev);

    // 计算GICP对应点距离
    // float gicp_distance = calculateCorrespondenceDistances(Final_gicp, last_frame_points_filtered);

    // // 更新GICP统计信息
    // gicp_distances.push_back(gicp_distance);
    // gicp_distance_sum += gicp_distance;
    // gicp_distance_mean = gicp_distance_sum / gicp_distances.size();

    // // 计算GICP标准差
    // float gicp_sum_of_squares = 0.0f;
    // for (float dist : gicp_distances)
    // {
    //     gicp_sum_of_squares += (dist - gicp_distance_mean) * (dist - gicp_distance_mean);
    // }
    // gicp_distance_stddev = std::sqrt(gicp_sum_of_squares / gicp_distances.size());

    // 输出统计信息
    // std::cout << "NDT 对应点距离: " << ndt_distance * 100 << std::endl;
    // std::cout << "NDT 对应点距离的平均值: " << ndt_distance_mean * 100<< std::endl;
    // std::cout << "NDT 对应点距离的标准差: " << ndt_distance_stddev * 100<< std::endl;

    std::cout << "ICP 对应点距离: " << icp_distance * 100<< std::endl;
    std::cout << "ICP 对应点距离的平均值: " << icp_distance_mean * 100<< std::endl;
    std::cout << "ICP 对应点距离的标准差: " << icp_distance_stddev * 100<< std::endl;

    // std::cout << "GICP 对应点距离: " << gicp_distance * 100<< std::endl;
    // std::cout << "GICP 对应点距离的平均值: " << gicp_distance_mean * 100<< std::endl;
    // std::cout << "GICP 对应点距离的标准差: " << gicp_distance_stddev * 100<< std::endl;
    // 更新目标帧点云
    pcl::copyPointCloud(*cloud, *last_frame_points);
}
