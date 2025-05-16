#include "point_process.h"

PointCloudProcessor::PointCloudProcessor() : nh(), frame_count(0)
{
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pointcloud", 10);
    pc_pub_target = nh.advertise<sensor_msgs::PointCloud2>("/map_pointcloud_target", 10);
    pc_icp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Icp_pointcloud", 10);
    pc_ndt_pub = nh.advertise<sensor_msgs::PointCloud2>("/Ndt_pointcloud", 10);
    pc_gicp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Gicp_pointcloud", 10);
    
    map_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    transformed_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    last_frame_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    current_frame_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    base_to_map = Eigen::Matrix4f::Identity();

    // 创建一个transform broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>();
    marker_pub_ = nh.advertise<visualization_msgs::Marker>("trajectory_marker", 10);
}    
void PointCloudProcessor::publishMarker(const Eigen::Matrix4f &transformation_total)
{
    float x = transformation_total(0, 3);
    float y = transformation_total(1, 3);
    float z = transformation_total(2, 3);
    // 添加当前点到轨迹点列表
        geometry_msgs::Point point;
        point.x = x;
        point.y = y;
        point.z = z;
        trajectory_points_.push_back(point);

        // 创建Marker消息
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "trajectory";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;

        // 设置颜色
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        // 设置线宽
        marker.scale.x = 0.1;

        // 添加所有点到Marker消息
        marker.points = trajectory_points_;

        // 发布Marker消息
        marker_pub_.publish(marker);
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



// 修正后的函数：将 Eigen::Matrix4f 转换为 g2o::SE3Quat
g2o::SE3Quat matrixToSE3Quat(const Eigen::Matrix4f& transformation) {
    // 提取平移向量
    Eigen::Vector3d translation = transformation.block<3, 1>(0, 3).cast<double>();

    // 提取旋转部分并转换为四元数
    Eigen::Matrix3d rotationMatrix = transformation.block<3, 3>(0, 0).cast<double>();
    Eigen::Quaterniond rotation(rotationMatrix);
    
    // 直接使用四元数构造 SE3
    return g2o::SE3Quat(rotation, translation);
}

void PointCloudProcessor::optimizeTrajectory(std::vector<Eigen::Matrix4f>& transformations) {
    // 创建优化器
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 添加顶点
    for (size_t i = 0; i < transformations.size(); ++i) {
        auto pose = matrixToSE3Quat(transformations[i]);
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(i);
        v->setEstimate(pose);
        
        // 设置第一个顶点为固定点，提供参考系
        if (i == 0) {
            v->setFixed(true);
        }
        
        optimizer.addVertex(v);
    }

    // 添加边
    for (size_t i = 1; i < transformations.size(); ++i) {
        // 计算相对位姿：T_{i-1}^{-1} * T_i
        Eigen::Matrix4f relativeTransform = transformations[i-1].inverse() * transformations[i];
        auto relative_pose = matrixToSE3Quat(relativeTransform);
        
        g2o::EdgeSE3* e = new g2o::EdgeSE3();
        e->setVertex(0, optimizer.vertex(i - 1));
        e->setVertex(1, optimizer.vertex(i));
        e->setMeasurement(relative_pose);
        
        // 设置信息矩阵，可以根据实际情况调整
        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        
        optimizer.addEdge(e);
    }

    // 优化
    optimizer.initializeOptimization();
    optimizer.optimize(50);

    // 更新变换矩阵
    for (size_t i = 0; i < transformations.size(); ++i) {
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        if (vertex) {
            // std::cout << "优化前: \n" << transformations[i] << std::endl;
            transformations[i] = vertex->estimate().matrix().cast<float>();
            // std::cout << "优化后: \n" << transformations[i] << std::endl;
        } else {
            std::cerr << "Vertex " << i << " is not of type VertexSE3" << std::endl;
        }
    }
}

void recordTrajectory(std::ofstream& csv_file,const Eigen::Matrix4f transformation_total_) {
    csv_file.open("F20_after_optimization_Right_registration_distances.csv", std::ios::app);
    // 获取平移部分的x,y值，并记录在CSV文件中
    float translation_x = transformation_total_.block<3, 1>(0, 3)[0];
    float translation_y = transformation_total_.block<3, 1>(0, 3)[1];
    csv_file << translation_x << "," << translation_y << "\n";
    csv_file.close();
    }

void PointCloudProcessor::process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pc_msg)
{
    frame_count++;
    std::cout << "处理第" << frame_count << " 帧" << std::endl;

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
    vg.filter(*cloud_filtered);

    // 直通滤波过滤掉高度小于0.1的点
    pcl::copyPointCloud(*cloud_filtered, *current_frame_points);
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame_points_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame_points_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    filterPointCloudByField(last_frame_points, last_frame_points_filtered);
    filterPointCloudByField(current_frame_points, current_frame_points_filtered);

    // 相邻帧匹配（icp）
    // pcl::PointCloud<pcl::PointXYZ>::Ptr Final_ndt(new pcl::PointCloud<pcl::PointXYZ>);
    // Eigen::Matrix4f transformation_ndt = ndt_registration(current_frame_points_filtered, last_frame_points_filtered, Final_ndt);
    pcl::PointCloud<pcl::PointXYZ>::Ptr Final_icp(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f transformation_icp = icp_registration(current_frame_points_filtered, last_frame_points_filtered, Final_icp);


    // publish_pointcloud(last_frame_points_filtered, "base_link", pc_pub_target);
    // 存储变换矩阵
    transformations.push_back(transformation_icp);
    cloud_buffer.push_back(cloud_filtered);

    // 变换矩阵累计20就进优化
    // 取优窗口左边的变换矩阵
    // if (transformations.size() == 20)
    // {
    //     // 在新线程中调用优化函数
    //     // std::thread optimization_thread([this]() { 

    //     optimizeTrajectory(transformations); // 调用优化函数
    //     transformation_total_ = (*transformations.begin()) * transformation_total_;
    //     // 获取平移部分的x,y值，并记录在CSV文件中
    //     publishMarker(transformation_total_);
    //     recordTrajectory(csv_file,transformation_total_);
    //     publishTransform(transformation_total_); // 发布坐标转换关系
    //     // 更新当前帧cloud转换到map并添加到地图点云中
    //     pcl::transformPointCloud(*(*cloud_buffer.begin()), *transformed_cloud, transformation_total_);
    //     *map_points += *transformed_cloud; // 合并转换后的点云数据到地图点云
    //     // 发布地图点云
    //     publish_pointcloud(map_points, "map", pc_pub);

    //     // 保持 transformations 和 cloud_buffer 的数量不变
    //     transformations.erase(transformations.begin());
    //     cloud_buffer.erase(cloud_buffer.begin());

    //     // });
    //     // optimization_thread.detach(); // 分离线程，让其在后台运行
    // }

    // 变换矩阵累计20就进优化
    // 取窗口右边的变换矩阵
            if (transformations.size() == 20)
            {
                // 在新线程中调用优化函数
                // std::thread optimization_thread([this]() { 

                optimizeTrajectory(transformations); // 调用优化函数
                for (size_t i = 0; i < transformations.size(); ++i)
                {
                    transformation_total_ = transformations[i] * transformation_total_;
                    // 获取平移部分的x,y值，并记录在CSV文件中
                    publishMarker(transformation_total_);
                    recordTrajectory(csv_file,transformation_total_);
                    publishTransform(transformation_total_); // 发布坐标转换关系
                    // 更新当前帧cloud转换到map并添加到地图点云中
                    pcl::transformPointCloud(*cloud_buffer[i], *transformed_cloud, transformation_total_);
                    *map_points += *transformed_cloud; // 合并转换后的点云数据到地图点云
                    // 发布地图点云
                    publish_pointcloud(map_points, "map", pc_pub);
                }

                // });
                // optimization_thread.detach(); // 分离线程，让其在后台运行
            }

            if (transformations.size() == 21)
            {
                // 在新线程中调用优化函数
                // std::thread optimization_thread([this]() { 

                optimizeTrajectory(transformations); // 调用优化函数
                transformation_total_ = transformations.back() * transformation_total_;
                // 获取平移部分的x,y值，并记录在CSV文件中
                publishMarker(transformation_total_);
                recordTrajectory(csv_file,transformation_total_);                
                publishTransform(transformation_total_); // 发布坐标转换关系
                // 更新当前帧cloud转换到map并添加到地图点云中

                // publish_pointcloud(cloud_buffer.back(), "base_link", pc_ndt_pub);
                pcl::transformPointCloud(*cloud_buffer.back(), *transformed_cloud, transformation_total_);
                // publish_pointcloud(transformed_cloud, "map", pc_icp_pub);

                *map_points += *transformed_cloud; // 合并转换后的点云数据到地图点云
                // 发布地图点云
                publish_pointcloud(map_points, "map", pc_pub);

                // 保持 transformations 和 cloud_buffer 的数量不变
                transformations.erase(transformations.begin());
                cloud_buffer.erase(cloud_buffer.begin());

                // });
                // optimization_thread.detach(); // 分离线程，让其在后台运行
            }



    // 发布测试点云，在车体坐标系下查看相邻帧匹配效果
    
    
    // publish_pointcloud(Final_icp, "base_link", pc_icp_pub);


    // 计算ICP对应点距离
    // float icp_distance = calculateCorrespondenceDistances(Final_icp, last_frame_points_filtered);
    // std::cout << "ICP 对应点距离: " << icp_distance * 100<< std::endl;

    // 更新目标帧点云
    pcl::copyPointCloud(*current_frame_points_filtered, *last_frame_points);
}
