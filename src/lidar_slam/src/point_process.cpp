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

void computeCenterPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, Eigen::Vector3f &center_point)
{
    if (input_cloud->points.empty())
    {
        std::cerr << "输入点云为空，无法进行滤波。" << std::endl;
        return;
    }

    // 创建直通滤波器对象
    pcl::PassThrough<pcl::PointXYZ> pass;

    // 初始化输出点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 对 x 轴进行过滤
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0.5, 2.5);
    pass.filter(*output_cloud);

    // 对 y 轴进行过滤
    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.5, 0.5);
    pass.filter(*output_cloud);

    // 对 z 轴进行过滤
    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 0.5);
    pass.filter(*output_cloud);

    // 检查过滤后的点云是否为空
    if (output_cloud->points.empty())
    {
        std::cerr << "过滤后的点云为空，无法计算中心点。" << std::endl;
        return;
    }

    // 计算中心点坐标
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*output_cloud, centroid);
    center_point = centroid.head<3>(); // 使用Eigen::Vector3f来存储中心点坐标
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
    float resolution = 0.1f;
    ndt.setResolution(resolution); // 设置分辨率
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

// 将 Eigen::Matrix4f 转换为 g2o::SE3Quat
g2o::SE3Quat matrixToSE3Quat(const Eigen::Matrix4f &transformation)
{
    Eigen::Vector3d translation = transformation.block<3, 1>(0, 3).cast<double>();
    Eigen::Matrix3d rotationMatrix = transformation.block<3, 3>(0, 0).cast<double>();
    Eigen::Quaterniond rotation(rotationMatrix);
    return g2o::SE3Quat(rotation, translation);
}

void PointCloudProcessor::optimizeTrajectory(
    std::vector<Eigen::Matrix4f> &transformations,
    const std::vector<Eigen::Vector3f> &landmarks,
    const std::vector<int> &landmark_indices)
{
    // 创建优化器
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    // auto solver = std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>());
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    optimizer.setAlgorithm(solver);

    std::cout << "Adding " << transformations.size() << " poses and " << landmarks.size() << " landmarks." << std::endl;
    // 添加位姿顶点
    for (size_t i = 0; i < transformations.size(); ++i)
    {
        auto pose = matrixToSE3Quat(transformations[i]);
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(i);
        v->setEstimate(pose);
        if (i == 0)
        {
            v->setFixed(true); // 第一个顶点固定
        }
        optimizer.addVertex(v);
    }
    std::cout << "添加位姿顶点成功" << std::endl;
    // 添加路标顶点
    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        g2o::VertexPointXYZ *landmark = new g2o::VertexPointXYZ();
        int poseIdx = landmark_indices[i];
        landmark->setId(poseIdx);                           // 路标顶点的 ID 要与位姿顶点区分开
        landmark->setEstimate(landmarks[i].cast<double>()); // 设置路标点的初始估计值
        optimizer.addVertex(landmark);
    }
    std::cout << "添加路标顶点" << std::endl;

    // 遍历优化器中的所有顶点
    /*/
    for (g2o::OptimizableGraph::VertexIDMap::iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
    {
        g2o::HyperGraph::Vertex *vertex = it->second; // 获取顶点指针
        int id = it->first;                           // 获取顶点的 ID

        // 根据顶点类型获取位置
        if (dynamic_cast<g2o::VertexSE3 *>(vertex)) // 如果是位姿顶点
        {
            g2o::VertexSE3 *poseVertex = dynamic_cast<g2o::VertexSE3 *>(vertex);
            Eigen::Isometry3d pose = poseVertex->estimate(); // 获取位姿估计值
            std::cout << "位姿顶点 ID: " << id << ", 位姿: " << std::endl
                      << pose.matrix() << std::endl; // 输出位姿矩阵
        }
        else if (dynamic_cast<g2o::VertexPointXYZ *>(vertex)) // 如果是路标顶点
        {
            g2o::VertexPointXYZ *landmarkVertex = dynamic_cast<g2o::VertexPointXYZ *>(vertex);
            Eigen::Vector3d position = landmarkVertex->estimate(); // 获取路标位置
            std::cout << "路标顶点 ID: " << id << ", 位置: " << position.transpose() << std::endl;
        }
    }
*/
    // 添加位姿之间的边
    for (size_t i = 1; i < transformations.size(); ++i)
    {
        Eigen::Matrix4f relativeTransform = transformations[i - 1].inverse() * transformations[i];
        auto relative_pose = matrixToSE3Quat(relativeTransform);
        g2o::EdgeSE3 *e = new g2o::EdgeSE3();
        e->setVertex(0, optimizer.vertex(i - 1));
        e->setVertex(1, optimizer.vertex(i));
        e->setMeasurement(relative_pose);
        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        optimizer.addEdge(e);
    }
    std::cout << "添加位姿之间的边成功" << std::endl;

    // 打印所有路标顶点的索引
    for (size_t i = 0; i < landmark_indices.size(); ++i)
    {
        std::cout << "路标顶点索引: " << landmark_indices[i] << std::endl;
    }

    // 添加观测边
    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        int poseIdx = landmark_indices[i];                            // 当前观测点对应的位姿顶点索引
        Eigen::Vector3d observed_point = landmarks[i].cast<double>(); // 当前帧观测到的路标点
        g2o::EdgeSE3PointXYZ *edge = new g2o::EdgeSE3PointXYZ();

        // 设置边的顶点
        edge->setVertex(0, optimizer.vertex(poseIdx));                    // 位姿顶点
        edge->setVertex(1, optimizer.vertex(transformations.size() + i)); // 路标顶点

        // 输出位姿顶点和路标顶点的信息
        std::cout << "处理观测边 " << i << "：" << std::endl;

        // 获取位姿顶点并打印估计值
        g2o::VertexSE3 *poseVertex = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
        if (poseVertex)
        {
            std::cout << "位姿顶点 ID: " << i << ", 估计值: "
                      << poseVertex->estimate().matrix() << std::endl;
        }
        else
        {
            std::cerr << "无法获取位姿顶点 " << i << std::endl;
        }

        // 获取路标顶点并打印估计值
        g2o::VertexPointXYZ *landmarkVertex = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(poseIdx));
        if (landmarkVertex)
        {
            std::cout << "路标顶点 ID: " << poseIdx << ", 估计值: "
                      << landmarkVertex->estimate().transpose() << std::endl; // 注意：transpose() 用于将列向量转置为行向量以便打印
        }
        else
        {
            std::cerr << "无法获取路标顶点 " << poseIdx << std::endl;
        }

        // 设置边的观测值和信息矩阵
        edge->setMeasurement(observed_point);              // 观测值
        edge->setInformation(Eigen::Matrix3d::Identity()); // 设置信息矩阵

        std::cout << "观测边添加成功" << std::endl;
    }

    std::cout << "添加测试" << std::endl;
    // 优化
    // optimizer.iteration = ();
    // optimizer.setVerbose(true); // 打印优化过程中的详细信息
                                // 优化
    std::cout << "开始进行初始化检查" << std::endl;
    optimizer.initializeOptimization();
    std::cout << "完成初始化检查" << std::endl;
    optimizer.optimize(50);

    // 更新变换矩阵
    for (size_t i = 0; i < transformations.size(); ++i)
    {
        g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
        if (vertex)
        {
            transformations[i] = vertex->estimate().matrix().cast<float>();
        }
        else
        {
            std::cerr << "Vertex " << i << " is not of type VertexSE3" << std::endl;
        }
    }
}

void recordTrajectory(const std::string &filename, const Eigen::Matrix4f &transformation_total_, double timestamp_)
{
    std::ofstream csv_file;
    csv_file.open(filename, std::ios::app); // 以追加模式打开文件

    // 提取平移向量
    Eigen::Vector3d translation = transformation_total_.block<3, 1>(0, 3).cast<double>();

    // 提取旋转部分并转换为四元数
    Eigen::Matrix3d rotationMatrix = transformation_total_.block<3, 3>(0, 0).cast<double>();
    Eigen::Quaterniond rotation(rotationMatrix);

    // 把平移向量和四元数写入文件，使用空格分隔
    csv_file << timestamp_ << " " << translation[0] << " " << translation[1] << " " << translation[2] << " "
             << rotation.x() << " " << rotation.y() << " " << rotation.z() << " " << rotation.w() << "\n";

    csv_file.close();
}

Eigen::Matrix3f computeRotation(const Eigen::Vector3f &point1, const Eigen::Vector3f &point2)
{
    Eigen::Vector3f v1 = point1.normalized();
    Eigen::Vector3f v2 = point2.normalized();

    // 计算旋转轴和角度
    Eigen::Vector3f axis = v1.cross(v2);
    float angle = std::acos(std::min(std::max(v1.dot(v2), -1.0f), 1.0f));

    if (axis.norm() == 0)
    {
        // 当两个向量同向或反向时，直接返回身份矩阵或180度旋转矩阵
        if (angle < 1e-6)
            return Eigen::Matrix3f::Identity();
        else
            return Eigen::Matrix3f::Identity(); // 反向旋转可选
    }

    Eigen::AngleAxisf rotation(angle, axis.normalized());
    return rotation.toRotationMatrix();
}

Eigen::Matrix4f computeTransformation(const Eigen::Vector3f &point1, const Eigen::Vector3f &point2)
{
    Eigen::Matrix3f rotation = computeRotation(point1, point2);

    // 正确的平移应该是 point2 - rotation * point1
    Eigen::Vector3f translation = point2 - rotation * point1;
    // std::cout << "z轴平移量: " << translation.z() << std::endl;

    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation.topLeftCorner<3, 3>() = rotation;
    transformation.topRightCorner<3, 1>() = translation;

    return transformation;
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
        // 求箱子的中心点
        computeCenterPoint(cloud, center_point_map);
        // 打印中心点
        std::cout << "中心点: " << center_point_map.transpose() << std::endl;
        reality_timestamp = pc_msg->header.stamp.toSec();
        // // 将第一帧的中心点作为路标点
        // landmarks.push_back(center_point_map);
        // landmark_indices.push_back(0); // 第一帧的索引
        return;
    }

    // 获取当前时间戳
    double timestamp = pc_msg->header.stamp.toSec() - reality_timestamp;
    computeCenterPoint(cloud, center_point_current);
    // std::cout << "目标中心点: " << center_point_map.transpose() << "当前帧中心点: " << center_point_current.transpose() << std::endl;
    reality_transformation = computeTransformation(center_point_current, center_point_map);
    recordTrajectory("reality_trajectory_improved.csv", reality_transformation, timestamp);

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
    Eigen::Matrix4f transformation_icp = gicp_registration(current_frame_points_filtered, last_frame_points_filtered, Final_icp);
    // transformation_icp = transformation_icp * transformation_ndt;

    // 更新总变换矩阵
    transformation_total_ = transformation_icp * transformation_total_;

    publish_pointcloud(last_frame_points_filtered, "base_link", pc_pub_target);
    publish_pointcloud(current_frame_points_filtered, "base_link", pc_ndt_pub);
    publish_pointcloud(Final_icp, "base_link", pc_icp_pub);

    // 存储变换矩阵
    transformations.push_back(transformation_total_);
    cloud_buffer.push_back(cloud_filtered);
    timestamps_buffer.push_back(timestamp);
    // 每一帧都观测到了中心点，将中心点加入到路标点列表
    landmarks.push_back(center_point_current);
    // landmark_indices.push_back(frame_count - 2); // 当前帧的索引
    int landmark_index = frame_count + 10000 - 2;
    // 输出当前的路标索引
    std::cout << "当前路标索引: " << landmark_index << std::endl;
    landmark_indices.push_back(landmark_index); // 当前帧的索引
                                                // cout

    // 变换矩阵累计20就进优化
    // 取优窗口左边的变换矩阵
    // if (transformations.size() == 1)
    // {
    //     // 在新线程中调用优化函数
    //     // std::thread optimization_thread([this]() {

    //     // optimizeTrajectory(transformations); // 调用优化函数
    //     transformation_total_ = (*transformations.begin()) * transformation_total_;
    //     // 获取平移部分的x,y值，并记录在CSV文件中
    //     publishMarker(transformation_total_);
    //     recordTrajectory("No_Optimization_icp.csv",transformation_total_, *(timestamps_buffer.begin()));
    //     publishTransform(transformation_total_); // 发布坐标转换关系
    //     // 更新当前帧cloud转换到map并添加到地图点云中
    //     // publish_pointcloud(*cloud_buffer.begin(), "base_link", pc_ndt_pub);
    //     pcl::transformPointCloud(*(*cloud_buffer.begin()), *transformed_cloud, transformation_total_);
    //     // publish_pointcloud(transformed_cloud, "map", pc_icp_pub);
    //     *map_points += *transformed_cloud; // 合并转换后的点云数据到地图点云
    //     // 发布地图点云
    //     publish_pointcloud(map_points, "map", pc_pub);

    //     // 保持 transformations 和 cloud_buffer 的数量不变
    //     transformations.erase(transformations.begin());
    //     cloud_buffer.erase(cloud_buffer.begin());
    //     timestamps_buffer.erase(timestamps_buffer.begin());

    //     // });
    //     // optimization_thread.detach(); // 分离线程，让其在后台运行
    // }

    // 变换矩阵累计20就进优化
    // 取窗口右边的变换矩阵
    if (transformations.size() == 20)
    {
        // 在新线程中调用优化函数
        // std::thread optimization_thread([this]() {
        optimizeTrajectory(transformations, landmarks, landmark_indices); // 调用优化函数
        for (size_t i = 0; i < transformations.size(); ++i)
        {
            // 获取平移部分的x,y值，并记录在CSV文件中
            publishMarker(transformations[i]);
            recordTrajectory("Optimization_once_gicp.csv", transformations[i], timestamps_buffer[i]);
            publishTransform(transformations[i]); // 发布坐标转换关系
            // 更新当前帧cloud转换到map并添加到地图点云中
            pcl::transformPointCloud(*cloud_buffer[i], *transformed_cloud, transformations[i]);
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

        transformations.erase(transformations.begin());
        cloud_buffer.erase(cloud_buffer.begin());
        timestamps_buffer.erase(timestamps_buffer.begin());
        landmarks.erase(landmarks.begin());
        landmark_indices.erase(landmark_indices.begin());
        optimizeTrajectory(transformations, landmarks, landmark_indices); // 调用优化函数
        // 获取平移部分的x,y值，并记录在CSV文件中
        publishMarker(transformations.back());
        recordTrajectory("Optimization_once_gicp.csv", transformations.back(), timestamps_buffer.back());
        publishTransform(transformations.back()); // 发布坐标转换关系
        // 更新当前帧cloud转换到map并添加到地图点云中

        publish_pointcloud(cloud_buffer.back(), "base_link", pc_ndt_pub);
        pcl::transformPointCloud(*cloud_buffer.back(), *transformed_cloud, transformations.back());
        publish_pointcloud(transformed_cloud, "map", pc_icp_pub);

        *map_points += *transformed_cloud; // 合并转换后的点云数据到地图点云
        // 发布地图点云
        publish_pointcloud(map_points, "map", pc_pub);

        // 保持 transformations 和 cloud_buffer 的数量不变

        // });
        // optimization_thread.detach(); // 分离线程，让其在后台运行
    }

    // 发布测试点云，在车体坐标系下查看相邻帧匹配效果

    // 计算ICP对应点距离
    // float icp_distance = calculateCorrespondenceDistances(Final_icp, last_frame_points_filtered);
    // std::cout << "ICP 对应点距离: " << icp_distance * 100<< std::endl;

    // 更新目标帧点云
    pcl::copyPointCloud(*current_frame_points_filtered, *last_frame_points);
}
