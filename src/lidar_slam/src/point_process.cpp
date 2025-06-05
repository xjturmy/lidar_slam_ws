#include "point_process.h"

PointCloudProcessor::PointCloudProcessor() : nh(), first_frame_flag_(true)
{
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_pointcloud", 10);
    pc_icp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Icp_pointcloud", 10);
    pc_ndt_pub = nh.advertise<sensor_msgs::PointCloud2>("/Ndt_pointcloud", 10);
    pc_gicp_pub = nh.advertise<sensor_msgs::PointCloud2>("/Gicp_pointcloud", 10);

    map_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    transformed_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    last_frame_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>();
    marker_pub_ = nh.advertise<visualization_msgs::Marker>("trajectory_marker", 10);

    imu_handler_ = std::make_unique<ImuDataHandler>();
}

void PointCloudProcessor::publishMarker(const Eigen::Matrix4f &transformation_total)
{
    float x = transformation_total(0, 3);
    float y = transformation_total(1, 3);
    float z = transformation_total(2, 3);
    geometry_msgs::Point point;
    point.x = x;
    point.y = y;
    point.z = z;
    trajectory_points_.push_back(point);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "trajectory";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    marker.scale.x = 0.1;

    marker.points = trajectory_points_;

    marker_pub_.publish(marker);
}

void PointCloudProcessor::publishTransform(const Eigen::Matrix4f &transformation_total)
{
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "base_link";

    transformStamped.transform.translation.x = transformation_total(0, 3);
    transformStamped.transform.translation.y = transformation_total(1, 3);
    transformStamped.transform.translation.z = transformation_total(2, 3);

    Eigen::Matrix3f rotation_matrix = transformation_total.block<3, 3>(0, 0);
    tf2::Matrix3x3 tf_rotation_matrix(rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2),
                                      rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2),
                                      rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2));
    tf2::Quaternion quaternion;
    tf_rotation_matrix.getRotation(quaternion);

    transformStamped.transform.rotation.x = quaternion.x();
    transformStamped.transform.rotation.y = quaternion.y();
    transformStamped.transform.rotation.z = quaternion.z();
    transformStamped.transform.rotation.w = quaternion.w();

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
    sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera1/points2/original", 10, &PointCloudProcessor::callback, this);
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

void computeCenterPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, Eigen::Vector3f &center_point)
{
    if (input_cloud->points.empty())
    {
        std::cerr << "输入点云为空，无法计算中心点。" << std::endl;
        return;
    }

    pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0.5, 2.5);
    pass.filter(*output_cloud);

    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.5, 0.5);
    pass.filter(*output_cloud);

    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 0.6);
    pass.filter(*output_cloud);

    if (output_cloud->points.empty())
    {
        std::cerr << "过滤后的点云为空，无法计算中心点。" << std::endl;
        return;
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*output_cloud, centroid);
    center_point = centroid.head<3>();
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

    Eigen::Vector3f translation = point2 - rotation * point1;
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation.topLeftCorner<3, 3>() = rotation;
    transformation.topRightCorner<3, 1>() = translation;

    return transformation;
}

void PointCloudProcessor::handleFirstFrame(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_001, double timestamp)
{
    *last_frame_points = *cloud_001;
    *map_points = *cloud_001;
    computeCenterPoint(cloud_001, center_point_map);
    reality_timestamp = timestamp;
    first_frame_flag_ = false;
}

void PointCloudProcessor::recordTrajectory(const std::string &filename, const Eigen::Matrix4f &transformation_total_, double timestamp_)
{
    std::ofstream csv_file;
    csv_file.open(filename, std::ios::app); // 以追加模式打开文件

    Eigen::Vector3d translation = transformation_total_.block<3, 1>(0, 3).cast<double>();
    Eigen::Matrix3d rotationMatrix = transformation_total_.block<3, 3>(0, 0).cast<double>();
    Eigen::Quaterniond rotation(rotationMatrix);

    csv_file << timestamp_ << " " << translation[0] << " " << translation[1] << " " << translation[2] << " "
             << rotation.x() << " " << rotation.y() << " " << rotation.z() << " " << rotation.w() << "\n";

    csv_file.close();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*filtered_cloud);
    return filtered_cloud;
}

void filterPointCloudByField(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud)
{
    if (input_cloud->points.empty())
    {
        std::cerr << "输入点云为空，无法进行滤波。" << std::endl;
        return;
    }
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 10);
    pass.filter(*output_cloud);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr filterPointCloudByHeight(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    filterPointCloudByField(cloud, filtered_cloud);
    return filtered_cloud;
}
Eigen::Matrix4f PointCloudProcessor::gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_src,
                                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(cloud_src);
    gicp.setInputTarget(cloud_tgt);
    gicp.align(*Final);
    return gicp.getFinalTransformation();
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

float PointCloudProcessor::calculateCorrespondenceDistances(const pcl::PointCloud<pcl::PointXYZ>::Ptr &Final,
                                                            const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_tgt)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_tgt);

    std::vector<float> distances;

    for (const auto &point : Final->points)
    {
        std::vector<int> indices(1);
        std::vector<float> squared_distances(1);
        kdtree.nearestKSearch(point, 1, indices, squared_distances);

        float distance = std::sqrt(squared_distances[0]);
        distances.push_back(distance);
    }

    float average_distance = 0.0f;
    for (const auto &distance : distances)
    {
        average_distance += distance;
    }
    average_distance /= distances.size();

    return average_distance;
}

void PointCloudProcessor::projectPointCloudToXYPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    for (auto &point : cloud->points)
    {
        point.z = 0; // 将 Z 值设置为 0
    }
}

void PointCloudProcessor::storeToBuffers(const Eigen::Matrix4f &transformation, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double timestamp)
{
    transformations.push_back(transformation);
    cloud_buffer.push_back(cloud);
    timestamps_buffer.push_back(timestamp);
}

g2o::SE3Quat matrixToSE3Quat(const Eigen::Matrix4f &transformation)
{
    Eigen::Vector3d translation = transformation.block<3, 1>(0, 3).cast<double>();
    Eigen::Matrix3d rotationMatrix = transformation.block<3, 3>(0, 0).cast<double>();
    Eigen::Quaterniond rotation(rotationMatrix);
    return g2o::SE3Quat(rotation, translation);
}
void optimizeTrajectory(std::vector<Eigen::Matrix4f> &transformations)
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    optimizer.setAlgorithm(solver);

    for (size_t i = 0; i < transformations.size(); ++i)
    {
        g2o::SE3Quat pose = matrixToSE3Quat(transformations[i]);
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(i);
        v->setEstimate(pose);
        if (i == 0)
        {
            v->setFixed(true);
        }
        optimizer.addVertex(v);
    }

    for (size_t i = 1; i < transformations.size(); ++i)
    {
        Eigen::Matrix4f relativeTransform = transformations[i - 1].inverse() * transformations[i];
        g2o::SE3Quat relative_pose = matrixToSE3Quat(relativeTransform);
        g2o::EdgeSE3 *e = new g2o::EdgeSE3();
        e->setVertex(0, optimizer.vertex(i - 1));
        e->setVertex(1, optimizer.vertex(i));
        e->setMeasurement(relative_pose);
        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        optimizer.addEdge(e);
    }

    std::cout << "Optimizing with " << optimizer.edges().size() << " edges" << std::endl;

    // optimizer.initializeOptimization();
    optimizer.optimize(100);

    for (size_t i = 0; i < transformations.size(); ++i)
    {
        g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
        transformations[i] = vertex->estimate().matrix().cast<float>();
    }
}

void PointCloudProcessor::updateMapPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const Eigen::Matrix4f &transformation)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transformation);
    *map_points += *transformed_cloud;
}

void PointCloudProcessor::optimizeAndPublishAll(const std::string &filename)
{
    optimizeTrajectory(transformations);
    for (size_t i = 0; i < transformations.size(); ++i)
    {
        publishMarker(transformations[i]);
        recordTrajectory(filename, transformations[i], timestamps_buffer[i]);
        publishTransform(transformations[i]);
        updateMapPointCloud(cloud_buffer[i], transformations[i]);
        publish_pointcloud(map_points, "map", pc_pub);
    }
}

void PointCloudProcessor::updateAndPublishRecent(const std::string &filename)
{
    transformations.erase(transformations.begin());
    cloud_buffer.erase(cloud_buffer.begin());
    timestamps_buffer.erase(timestamps_buffer.begin());

    optimizeTrajectory(transformations);
    publishMarker(transformations.back());
    recordTrajectory(filename, transformations.back(), timestamps_buffer.back());
    publishTransform(transformations.back());
    updateMapPointCloud(cloud_buffer.back(), transformations.back());
    publish_pointcloud(map_points, "map", pc_pub);
}

void PointCloudProcessor::handleBuffers(const std::string &filename)
{
    if (transformations.size() == 20)
    {
        optimizeAndPublishAll(filename);
    }
    else if (transformations.size() == 21)
    {
        updateAndPublishRecent(filename);
    }
}

void PointCloudProcessor::process_pointcloud(const sensor_msgs::PointCloud2::ConstPtr &pc_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*pc_msg, *cloud);
    Eigen::Matrix4f rotation_001 = imu_handler_->getMatrix001();
    std::cout <<  "获取到IMU的旋转矩阵：" << std::endl << rotation_001 << std::endl;

    // 旋转点云
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *rotated_cloud, rotation_001);
    // publish_pointcloud(rotated_cloud, "base_link", pc_gicp_pub);
    if (first_frame_flag_)
    {
        handleFirstFrame(rotated_cloud, pc_msg->header.stamp.toSec());
        return;
    }

    double timestamp = pc_msg->header.stamp.toSec() - reality_timestamp;
    // computeCenterPoint(rotated_cloud, center_point_current);
    // reality_transformation = computeTransformation(center_point_current, center_point_map);
    // recordTrajectory("real_trajectory_No_IMU.csv", reality_transformation, timestamp);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = downsamplePointCloud(rotated_cloud, 0.05f);

    pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame_points_filtered = filterPointCloudByHeight(cloud_filtered);

    Eigen::Matrix4f transformation_icp = gicp_registration(current_frame_points_filtered, last_frame_points);

    transformation_total_ = transformation_icp * transformation_total_;

    storeToBuffers(transformation_total_, cloud_filtered, timestamp);

    handleBuffers("test_trajectory.csv");

    *last_frame_points = *current_frame_points_filtered;
}