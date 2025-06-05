#include "point_process.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_slam");
    std::cout << "Lidar SLAM started" << std::endl;
    PointCloudProcessor processor;
    processor.start();
    return 0;
}