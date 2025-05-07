#include "point_process.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_slam");
    PointCloudProcessor processor;
    processor.start();
    return 0;
}