#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;         // 相机位姿

    ifstream fin("../../dense_RGBD/data/pose.txt"); // 读取姿态信息到fin数据流中
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {   // 遍历读取5张图片的信息
        
        boost::format fmt("../../dense_RGBD/data/%s/%d.%s"); //图像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));   // 读彩色图信息
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 使用-1读取原始图像（深度图）

        double data[7] = {0};   // 读pose.txt文件的位姿信息
        for (int j = 0; j < 7; j++) {
            fin >> data[j];
        }
        // cout << "collect pose from img" << i+1 << " to data" << endl;

        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);   // 四元数 q
        Eigen::Isometry3d T(q); // 用q创建 线性变换类实例 T
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); // 给变换矩阵 添加 平移向量
        poses.push_back(T);     // 最终的 T 是带有【平移】和【旋转】的矩阵
    }

    

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为点云..." << endl;

    // 定义点云使用的格式：这里用的是XYZRGB
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // 新建一个点云指针
    PointCloud::Ptr pointCloud(new PointCloud);
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i]; // 位姿变换矩阵
        for (int v = 0; v < color.rows; v++) {  // 遍历行
            for (int u = 0; u < color.cols; u++) {  // 遍历列
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                
                // 由像素坐标到世界坐标的转换
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                // 计算完的结果赋值给 xyzbgr
                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];

                // 保存数据
                current->points.push_back(p);
            }
        }
        
        // depth filter and statistical removal 
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(50);    // 用 50 个点算平均距离
        statistical_filter.setStddevMulThresh(1.0);// 标准差乘数
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;  // 更新滤波后的点云
    }

    pointCloud->is_dense = false;
    cout << "点云共有" << pointCloud->size() << "个点." << endl;

    // voxel filter 体素网络滤波器 降采样
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.03;
    voxel_filter.setLeafSize(resolution, resolution, resolution);// 设置网格大小 resolution
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);// 输入点云
    voxel_filter.filter(*tmp);  // 滤波完的结果存放在 tmp 中
    tmp->swap(*pointCloud);     // 数据存放回 pointCloud 中

    cout << "滤波之后，点云共有" << pointCloud->size() << "个点." << endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;
}