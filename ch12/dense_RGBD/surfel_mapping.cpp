//
// Created by gaoxiang on 19-4-25.
//

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

// typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

// 表面重建函数，将pcl库中的函数进行封装
SurfelCloudPtr reconstructSurface(const PointCloudPtr &input, float radius, int polynomial_order) {
    pcl::MovingLeastSquares<PointT, SurfelT> mls;   // 移动最小二乘法 将点云处理成平滑的面
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>); // k叉树
    mls.setSearchMethod(tree);              // 设置查找方法为 k叉树查找
    mls.setSearchRadius(radius);            // 设置拟合 最邻近半径
    mls.setComputeNormals(true);            // 是否保存法线
    mls.setSqrGaussParam(radius * radius);  // 设置邻居基于距离加权的参数
    // mls.setPolynomialFit(polynomial_order > 1);
    mls.setPolynomialOrder(polynomial_order);// 设置拟合多项式的阶数
    mls.setInputCloud(input);               // 输入点云
    SurfelCloudPtr output(new SurfelCloud); // 新建输出点云指针
    mls.process(*output);                   //计算，结果存放到output中
    return (output);
}

pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels) {// 网格三角化
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>);// k叉树
    tree->setInputCloud(surfels); // 导入点云

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3;    // 对点云三角化
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh);// 返回值 存放返回数据

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.05);

    // Set typical values for the parameters
    gp3.setMu(2.5); // 乘数？
    gp3.setMaximumNearestNeighbors(100);    // 最大邻近值个数
    gp3.setMaximumSurfaceAngle(M_PI / 4);   // 最大表面角度 45 degrees
    gp3.setMinimumAngle(M_PI / 18);         // 最小角度 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3);      // 最大角度 120 degrees
    gp3.setNormalConsistency(true);         // 输入法线方向是否一致？ 是。

    // Get result
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);

    return triangles;
}

int main(int argc, char **argv) {

    // Load the points
    PointCloudPtr cloud(new PointCloud);
    if (argc == 0 || pcl::io::loadPCDFile(argv[1], *cloud)) {   // 第二个表达式，返回0表示成功，返回负数表示失败
        cout << "failed to load point cloud!";
        return 1;
    }
    cout << "point cloud loaded, points: " << cloud->points.size() << endl;

    // Compute surface elements
    cout << "computing normals ... " << endl;
    double mls_radius = 0.05, polynomial_order = 2;
    auto surfels = reconstructSurface(cloud, mls_radius, polynomial_order);// 重建表面

    // Compute a greedy surface triangulation
    cout << "computing mesh ... " << endl;
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels);

    cout << "display mesh ... " << endl;
    pcl::visualization::PCLVisualizer vis;
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame");
    vis.addPolygonMesh(*mesh, "mesh");
    vis.resetCamera();
    vis.spin();
}