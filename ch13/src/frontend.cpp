//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

    Frontend::Frontend() {
        gftt_ =
            cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");
    }

    bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
        current_frame_ = frame;

        switch (status_) {
            case FrontendStatus::INITING:
                StereoInit();   // 初始化双目
                break;
            case FrontendStatus::TRACKING_GOOD:
            case FrontendStatus::TRACKING_BAD:
                Track();
                break;
            case FrontendStatus::LOST:
                Reset();
                break;
        }

        last_frame_ = current_frame_;
        return true;
    }

    void Frontend::SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

    bool Frontend::Track() {
        if (last_frame_) {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
        }

        int num_track_last = TrackLastFrame();
        tracking_inliers_ = EstimateCurrentPose();

        if (tracking_inliers_ > num_features_tracking_) {
            // tracking good
            status_ = FrontendStatus::TRACKING_GOOD;
        } else if (tracking_inliers_ > num_features_tracking_bad_) {
            // tracking bad
            status_ = FrontendStatus::TRACKING_BAD;
        } else {
            // lost
            status_ = FrontendStatus::LOST;
        }

        InsertKeyframe();
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

        if (viewer_) viewer_->AddCurrentFrame(current_frame_);
        return true;
    }

    bool Frontend::InsertKeyframe() {
        if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
            // still have enough features, don't insert keyframe
            return false;
        }
        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                << current_frame_->keyframe_id_;

        SetObservationsForKeyFrame();
        DetectFeatures();  // detect new features

        // track in right image
        FindFeaturesInRight();
        // triangulate map points
        TriangulateNewPoints();
        // update backend because we have a new keyframe
        backend_->UpdateMap();

        if (viewer_) viewer_->UpdateMap();

        return true;
    }

    void Frontend::SetObservationsForKeyFrame() {
        for (auto &feat : current_frame_->features_left_) {
            auto mp = feat->map_point_.lock();
            if (mp) mp->AddObservation(feat);
        }
    }

    int Frontend::TriangulateNewPoints() {
        std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
        SE3 current_pose_Twc = current_frame_->Pose().inverse();
        int cnt_triangulated_pts = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_left_[i]->map_point_.expired() &&
                current_frame_->features_right_[i] != nullptr) {
                // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
                std::vector<Vec3> points{
                    camera_left_->pixel2camera(
                        Vec2(current_frame_->features_left_[i]->position_.pt.x,
                            current_frame_->features_left_[i]->position_.pt.y)),
                    camera_right_->pixel2camera(
                        Vec2(current_frame_->features_right_[i]->position_.pt.x,
                            current_frame_->features_right_[i]->position_.pt.y))};
                Vec3 pworld = Vec3::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    pworld = current_pose_Twc * pworld;
                    new_map_point->SetPos(pworld);
                    new_map_point->AddObservation(
                        current_frame_->features_left_[i]);
                    new_map_point->AddObservation(
                        current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    int Frontend::EstimateCurrentPose() {
        // setup g2o 用g2o求解
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
        );  // 初始化求解器
        g2o::SparseOptimizer optimizer; // 新建优化器
        optimizer.setAlgorithm(solver); // 设置优化器的求解器

        // vertex 顶点
        VertexPose *vertex_pose = new VertexPose();  // 新建 相机位姿 顶点 camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.addVertex(vertex_pose);// 向优化器添加顶点

        // K
        Mat33 K = camera_left_->K();    // 获取相机内参

        // 图优化的边 edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;    // 边的容器
        std::vector<Feature::Ptr> features;             // 特征点指针的容器
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {// 遍历左图像特征点
            auto mp = current_frame_->features_left_[i]->map_point_.lock(); // 互斥锁
            if (mp) {
                features.push_back(current_frame_->features_left_[i]);  // 保存 当前帧 左相机的特征点
                EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->pos_, K);// 边的实例
                edge->setId(index);
                edge->setVertex(0, vertex_pose);    // 设置顶点
                edge->setMeasurement(toVec2(current_frame_->features_left_[i]->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());  // 单位矩阵
                edge->setRobustKernel(new g2o::RobustKernelHuber);  // 设置核函数
                edges.push_back(edge);  // 添加边
                optimizer.addEdge(edge);
                index++;    // 计数
            }
        }

        // 估计位姿 estimate the Pose the determine the outliers
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration) {   // 迭代4次
            vertex_pose->setEstimate(current_frame_->Pose());   //设置初始值 当前帧的姿态
            optimizer.initializeOptimization(); // 初始化优化器
            optimizer.optimize(10); // 迭代 10 次
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) { // 遍历每一条边
                // auto e = edges[i];
                if (features[i]->is_outlier_) { // 是异常点
                    edges[i]->computeError();   // 算误差 更新 _error
                }
                if (edges[i]->chi2() > chi2_th) {// 大于阈值
                    features[i]->is_outlier_ = true;// 认为是异常点
                    edges[i]->setLevel(1);// 设置为 level1  可能是一个flag吧。。。
                    cnt_outlier++;  // 异常点计数
                } else {
                    features[i]->is_outlier_ = false;// 不认为是异常点
                    edges[i]->setLevel(0);// 设置为 level0
                };

                if (iteration == 2) {// 不知道干吗用的。。。
                    edges[i]->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/" << features.size() - cnt_outlier;
        
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

        for (auto &feat : features) {
            if (feat->is_outlier_) {
                feat->map_point_.reset();   // 将对象重置为 nullptr
                feat->is_outlier_ = false;  // maybe we can still use it in future
            }
        }
        return features.size() - cnt_outlier;   // 返回有效特征点数
    }

    int Frontend::TrackLastFrame() {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_left_) {// 遍历每一个指向特征点的指针
            if (kp->map_point_.lock()) {// 互斥锁有效
                // use project point
                auto mp = kp->map_point_.lock();    // 上锁，同时 mp 是 kp 的 map_point
                auto px = camera_left_->world2pixel(mp->pos_, current_frame_->Pose());// 2维向量，关联地图点的像素坐标
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            } else {    // 如果指针对象 过期 或 为空
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(   // 金字塔 光流法
            last_frame_->left_img_,
            current_frame_->left_img_,
            kps_last,
            kps_current,    // 输出
            status, // 对应下表 有flow置1，否则置0
            error,
            cv::Size(11, 11),
            3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW
        );

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {    // 对应位 为1 说明检测到光流
                cv::KeyPoint kp(kps_current[i], 7);// 第二个参数是维度
                Feature::Ptr feature(new Feature(current_frame_, kp));  // 参数：帧指针，特征点坐标
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;// 地标保存到feature
                current_frame_->features_left_.push_back(feature);// 保存feature
                num_good_pts++; // 计数
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }

    bool Frontend::StereoInit() {
        int num_features_left = DetectFeatures();
        int num_coor_features = FindFeaturesInRight();
        if (num_coor_features < num_features_init_) {   // 小于设定的阈值
            return false;
        }

        bool build_map_success = BuildInitMap();
        if (build_map_success) {
            status_ = FrontendStatus::TRACKING_GOOD;
            if (viewer_) {
                viewer_->AddCurrentFrame(current_frame_);
                viewer_->UpdateMap();
            }
            return true;
        }
        return false;
    }

    int Frontend::DetectFeatures() {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
        for (auto &feat : current_frame_->features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                        feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }// 在每一个特征点上画矩形（矩形边长20个像素）

        std::vector<cv::KeyPoint> keypoints;
        gftt_->detect(current_frame_->left_img_, keypoints, mask);// 在mask处找关键点
        int cnt_detected = 0;
        for (auto &kp : keypoints) {
            current_frame_->features_left_.push_back(Feature::Ptr(new Feature(current_frame_, kp)));// 把带有关键点信息的帧存入features_left_
            cnt_detected++; // 计数
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;
    }

    int Frontend::FindFeaturesInRight() {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_left, kps_right;
        for (auto &kp : current_frame_->features_left_) {
            kps_left.push_back(kp->position_.pt);
            auto mp = kp->map_point_.lock();
            if (mp) {
                // use projected points as initial guess
                auto px =
                    camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
                kps_right.push_back(cv::Point2f(px[0], px[1]));
            } else {
                // use same pixel in left iamge
                kps_right.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
            current_frame_->left_img_, current_frame_->right_img_, kps_left,
            kps_right, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                            0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_right[i], 7);
                Feature::Ptr feat(new Feature(current_frame_, kp));
                feat->is_on_left_image_ = false;
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;
            } else {
                current_frame_->features_right_.push_back(nullptr);
            }
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
    }

    bool Frontend::BuildInitMap() {
        std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
        size_t cnt_init_landmarks = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_right_[i] == nullptr) continue;
            // create map point from triangulation
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                        current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                        current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                cnt_init_landmarks++;
                map_->InsertMapPoint(new_map_point);
            }
        }
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        backend_->UpdateMap();

        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                << " map points";

        return true;
    }

    bool Frontend::Reset() {
        LOG(INFO) << "Reset is not implemented. ";
        return true;
    }

}  // namespace myslam