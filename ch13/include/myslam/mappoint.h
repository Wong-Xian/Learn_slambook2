#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {

    struct Frame;

    struct Feature;

    /**
     * 路标点类
     * 特征点在三角化之后形成路标点
     */
    struct MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0;  // ID
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();  // Position in world
        std::mutex data_mutex_;
        int observed_times_ = 0;  // being observed by feature matching algo.
        std::list<std::weak_ptr<Feature>> observations_;

        MapPoint() {}

        MapPoint(long id, Vec3 position);

        Vec3 Pos();

        void SetPos(const Vec3 &pos);

        void AddObservation(std::shared_ptr<Feature> feature);

        void RemoveObservation(std::shared_ptr<Feature> feat);

        std::list<std::weak_ptr<Feature>> GetObs();

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };
}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H
