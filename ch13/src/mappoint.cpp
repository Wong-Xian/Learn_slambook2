#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam {

    MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

    Vec3 MapPoint::Pos() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void MapPoint::SetPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    void MapPoint::AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end();
            iter++) {
            if (iter->lock() == feat) {
                observations_.erase(iter);
                feat->map_point_.reset();
                observed_times_--;
                break;
            }
        }
    }

    std::list<std::weak_ptr<Feature>> MapPoint::GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    MapPoint::Ptr MapPoint::CreateNewMappoint() {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

}  // namespace myslam
