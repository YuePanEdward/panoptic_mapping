#ifndef PANOPTIC_MAPPING_REGISTRATION_TSDF_REGISTRATOR_H_
#define PANOPTIC_MAPPING_REGISTRATION_TSDF_REGISTRATOR_H_

#include <voxblox/interpolator/interpolator.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/core/common.h"
#include "panoptic_mapping/core/submap.h"
#include "panoptic_mapping/core/submap_collection.h"

namespace panoptic_mapping {

/**
 * This class directly compares TSDF volumes of two submaps with each other,
 * using the registration constraints adapted from voxgraph to detect matches
 * and solve for transformations using ceres.
 * Due to the current implementation this is also the de-facto change detection
 * module.
 */

class TsdfRegistrator {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;
    float min_voxel_weight = 1e-6;
    float error_threshold = -1;  // m, negative values are multiples of
    // the voxel_size.
    int match_rejection_points = 50;
    float match_rejection_percentage = 0.1;
    int match_acceptance_points = 50;
    float match_acceptance_percentage = 0.1;

    Config() { setConfigName("TsdfRegistrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  explicit TsdfRegistrator(const Config& config);
  virtual ~TsdfRegistrator() = default;

  // Check whether there is significant difference between the two submaps.
  void computeIsoSurfacePoints(Submap* submap) const;
  void checkSubmapCollectionForChange(const SubmapCollection& submaps) const;
  bool submapsConflict(const Submap& reference, const Submap& other,
                       unsigned int* matching_points = nullptr) const;

 private:
  const Config config_;

  // Methods.
  bool getDistanceAtPoint(
      float* distance, const IsoSurfacePoint& point,
      const Transformation& T_P_S,
      const voxblox::Interpolator<TsdfVoxel>& interpolator) const;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_REGISTRATION_TSDF_REGISTRATOR_H_