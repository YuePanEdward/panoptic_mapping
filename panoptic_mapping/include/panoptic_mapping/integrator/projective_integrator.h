#ifndef PANOPTIC_MAPPING_INTEGRATOR_PROJECTIVE_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATOR_PROJECTIVE_INTEGRATOR_H_

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/core/common.h"
#include "panoptic_mapping/integrator/integrator_base.h"
#include "panoptic_mapping/integrator/projection_interpolators.h"

namespace panoptic_mapping {

/**
 * Allocate blocks based on the image and project all visible blocks into the
 * image for updates.
 */
class ProjectiveIntegrator : public IntegratorBase {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    // camera settings  [px]
    int width = 640;
    int height = 480;
    float vx = 320;
    float vy = 240;
    float focal_length = 320;

    // integration params
    float max_range = 5;    // m
    float min_range = 0.1;  // m
    bool use_weight_dropoff = true;
    bool use_constant_weight = false;
    bool foreign_rays_clear = true;  // observations of object B can clear
    // spcae in object A
    float sparsity_compensation_factor = 1.0;
    float max_weight = 1e5;
    std::string interpolation_method;  // nearest, bilinear, adaptive, semantic

    // system params
    int integration_threads = std::thread::hardware_concurrency();

    Config() { setConfigName("ProjectiveIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  explicit ProjectiveIntegrator(const Config& config);
  ~ProjectiveIntegrator() override = default;

  void processImages(SubmapCollection* submaps, const Transformation& T_M_C,
                     const cv::Mat& depth_image, const cv::Mat& color_image,
                     const cv::Mat& id_image) override;

 protected:
  class ThreadSafeIndexGetter {
   public:
    explicit ThreadSafeIndexGetter(std::vector<int> indices);
    bool getNextIndex(int* index);

   private:
    std::mutex mutex_;
    std::vector<int> indices_;
    size_t current_index_;
  };

  // Components.
  const Config config_;
  std::vector<std::unique_ptr<InterpolatorBase>>
      interpolators_;  // one for each thread.

  // Cached data.
  Eigen::MatrixXf range_image_;
  float max_range_in_image_;

  // Precomputed stored values.
  std::vector<Point> view_frustum_;  // top, right, bottom, left plane normals

  // Methods.
  void allocateNewBlocks(SubmapCollection* submaps, const Transformation& T_M_C,
                         const cv::Mat& depth_image, const cv::Mat& id_image);

  bool submapIsInViewFrustum(const Submap& submap,
                             const Transformation& T_M_C) const;

  bool blockIsInViewFrustum(const Point& center_point_C,
                            float block_diag_half) const;

  void findVisibleBlocks(const Submap& submap, const Transformation& T_M_C,
                         voxblox::BlockIndexList* block_list) const;

  void updateSubmap(Submap* submap, InterpolatorBase* interpolator,
                    const voxblox::BlockIndexList& block_indices,
                    const Transformation& T_M_C, const cv::Mat& color_image,
                    const cv::Mat& id_image) const;

  void updateTsdfBlock(Submap* submap, InterpolatorBase* interpolator,
                       const voxblox::BlockIndex& block_index,
                       const Transformation& T_M_C, const cv::Mat& color_image,
                       const cv::Mat& id_image) const;

  bool computeVoxelDistanceAndWeight(
      float* sdf, float* weight, bool* point_belongs_to_this_submap,
      InterpolatorBase* interpolator, const Point& p_C,
      const cv::Mat& color_image, const cv::Mat& id_image, int submap_id,
      float truncation_distance, float voxel_size,
      bool is_free_space_submap) const;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATOR_PROJECTIVE_INTEGRATOR_H_
