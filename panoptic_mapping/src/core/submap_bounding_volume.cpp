#include "panoptic_mapping/core/submap_bounding_volume.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "panoptic_mapping/core/submap.h"

namespace panoptic_mapping {

SubmapBoundingVolume::SubmapBoundingVolume(const Submap& submap)
    : submap_(&submap) {}

void SubmapBoundingVolume::update() {
  // A conservative approximation that computes the centroid from the
  // grid-aligned bounding box and then shrinks a sphere on it.

  // Prevent redundant updates.
  if (submap_->getTsdfLayer().getNumberOfAllocatedBlocks() ==
      num_previous_blocks_) {
    return;
  } else {
    num_previous_blocks_ = submap_->getTsdfLayer().getNumberOfAllocatedBlocks();
  }

  // Setup.
  voxblox::BlockIndexList block_indices;
  submap_->getTsdfLayer().getAllAllocatedBlocks(&block_indices);
  std::vector<Point> block_centers;
  block_centers.reserve(block_indices.size());
  Point min_dimension =
      Point(1, 1, 1) * std::numeric_limits<FloatingPoint>::max();
  Point max_dimension =
      Point(1, 1, 1) * std::numeric_limits<FloatingPoint>::min();
  const FloatingPoint grid_size = submap_->getTsdfLayer().block_size();

  // Get centers and grid aligned bounding box.
  for (const voxblox::BlockIndex& index : block_indices) {
    block_centers.emplace_back(
        voxblox::getCenterPointFromGridIndex(index, grid_size));
    min_dimension = min_dimension.cwiseMin(block_centers.back());
    max_dimension = max_dimension.cwiseMax(block_centers.back());
  }

  // Compute bounding sphere center and radius.
  center_ = (min_dimension + max_dimension) / 2.f;
  radius_ = 0.f;
  for (const Point& center : block_centers) {
    radius_ = std::max(radius_, (center - center_).norm());
  }
  radius_ += std::sqrt(3.f) * grid_size / 2.f;  // outermost voxel.
}

bool SubmapBoundingVolume::contains(const Point& point_S) const {
  // Point is expected in submap frame.
  return (center_ - point_S).norm() <= radius_;
}

bool SubmapBoundingVolume::intersects(const SubmapBoundingVolume& other) const {
  const Transformation T_S_other =
      submap_->getT_S_M() * other.submap_->getT_M_S();
  return (center_ - T_S_other * other.center_).norm() <=
         radius_ + other.radius_;
}

bool SubmapBoundingVolume::isInsidePlane(const Point& normal_S) const {
  // The normal is expected in submap frame.
  return center_.dot(normal_S) >= -radius_;
}

}  // namespace panoptic_mapping
