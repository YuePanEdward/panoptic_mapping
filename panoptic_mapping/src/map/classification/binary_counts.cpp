#include "panoptic_mapping/map/classification/binary_counts.h"

#include <climits>
#include <memory>
#include <vector>

namespace panoptic_mapping {

ClassVoxelType BinaryCountVoxel::getVoxelType() const {
  return ClassVoxelType::kBinaryCounts;
}

bool BinaryCountVoxel::isObserverd() const {
  return belongs_count > 0 || foreign_count > 0;
}

bool BinaryCountVoxel::belongsToSubmap() const {
  // In doubt we count the voxel as belonging. This also applies for unobserved
  // voxels.
  return belongs_count >= foreign_count;
}

float BinaryCountVoxel::getBelongingProbability() const {
  return static_cast<float>(belongs_count) /
         static_cast<float>(belongs_count + foreign_count);
}

int BinaryCountVoxel::getBelongingID() const {
  // 0 - belongin submap, 1 - else
  return foreign_count < belongs_count;
}

float BinaryCountVoxel::getProbability(const int id) const {
  return static_cast<float>(id == 0 ? belongs_count : foreign_count) /
         static_cast<float>(belongs_count + foreign_count);
}

void BinaryCountVoxel::incrementCount(const int id, const float weight) {
  // ID 0 is used for belonging voxels.
  if (id == 0) {
    belongs_count++;
  } else {
    foreign_count++;
  }
}

void BinaryCountVoxel::serializeVoxelToInt(std::vector<uint32_t>* data) const {}

void BinaryCountVoxel::deseriliazeVoxelFromInt(
    const std::vector<uint32_t>& data, size_t& data_index) {}

config_utilities::Factory::RegistrationRos<ClassLayer, BinaryCountLayer, float,
                                           size_t>
    BinaryCountLayer::registration_("binary_counts");

BinaryCountLayer::BinaryCountLayer(const Config& config, const float voxel_size,
                                   const size_t voxels_per_side)
    : config_(config.checkValid()),
      ClassLayerImpl(voxel_size, voxels_per_side) {}

ClassVoxelType BinaryCountLayer::getVoxelType() const {
  return ClassVoxelType::kBinaryCounts;
}

std::unique_ptr<ClassLayer> BinaryCountLayer::clone() const {
  return std::make_unique<BinaryCountLayer>(*this);
}

}  // namespace panoptic_mapping