#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <limits>
#include <numeric>

namespace mlir::triton::ascend::ub {
namespace {

FailureOr<int64_t> checkedResourceBytes(const MandatoryUBResource &resource) {
  if (resource.minPayloadBytes < 0 || resource.minInstances < 0)
    return failure();
  if (resource.minPayloadBytes != 0 &&
      resource.minInstances > INT64_MAX / resource.minPayloadBytes)
    return failure();
  return resource.minPayloadBytes * resource.minInstances;
}

LogicalResult checkedAdd(int64_t &total, int64_t value) {
  assert(total >= 0 && value >= 0);
  if (value > INT64_MAX - total)
    return failure();
  total += value;
  return success();
}

class DisjointSet {
public:
  explicit DisjointSet(size_t size) : parent(size) {
    std::iota(parent.begin(), parent.end(), ResourceId{0});
  }

  ResourceId find(ResourceId id) {
    ResourceId root = id;
    while (parent[root] != root)
      root = parent[root];
    while (parent[id] != id) {
      ResourceId next = parent[id];
      parent[id] = root;
      id = next;
    }
    return root;
  }

  void unite(ResourceId lhs, ResourceId rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    parent[rhs] = lhs;
  }

private:
  SmallVector<ResourceId> parent;
};

uint64_t relationKey(ResourceId lhs, ResourceId rhs) {
  if (rhs < lhs)
    std::swap(lhs, rhs);
  return (static_cast<uint64_t>(lhs) << 32) | rhs;
}

} // namespace

ResourceId
MandatoryUBResourceGraph::addResource(MandatoryUBResource resource) {
  assert(resources_.size() <= std::numeric_limits<ResourceId>::max());
  ResourceId id = static_cast<ResourceId>(resources_.size());
  resources_.push_back(std::move(resource));
  return id;
}

void MandatoryUBResourceGraph::addRelation(
    SmallVectorImpl<ResourcePair> &relations, ResourceId lhs, ResourceId rhs) {
  if (lhs >= resources_.size() || rhs >= resources_.size()) {
    malformed_ = true;
    return;
  }
  if (rhs < lhs)
    std::swap(lhs, rhs);
  relations.emplace_back(lhs, rhs);
}

void MandatoryUBResourceGraph::addMayAlias(ResourceId lhs, ResourceId rhs) {
  addRelation(mayAliases_, lhs, rhs);
}

void MandatoryUBResourceGraph::addMustAlias(ResourceId lhs, ResourceId rhs) {
  addRelation(mustAliases_, lhs, rhs);
}

void MandatoryUBResourceGraph::addMustDistinct(ResourceId lhs, ResourceId rhs) {
  addRelation(mustDistinct_, lhs, rhs);
}

WitnessId MandatoryUBResourceGraph::addWitness(CoexistenceWitness witness) {
  assert(witnesses_.size() <= std::numeric_limits<WitnessId>::max());
  for (ResourceId id : witness.resources) {
    if (id >= resources_.size()) {
      malformed_ = true;
      break;
    }
  }
  WitnessId id = static_cast<WitnessId>(witnesses_.size());
  witnesses_.push_back(std::move(witness));
  return id;
}

WitnessId MandatoryUBResourceGraph::addWitness(
    std::initializer_list<ResourceId> resources) {
  CoexistenceWitness witness;
  witness.resources.append(resources.begin(), resources.end());
  return addWitness(std::move(witness));
}

void MandatoryUBResourceGraph::invalidate(ResourceId id, StringRef reason) {
  if (id >= resources_.size()) {
    malformed_ = true;
    return;
  }
  resources_[id].validity = ValidityState::Invalid;
  resources_[id].invalidReason = reason.str();
}

FailureOr<LowerBoundCertificate>
MandatoryUBResourceGraph::solveSingletonLowerBound() const {
  if (malformed_)
    return failure();

  LowerBoundCertificate result;
  result.kind = "singleton";
  for (ResourceId id = 0; id < resources_.size(); ++id) {
    const MandatoryUBResource &resource = resources_[id];
    if (resource.validity != ValidityState::Valid)
      continue;
    FailureOr<int64_t> bytes = checkedResourceBytes(resource);
    if (failed(bytes))
      return failure();
    if (*bytes > result.bytes) {
      result.bytes = *bytes;
      result.resourceIds.assign({id});
    }
  }
  return result;
}

FailureOr<LowerBoundCertificate>
MandatoryUBResourceGraph::solveWitnessLowerBound() const {
  FailureOr<LowerBoundCertificate> singleton = solveSingletonLowerBound();
  if (failed(singleton))
    return failure();

  LowerBoundCertificate result = std::move(*singleton);
  DisjointSet aliases(resources_.size());
  for (auto [lhs, rhs] : mustAliases_)
    aliases.unite(lhs, rhs);

  llvm::DenseSet<uint64_t> mayAliasClasses;
  for (auto [lhs, rhs] : mayAliases_) {
    lhs = aliases.find(lhs);
    rhs = aliases.find(rhs);
    if (lhs != rhs)
      mayAliasClasses.insert(relationKey(lhs, rhs));
  }

  llvm::DenseSet<uint64_t> mustDistinctClasses;
  for (auto [lhs, rhs] : mustDistinct_) {
    lhs = aliases.find(lhs);
    rhs = aliases.find(rhs);
    if (lhs == rhs)
      return failure();
    uint64_t key = relationKey(lhs, rhs);
    if (mayAliasClasses.contains(key))
      return failure();
    mustDistinctClasses.insert(key);
  }

  SmallVector<int64_t> classBytes(resources_.size(), 0);
  SmallVector<ResourceId> classResource(resources_.size(), 0);
  SmallVector<bool> classHasResource(resources_.size(), false);
  for (ResourceId id = 0; id < resources_.size(); ++id) {
    const MandatoryUBResource &resource = resources_[id];
    if (resource.validity != ValidityState::Valid)
      continue;
    FailureOr<int64_t> bytes = checkedResourceBytes(resource);
    if (failed(bytes))
      return failure();
    ResourceId root = aliases.find(id);
    if (!classHasResource[root] || *bytes > classBytes[root]) {
      classHasResource[root] = true;
      classBytes[root] = *bytes;
      classResource[root] = id;
    }
  }

  for (const CoexistenceWitness &witness : witnesses_) {
    SmallVector<ResourceId> classes;
    for (ResourceId id : witness.resources) {
      if (resources_[id].validity != ValidityState::Valid)
        continue;
      ResourceId root = aliases.find(id);
      if (std::find(classes.begin(), classes.end(), root) == classes.end())
        classes.push_back(root);
    }

    bool allMustDistinct = true;
    for (size_t lhsIndex = 0; lhsIndex < classes.size(); ++lhsIndex) {
      for (size_t rhsIndex = lhsIndex + 1; rhsIndex < classes.size();
           ++rhsIndex) {
        uint64_t key = relationKey(classes[lhsIndex], classes[rhsIndex]);
        if (!mustDistinctClasses.contains(key) ||
            mayAliasClasses.contains(key)) {
          allMustDistinct = false;
          break;
        }
      }
      if (!allMustDistinct)
        break;
    }
    if (!allMustDistinct)
      continue;

    int64_t bytes = 0;
    SmallVector<ResourceId> resourceIds;
    for (ResourceId root : classes) {
      if (!classHasResource[root])
        continue;
      if (failed(checkedAdd(bytes, classBytes[root])))
        return failure();
      resourceIds.push_back(classResource[root]);
    }
    if (bytes > result.bytes) {
      result.bytes = bytes;
      result.resourceIds = std::move(resourceIds);
      result.kind = "witness";
    }
  }

  return result;
}

ArrayRef<MandatoryUBResource> MandatoryUBResourceGraph::resources() const {
  return resources_;
}

} // namespace mlir::triton::ascend::ub
