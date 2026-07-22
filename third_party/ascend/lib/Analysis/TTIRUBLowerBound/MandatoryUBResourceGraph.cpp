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

template <typename Id> FailureOr<Id> checkedId(size_t ordinal) {
  // The maximum value is reserved as an invalid sentinel and is never issued.
  if (ordinal >= std::numeric_limits<Id>::max())
    return failure();
  return static_cast<Id>(ordinal);
}

} // namespace

MandatoryUBResourceGraph::MandatoryUBResourceGraph(StableIdLimits idLimits)
    : idLimits_(idLimits) {}

ResourceId
MandatoryUBResourceGraph::addResource(MandatoryUBResource resource) {
  FailureOr<ResourceId> id = checkedId<ResourceId>(resources_.size());
  if (failed(id) || resources_.size() >= idLimits_.resourceCapacity) {
    malformed_ = true;
    return InvalidResourceId;
  }
  resources_.push_back(std::move(resource));
  return *id;
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

WitnessId
MandatoryUBResourceGraph::addWitness(CoexistenceWitness witness) {
  FailureOr<WitnessId> witnessId = checkedId<WitnessId>(witnesses_.size());
  if (failed(witnessId) || witnesses_.size() >= idLimits_.witnessCapacity) {
    malformed_ = true;
    return InvalidWitnessId;
  }
  for (ResourceId id : witness.resources) {
    if (id >= resources_.size()) {
      malformed_ = true;
      return InvalidWitnessId;
    }
  }
  witnesses_.push_back(std::move(witness));
  return *witnessId;
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

LogicalResult MandatoryUBResourceGraph::lowerResourcePayload(
    ResourceId id, int64_t minPayloadBytes, StringRef contractId) {
  if (id >= resources_.size() || minPayloadBytes < 0 ||
      minPayloadBytes > resources_[id].minPayloadBytes) {
    malformed_ = true;
    return failure();
  }
  resources_[id].minPayloadBytes = minPayloadBytes;
  resources_[id].contractTrace.push_back(contractId.str());
  return success();
}

LogicalResult MandatoryUBResourceGraph::appendResourceTrace(
    ResourceId id, StringRef contractId) {
  if (id >= resources_.size() || contractId.empty()) {
    malformed_ = true;
    return failure();
  }
  resources_[id].contractTrace.push_back(contractId.str());
  return success();
}

FailureOr<LowerBoundCertificate>
MandatoryUBResourceGraph::solveSingletonLowerBound() const {
  if (malformed_)
    return failure();

  LowerBoundCertificate result;
  result.kind = "singleton";
  for (size_t ordinal = 0; ordinal < resources_.size(); ++ordinal) {
    FailureOr<ResourceId> id = checkedId<ResourceId>(ordinal);
    if (failed(id))
      return failure();
    const MandatoryUBResource &resource = resources_[ordinal];
    if (resource.validity != ValidityState::Valid)
      continue;
    FailureOr<int64_t> bytes = checkedResourceBytes(resource);
    if (failed(bytes))
      return failure();
    if (*bytes > result.bytes) {
      result.bytes = *bytes;
      result.resourceIds.assign({*id});
      result.contractTrace = resource.contractTrace;
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
  for (size_t ordinal = 0; ordinal < resources_.size(); ++ordinal) {
    FailureOr<ResourceId> id = checkedId<ResourceId>(ordinal);
    if (failed(id))
      return failure();
    const MandatoryUBResource &resource = resources_[ordinal];
    if (resource.validity != ValidityState::Valid)
      continue;
    FailureOr<int64_t> bytes = checkedResourceBytes(resource);
    if (failed(bytes))
      return failure();
    ResourceId root = aliases.find(*id);
    if (!classHasResource[root] || *bytes > classBytes[root]) {
      classHasResource[root] = true;
      classBytes[root] = *bytes;
      classResource[root] = *id;
    }
  }

  for (const CoexistenceWitness &witness : witnesses_) {
    if (std::any_of(witness.resources.begin(), witness.resources.end(),
                    [&](ResourceId id) {
                      return resources_[id].validity != ValidityState::Valid;
                    }))
      continue;

    SmallVector<ResourceId> classes;
    for (ResourceId id : witness.resources) {
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
      result.contractTrace = witness.contractTrace;
      for (ResourceId id : result.resourceIds)
        result.contractTrace.append(resources_[id].contractTrace.begin(),
                                    resources_[id].contractTrace.end());
    }
  }

  return result;
}

ArrayRef<MandatoryUBResource> MandatoryUBResourceGraph::resources() const {
  return resources_;
}

} // namespace mlir::triton::ascend::ub
