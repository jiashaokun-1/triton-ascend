#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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

template <typename Range>
bool containsRelation(const Range &relations, ResourceId lhs,
                      ResourceId rhs) {
  const uint64_t expected = relationKey(lhs, rhs);
  return llvm::any_of(relations, [&](const auto &relation) {
    return relationKey(relation.first, relation.second) == expected;
  });
}

bool isOrderedSubsequence(ArrayRef<std::string> subsequence,
                          ArrayRef<std::string> sequence) {
  size_t next = 0;
  for (const std::string &item : sequence)
    if (next < subsequence.size() && item == subsequence[next])
      ++next;
  return next == subsequence.size();
}

void appendUnique(SmallVectorImpl<std::string> &destination,
                  ArrayRef<std::string> source) {
  for (const std::string &item : source)
    if (!llvm::is_contained(destination, item))
      destination.push_back(item);
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
  if (resources_[lhs].executionScope != resources_[rhs].executionScope) {
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
  if (!witness.resources.empty()) {
    const UBExecutionScope scope =
        resources_[witness.resources.front()].executionScope;
    if (llvm::any_of(witness.resources, [&](ResourceId id) {
          return resources_[id].executionScope != scope;
        })) {
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

LogicalResult MandatoryUBResourceGraph::alignResourcePayload(
    ResourceId id, int64_t alignmentBytes, StringRef contractId) {
  if (id >= resources_.size() || alignmentBytes <= 0 || contractId.empty()) {
    malformed_ = true;
    return failure();
  }
  const int64_t payload = resources_[id].minPayloadBytes;
  if (payload < 0 || payload > INT64_MAX - (alignmentBytes - 1)) {
    malformed_ = true;
    return failure();
  }
  const int64_t alignedPayload =
      ((payload + alignmentBytes - 1) / alignmentBytes) * alignmentBytes;
  resources_[id].minPayloadBytes = alignedPayload;
  resources_[id].contractTrace.push_back(contractId.str());
  return success();
}

LogicalResult MandatoryUBResourceGraph::raiseResourceInstances(
    ResourceId id, int64_t minInstances, StringRef contractId) {
  if (id >= resources_.size() || minInstances <= 0 ||
      minInstances < resources_[id].minInstances || contractId.empty()) {
    malformed_ = true;
    return failure();
  }
  resources_[id].minInstances = minInstances;
  resources_[id].contractTrace.push_back(contractId.str());
  return success();
}

LogicalResult MandatoryUBResourceGraph::updateResourceLowerBound(
    ResourceId id, int64_t minPayloadBytes, int64_t minInstances,
    StringRef contractId) {
  if (id >= resources_.size() || minPayloadBytes < 0 ||
      minPayloadBytes > resources_[id].minPayloadBytes ||
      minInstances <= 0 || minInstances < resources_[id].minInstances ||
      contractId.empty()) {
    malformed_ = true;
    return failure();
  }
  resources_[id].minPayloadBytes = minPayloadBytes;
  resources_[id].minInstances = minInstances;
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

LogicalResult MandatoryUBResourceGraph::refineWitnessToMustDistinct(
    WitnessId id, StringRef contractId) {
  if (contractId.empty() || !hasPairwiseMayAliasWitness(id)) {
    malformed_ = true;
    return failure();
  }
  const ArrayRef<ResourceId> resources = witnesses_[id].resources;
  for (size_t lhsIndex = 0; lhsIndex < resources.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < resources.size();
         ++rhsIndex) {
      ResourceId lhs = resources[lhsIndex];
      ResourceId rhs = resources[rhsIndex];
      const uint64_t key = relationKey(lhs, rhs);
      llvm::erase_if(mayAliases_, [&](const ResourcePair &relation) {
        return relationKey(relation.first, relation.second) == key;
      });
      if (!containsRelation(mustDistinct_, lhs, rhs))
        addMustDistinct(lhs, rhs);
    }
  }
  witnesses_[id].contractTrace.push_back(contractId.str());
  return success();
}

LogicalResult MandatoryUBResourceGraph::refineMayAliasToMustAlias(
    ResourceId lhs, ResourceId rhs, StringRef contractId) {
  if (contractId.empty() || lhs >= resources_.size() ||
      rhs >= resources_.size() || lhs == rhs ||
      !containsRelation(mayAliases_, lhs, rhs) ||
      containsRelation(mustAliases_, lhs, rhs) ||
      containsRelation(mustDistinct_, lhs, rhs)) {
    malformed_ = true;
    return failure();
  }
  const uint64_t key = relationKey(lhs, rhs);
  llvm::erase_if(mayAliases_, [&](const ResourcePair &relation) {
    return relationKey(relation.first, relation.second) == key;
  });
  addMustAlias(lhs, rhs);
  return success();
}

bool MandatoryUBResourceGraph::hasMayAlias(ResourceId lhs,
                                           ResourceId rhs) const {
  return !malformed_ && lhs < resources_.size() && rhs < resources_.size() &&
         lhs != rhs && containsRelation(mayAliases_, lhs, rhs) &&
         !containsRelation(mustAliases_, lhs, rhs) &&
         !containsRelation(mustDistinct_, lhs, rhs);
}

bool MandatoryUBResourceGraph::hasMustAlias(ResourceId lhs,
                                            ResourceId rhs) const {
  return !malformed_ && lhs < resources_.size() && rhs < resources_.size() &&
         lhs != rhs && containsRelation(mustAliases_, lhs, rhs) &&
         !containsRelation(mayAliases_, lhs, rhs) &&
         !containsRelation(mustDistinct_, lhs, rhs);
}

bool MandatoryUBResourceGraph::hasPairwiseMayAliasWitness(WitnessId id) const {
  if (malformed_ || id >= witnesses_.size() ||
      witnesses_[id].resources.size() < 2)
    return false;
  const ArrayRef<ResourceId> resources = witnesses_[id].resources;
  for (size_t lhsIndex = 0; lhsIndex < resources.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < resources.size();
         ++rhsIndex) {
      ResourceId lhs = resources[lhsIndex];
      ResourceId rhs = resources[rhsIndex];
      if (lhs >= resources_.size() || rhs >= resources_.size() || lhs == rhs ||
          !containsRelation(mayAliases_, lhs, rhs) ||
          containsRelation(mustAliases_, lhs, rhs) ||
          containsRelation(mustDistinct_, lhs, rhs))
        return false;
    }
  }
  return true;
}

bool MandatoryUBResourceGraph::hasPairwiseDistinctWitness(WitnessId id) const {
  if (malformed_ || id >= witnesses_.size() ||
      witnesses_[id].resources.size() < 2)
    return false;
  const ArrayRef<ResourceId> resources = witnesses_[id].resources;
  for (size_t lhsIndex = 0; lhsIndex < resources.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < resources.size();
         ++rhsIndex) {
      ResourceId lhs = resources[lhsIndex];
      ResourceId rhs = resources[rhsIndex];
      if (lhs >= resources_.size() || rhs >= resources_.size() || lhs == rhs ||
          containsRelation(mayAliases_, lhs, rhs) ||
          !containsRelation(mustDistinct_, lhs, rhs))
        return false;
    }
  }
  return true;
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
      if (resourceIds.empty())
        continue;
      const ArrayRef<std::string> exactTrace =
          resources_[resourceIds.front()].contractTrace;
      result.bytes = bytes;
      result.resourceIds = std::move(resourceIds);
      result.kind = "witness";
      if (llvm::all_of(result.resourceIds, [&](ResourceId id) {
            return ArrayRef<std::string>(resources_[id].contractTrace) ==
                   exactTrace;
          }) &&
          isOrderedSubsequence(witness.contractTrace, exactTrace)) {
        // Profile-backed multi-resource proofs carry the same exact stage
        // chain on every resource.  Preserve repeated contract IDs and their
        // order; they represent distinct pipeline stages.
        result.contractTrace.assign(exactTrace.begin(), exactTrace.end());
      } else {
        // Generic graph clients may combine independently-derived resources.
        // Retain their compact provenance representation.
        result.contractTrace.clear();
        appendUnique(result.contractTrace, witness.contractTrace);
        for (ResourceId id : result.resourceIds)
          appendUnique(result.contractTrace, resources_[id].contractTrace);
      }
    }
  }

  return result;
}

ArrayRef<MandatoryUBResource> MandatoryUBResourceGraph::resources() const {
  return resources_;
}

ArrayRef<CoexistenceWitness> MandatoryUBResourceGraph::witnesses() const {
  return witnesses_;
}

} // namespace mlir::triton::ascend::ub
