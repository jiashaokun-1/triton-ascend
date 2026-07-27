#ifndef TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_MANDATORYUBRESOURCEGRAPH_H
#define TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_MANDATORYUBRESOURCEGRAPH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

namespace mlir::triton::ascend::ub {

using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;

using ResourceId = uint32_t;
using WitnessId = uint32_t;

inline constexpr ResourceId InvalidResourceId =
    std::numeric_limits<ResourceId>::max();
inline constexpr WitnessId InvalidWitnessId =
    std::numeric_limits<WitnessId>::max();

enum class ValidityState { Valid, Invalid };
enum class UBAddressSpace { UB };
// A UB allocation belongs to one physical execution domain.  AIC and AIV
// cores have independent UB instances, so resources in different domains
// must never be added into one coexistence certificate.
enum class UBExecutionScope { SingleCore, AIC, AIV };
enum class MaterializationKind {
  GMToUBLoad,
  ViewAlias,
  ReductionScratch,
  ReductionAccumulator,
  DynamicCVFixpipeOutput,
  DynamicCVVectorOutput
};

struct ProgramPoint {
  uint64_t ordinal = 0;
};

struct MandatoryUBResource {
  std::string debugName;
  int64_t minPayloadBytes;
  int64_t minInstances;
  std::string origin;
  UBAddressSpace addressSpace = UBAddressSpace::UB;
  UBExecutionScope executionScope = UBExecutionScope::SingleCore;
  MaterializationKind kind = MaterializationKind::GMToUBLoad;
  ProgramPoint birth;
  ProgramPoint lastRequiredUse;
  ValidityState validity = ValidityState::Valid;
  SmallVector<std::string> contractTrace;
  std::string invalidReason;
  int64_t sourceElements = -1;
  unsigned elementBitWidth = 0;
  std::string consumer;
};

struct CoexistenceWitness {
  SmallVector<ResourceId> resources;
  SmallVector<std::string> contractTrace;
};

struct LowerBoundCertificate {
  int64_t bytes = 0;
  SmallVector<ResourceId> resourceIds;
  std::string kind;
  SmallVector<std::string> contractTrace;
};

struct StableIdLimits {
  uint64_t resourceCapacity = InvalidResourceId;
  uint64_t witnessCapacity = InvalidWitnessId;
};

class MandatoryUBResourceGraph {
public:
  MandatoryUBResourceGraph() = default;
  explicit MandatoryUBResourceGraph(StableIdLimits idLimits);

  ResourceId addResource(MandatoryUBResource resource);
  void addMayAlias(ResourceId lhs, ResourceId rhs);
  void addMustAlias(ResourceId lhs, ResourceId rhs);
  void addMustDistinct(ResourceId lhs, ResourceId rhs);
  WitnessId addWitness(CoexistenceWitness witness);
  WitnessId addWitness(std::initializer_list<ResourceId> resources);
  void invalidate(ResourceId id, StringRef reason);
  LogicalResult lowerResourcePayload(ResourceId id, int64_t minPayloadBytes,
                                     StringRef contractId);
  LogicalResult alignResourcePayload(ResourceId id, int64_t alignmentBytes,
                                     StringRef contractId);
  LogicalResult raiseResourceInstances(ResourceId id, int64_t minInstances,
                                       StringRef contractId);
  LogicalResult updateResourceLowerBound(ResourceId id,
                                         int64_t minPayloadBytes,
                                         int64_t minInstances,
                                         StringRef contractId);
  LogicalResult appendResourceTrace(ResourceId id, StringRef contractId);
  LogicalResult refineWitnessToMustDistinct(WitnessId id,
                                            StringRef contractId);
  LogicalResult refineMayAliasToMustAlias(ResourceId lhs, ResourceId rhs,
                                          StringRef contractId);
  bool hasMayAlias(ResourceId lhs, ResourceId rhs) const;
  bool hasMustAlias(ResourceId lhs, ResourceId rhs) const;
  bool hasPairwiseMayAliasWitness(WitnessId id) const;
  bool hasPairwiseDistinctWitness(WitnessId id) const;
  FailureOr<LowerBoundCertificate> solveSingletonLowerBound() const;
  FailureOr<LowerBoundCertificate> solveWitnessLowerBound() const;
  ArrayRef<MandatoryUBResource> resources() const;
  ArrayRef<CoexistenceWitness> witnesses() const;

private:
  using ResourcePair = std::pair<ResourceId, ResourceId>;

  void addRelation(SmallVectorImpl<ResourcePair> &relations, ResourceId lhs,
                   ResourceId rhs);

  SmallVector<MandatoryUBResource> resources_;
  SmallVector<CoexistenceWitness> witnesses_;
  SmallVector<ResourcePair> mayAliases_;
  SmallVector<ResourcePair> mustAliases_;
  SmallVector<ResourcePair> mustDistinct_;
  StableIdLimits idLimits_;
  bool malformed_ = false;
};

} // namespace mlir::triton::ascend::ub

#endif // TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_MANDATORYUBRESOURCEGRAPH_H
