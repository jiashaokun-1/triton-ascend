#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/VerifierSafety.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <array>

namespace mlir::triton::ascend::ub {

namespace detail {

bool hasValidLoadOperandSegments(ArrayRef<int32_t> segments,
                                 unsigned numOperands) {
  if (segments.size() != 3 ||
      llvm::any_of(segments, [](int32_t size) { return size < 0; }))
    return false;
  if (segments[0] != 1 || segments[1] > 1 || segments[2] > 1 ||
      (segments[2] != 0 && segments[1] == 0))
    return false;
  int64_t total = static_cast<int64_t>(segments[0]) + segments[1] +
                  segments[2];
  return total == numOperands;
}

bool hasValidStoreOperandSegments(ArrayRef<int32_t> segments,
                                  unsigned numOperands) {
  if (segments.size() != 3 ||
      llvm::any_of(segments, [](int32_t size) { return size < 0; }))
    return false;
  if (segments[0] != 1 || segments[1] != 1 || segments[2] > 1)
    return false;
  int64_t total = static_cast<int64_t>(segments[0]) + segments[1] +
                  segments[2];
  return total == numOperands;
}

} // namespace detail

LogicalResult materializeDirectTensorLoads(
    ModuleOp module, MandatoryUBResourceGraph &graph,
    SmallVectorImpl<std::string> &unsupportedReasons);

namespace {

void addReason(SmallVectorImpl<std::string> &reasons, StringRef reason) {
  if (!llvm::is_contained(reasons, reason))
    reasons.push_back(reason.str());
}

bool isVerifierSafe(Operation *root) {
  SmallVector<Operation *> worklist{root};
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    StringRef name = operation->getName().getStringRef();
    if (name == "tt.make_range") {
      if (operation->getNumOperands() != 0 ||
          operation->getNumResults() != 1 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return false;
    } else if (name == "tt.splat") {
      if (operation->getNumOperands() != 1 ||
          operation->getNumResults() != 1 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return false;
    } else if (name == "tt.addptr") {
      if (operation->getNumOperands() != 2 ||
          operation->getNumResults() != 1 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return false;
    } else if (name == "tt.load") {
      if (!operation->getName().isRegistered() ||
          operation->getName().getTypeID() != TypeID::get<triton::LoadOp>() ||
          operation->getNumResults() != 1 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return false;
      OpaqueProperties storage = operation->getPropertiesStorage();
      if (!storage)
        return false;
      const auto *properties = storage.as<triton::LoadOp::Properties *>();
      if (!properties->boundaryCheck || !properties->cache ||
          !properties->evict || !properties->isVolatile ||
          !detail::hasValidLoadOperandSegments(
              properties->operandSegmentSizes,
              operation->getNumOperands()))
        return false;
    } else if (name == "tt.store") {
      if (!operation->getName().isRegistered() ||
          operation->getName().getTypeID() != TypeID::get<triton::StoreOp>() ||
          operation->getNumResults() != 0 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return false;
      OpaqueProperties storage = operation->getPropertiesStorage();
      if (!storage)
        return false;
      const auto *properties = storage.as<triton::StoreOp::Properties *>();
      if (!properties->boundaryCheck || !properties->cache ||
          !properties->evict)
        return false;
      // StoreOp has no AttrSizedOperandSegments property in this checkout;
      // its ODS groups are uniquely derived from the total operand count.
      int64_t maskSize = static_cast<int64_t>(operation->getNumOperands()) - 2;
      if (maskSize < 0 || maskSize > 1)
        return false;
      std::array<int32_t, 3> segments = {
          1, 1, static_cast<int32_t>(maskSize)};
      if (!detail::hasValidStoreOperandSegments(
              segments, operation->getNumOperands()))
        return false;
    }

    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          worklist.push_back(&nested);
  }
  return true;
}

bool hasKnownPipelineProfile(const TTIRUBAnalysisOptions &options,
                             const PipelineContractRegistry &registry) {
  const PipelineIdentity &identity = options.pipelineIdentity;
  // Task 2 deliberately ships no production profiles. The only V1 identity
  // accepted here is the explicit synthetic profile used with testing-only
  // contracts; arbitrary non-empty fingerprints must not become proof.
  return identity.openSourcePipeline == "synthetic-all-preserve" &&
         identity.relevantOptionsJson == "{}" &&
         identity.tritonVersion == "test" &&
         identity.cannVersionHash == "test" &&
         identity.sha256 == "synthetic-all-preserve-v1" &&
         identity.targetArch == options.targetArch &&
         options.stages.size() == 1 &&
         options.stages.front().stageName == "preserve" &&
         options.stages.front().options.empty() &&
         registry.hasExactlyOneMatchingContract(
             options.stages.front(), "fixed-disposition", "1");
}

} // namespace

TTIRUBAnalysisResult
analyzeTTIRUBLowerBound(ModuleOp module, const TTIRUBAnalysisOptions &options,
                        const PipelineContractRegistry &registry) {
  TTIRUBAnalysisResult result;
  if (!module || !isVerifierSafe(module.getOperation()) ||
      failed(verify(module.getOperation()))) {
    addReason(result.unsupportedReasons, "malformed-ir");
    return result;
  }
  result.capacityBytes = getUBCapacityBytes(options.targetArch);
  if (!result.capacityBytes) {
    addReason(result.unsupportedReasons, "unknown-target");
    return result;
  }
  if (options.compileMode != "aiv") {
    addReason(result.unsupportedReasons, "unsupported-compile-mode");
    return result;
  }
  if (!hasKnownPipelineProfile(options, registry)) {
    addReason(result.unsupportedReasons, "unknown-pipeline-profile");
    return result;
  }

  MandatoryUBResourceGraph graph;
  if (failed(materializeDirectTensorLoads(module, graph,
                                          result.unsupportedReasons)))
    return result;

  for (const PipelineStageContext &stage : options.stages) {
    if (failed(registry.applyOrInvalidateAll(graph, stage))) {
      addReason(result.unsupportedReasons, "pipeline-contract-internal-error");
      return result;
    }
  }

  for (const MandatoryUBResource &resource : graph.resources()) {
    if (resource.validity == ValidityState::Invalid)
      addReason(result.unsupportedReasons, resource.invalidReason);
  }
  if (!result.unsupportedReasons.empty())
    return result;

  FailureOr<LowerBoundCertificate> certificate =
      graph.solveSingletonLowerBound();
  if (failed(certificate)) {
    addReason(result.unsupportedReasons, "malformed-resource-graph");
    return result;
  }

  result.lowerBoundBytes = certificate->bytes;
  result.certificates.push_back(*certificate);
  if (result.lowerBoundBytes > *result.capacityBytes)
    result.decision = TTIRUBDecision::Reject;
  return result;
}

} // namespace mlir::triton::ascend::ub
