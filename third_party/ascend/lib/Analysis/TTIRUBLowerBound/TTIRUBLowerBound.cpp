#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::ascend::ub {

LogicalResult materializeDirectTensorLoads(
    ModuleOp module, MandatoryUBResourceGraph &graph,
    SmallVectorImpl<std::string> &unsupportedReasons);

namespace {

void addReason(SmallVectorImpl<std::string> &reasons, StringRef reason) {
  if (!llvm::is_contained(reasons, reason))
    reasons.push_back(reason.str());
}

bool hasKnownPipelineProfile(const TTIRUBAnalysisOptions &options) {
  const PipelineIdentity &identity = options.pipelineIdentity;
  // Task 2 deliberately ships no production profiles. The only V1 identity
  // accepted here is the explicit synthetic profile used with testing-only
  // contracts; arbitrary non-empty fingerprints must not become proof.
  return identity.openSourcePipeline == "synthetic-all-preserve" &&
         identity.relevantOptionsJson == "{}" &&
         identity.tritonVersion == "test" &&
         identity.cannVersionHash == "test" &&
         identity.sha256 == "synthetic-all-preserve-v1" &&
         !options.stages.empty() &&
         identity.targetArch == options.targetArch;
}

} // namespace

TTIRUBAnalysisResult
analyzeTTIRUBLowerBound(ModuleOp module, const TTIRUBAnalysisOptions &options,
                        const PipelineContractRegistry &registry) {
  TTIRUBAnalysisResult result;
  result.capacityBytes = getUBCapacityBytes(options.targetArch);
  if (!result.capacityBytes) {
    addReason(result.unsupportedReasons, "unknown-target");
    return result;
  }
  if (options.compileMode != "aiv") {
    addReason(result.unsupportedReasons, "unsupported-compile-mode");
    return result;
  }
  if (!hasKnownPipelineProfile(options)) {
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
