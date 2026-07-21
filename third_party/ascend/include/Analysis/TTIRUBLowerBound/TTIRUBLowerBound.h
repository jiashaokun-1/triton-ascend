#ifndef TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_TTIRUBLOWERBOUND_H
#define TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_TTIRUBLOWERBOUND_H

#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::triton::ascend::ub {

enum class TTIRUBDecision { Reject, Defer };

struct TTIRUBAnalysisOptions {
  std::string targetArch;
  std::string compileMode;
  PipelineIdentity pipelineIdentity;
  SmallVector<PipelineStageContext> stages;
};

struct TTIRUBAnalysisResult {
  TTIRUBDecision decision = TTIRUBDecision::Defer;
  int64_t lowerBoundBytes = 0;
  std::optional<int64_t> capacityBytes;
  SmallVector<LowerBoundCertificate> certificates;
  SmallVector<std::string> unsupportedReasons;
  std::string contractVersion = "ttir-ub-lb-v1";
};

TTIRUBAnalysisResult
analyzeTTIRUBLowerBound(ModuleOp module, const TTIRUBAnalysisOptions &options,
                        const PipelineContractRegistry &registry);

} // namespace mlir::triton::ascend::ub

#endif // TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_TTIRUBLOWERBOUND_H
