#ifndef TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_UBRESOURCECONTRACT_H
#define TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_UBRESOURCECONTRACT_H

#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"

#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::triton::ascend::ub {

using llvm::StringMap;

enum class ContractDisposition { Preserve, Transform, Invalidate, InternalError };

struct PipelineIdentity {
  std::string openSourcePipeline;
  std::string relevantOptionsJson;
  std::string targetArch;
  std::string tritonVersion;
  std::string cannVersionHash;
  std::string sha256;
};

struct PipelineStageContext {
  std::string stageName;
  StringMap<std::string> options;
};

class UBResourceContract {
public:
  virtual ~UBResourceContract() = default;
  virtual StringRef id() const = 0;
  virtual StringRef version() const = 0;
  virtual bool matches(const PipelineStageContext &context) const = 0;
  virtual ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &context) const = 0;
};

class PipelineContractRegistry {
public:
  void addForTesting(std::unique_ptr<UBResourceContract> contract);
  LogicalResult
  applyOrInvalidateAll(MandatoryUBResourceGraph &graph,
                       const PipelineStageContext &context) const;

private:
  std::vector<std::unique_ptr<UBResourceContract>> contracts;
};

std::unique_ptr<UBResourceContract> makeFixedTileContract(StringRef stageName,
                                                          int64_t maxTiles);

std::optional<int64_t> getUBCapacityBytes(StringRef targetArch);

} // namespace mlir::triton::ascend::ub

#endif // TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_UBRESOURCECONTRACT_H
