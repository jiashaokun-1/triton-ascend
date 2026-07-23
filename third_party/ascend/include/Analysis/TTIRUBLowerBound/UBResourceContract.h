#ifndef TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_UBRESOURCECONTRACT_H
#define TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_UBRESOURCECONTRACT_H

#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"

#include "llvm/ADT/ArrayRef.h"
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
  std::string canonicalTtirSha256;
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

struct PipelineContractBinding {
  PipelineStageContext stage;
  std::string contractId;
  std::string contractVersion;
  StringMap<std::string> parameters;
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
  void setProfileIdentity(PipelineIdentity identity);
  LogicalResult addProfileContract(
      PipelineContractBinding binding,
      std::unique_ptr<UBResourceContract> contract);
  bool matchesProfile(const PipelineIdentity &identity,
                      ArrayRef<PipelineStageContext> stages) const;
  LogicalResult applyProfileStage(MandatoryUBResourceGraph &graph,
                                  const PipelineStageContext &context,
                                  size_t stageOrdinal) const;

  void addForTesting(std::unique_ptr<UBResourceContract> contract);
  bool hasExactlyOneMatchingContract(const PipelineStageContext &context,
                                     StringRef expectedId,
                                     StringRef expectedVersion) const;
  LogicalResult
  applyOrInvalidateAll(MandatoryUBResourceGraph &graph,
                       const PipelineStageContext &context) const;

private:
  struct ProfileContractEntry {
    PipelineContractBinding binding;
    std::unique_ptr<UBResourceContract> contract;
  };

  std::optional<PipelineIdentity> profileIdentity;
  std::vector<ProfileContractEntry> profileContracts;
  std::vector<std::unique_ptr<UBResourceContract>> contracts;
};

std::unique_ptr<UBResourceContract> makeFixedTileContract(StringRef stageName,
                                                          int64_t maxTiles);
std::unique_ptr<UBResourceContract>
makeInvalidateContract(const PipelineStageContext &stage);
std::unique_ptr<UBResourceContract> makeDirectCopyPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes);
std::unique_ptr<UBResourceContract> makeDirectCopyMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles);
std::unique_ptr<UBResourceContract> makeBinaryAddPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes);
std::unique_ptr<UBResourceContract> makeBinaryAddMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles);
std::unique_ptr<UBResourceContract> makeReshapeCopyPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes);
std::unique_ptr<UBResourceContract> makeReshapeCopyMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles);
std::unique_ptr<UBResourceContract> makeReductionSumPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedScratchPayloadBytes,
    int64_t expectedAccumulatorPayloadBytes);
std::unique_ptr<UBResourceContract> makeReductionSumMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedScratchPayloadBytes,
    int64_t expectedAccumulatorPayloadBytes, int64_t maxTiles);
std::unique_ptr<UBResourceContract> makeReductionSumExtraBufferContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedScratchPayloadBytes,
    int64_t expectedAccumulatorPayloadBytes);

std::optional<int64_t> getUBCapacityBytes(StringRef targetArch);

} // namespace mlir::triton::ascend::ub

#endif // TRITON_ASCEND_ANALYSIS_TTIRUBLOWERBOUND_UBRESOURCECONTRACT_H
