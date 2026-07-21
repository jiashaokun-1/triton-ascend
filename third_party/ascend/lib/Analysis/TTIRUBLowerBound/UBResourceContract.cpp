#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include <utility>

namespace mlir::triton::ascend::ub {
namespace {

void invalidateAll(MandatoryUBResourceGraph &graph, StringRef reason) {
  for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal) {
    if (graph.resources()[ordinal].validity != ValidityState::Valid)
      continue;
    graph.invalidate(static_cast<ResourceId>(ordinal), reason);
  }
}

class FixedTileContract final : public UBResourceContract {
public:
  FixedTileContract(StringRef stageName, int64_t maxTiles)
      : stageName(stageName.str()), maxTiles(maxTiles) {}

  StringRef id() const override { return "testing-fixed-tile"; }
  StringRef version() const override { return "1"; }
  bool matches(const PipelineStageContext &context) const override {
    return context.stageName == stageName;
  }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (maxTiles <= 0)
      return ContractDisposition::InternalError;
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal) {
      const MandatoryUBResource &resource = graph.resources()[ordinal];
      if (resource.validity != ValidityState::Valid ||
          resource.minPayloadBytes < 0)
        continue;
      int64_t transformed = resource.minPayloadBytes / maxTiles;
      if (resource.minPayloadBytes % maxTiles != 0)
        ++transformed;
      if (failed(graph.lowerResourcePayload(static_cast<ResourceId>(ordinal),
                                            transformed, id())))
        return ContractDisposition::InternalError;
    }
    return ContractDisposition::Transform;
  }

private:
  std::string stageName;
  int64_t maxTiles;
};

} // namespace

void PipelineContractRegistry::addForTesting(
    std::unique_ptr<UBResourceContract> contract) {
  contracts.push_back(std::move(contract));
}

bool PipelineContractRegistry::hasExactlyOneMatchingContract(
    const PipelineStageContext &context, StringRef expectedId,
    StringRef expectedVersion) const {
  if (contracts.size() != 1 || !contracts.front())
    return false;
  const UBResourceContract &contract = *contracts.front();
  return contract.matches(context) && contract.id() == expectedId &&
         contract.version() == expectedVersion;
}

LogicalResult PipelineContractRegistry::applyOrInvalidateAll(
    MandatoryUBResourceGraph &graph,
    const PipelineStageContext &context) const {
  const UBResourceContract *matchedContract = nullptr;
  for (const auto &contract : contracts) {
    if (!contract->matches(context))
      continue;
    if (matchedContract)
      return failure();
    matchedContract = contract.get();
  }

  if (!matchedContract) {
    invalidateAll(graph, "unmatched-pipeline-stage");
    return success();
  }

  switch (matchedContract->apply(graph, context)) {
  case ContractDisposition::Preserve:
  case ContractDisposition::Transform:
    return success();
  case ContractDisposition::Invalidate:
    invalidateAll(graph, matchedContract->id());
    return success();
  case ContractDisposition::InternalError:
    return failure();
  }
  return failure();
}

std::unique_ptr<UBResourceContract> makeFixedTileContract(StringRef stageName,
                                                          int64_t maxTiles) {
  return std::make_unique<FixedTileContract>(stageName, maxTiles);
}

std::optional<int64_t> getUBCapacityBytes(StringRef targetArch) {
  if (targetArch.starts_with("Ascend910B") ||
      targetArch.starts_with("Ascend910_93"))
    return 192 * 1024;
  if (targetArch.starts_with("Ascend910_95") ||
      targetArch.starts_with("Ascend950"))
    return 256 * 1024;
  return std::nullopt;
}

} // namespace mlir::triton::ascend::ub
