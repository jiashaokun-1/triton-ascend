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
  static const llvm::StringMap<int64_t> capacities = {
      {"Ascend910B", 192 * 1024},
      {"Ascend910_93", 192 * 1024},
      {"Ascend910B1", 192 * 1024},
      {"Ascend910B2", 192 * 1024},
      {"Ascend910B3", 192 * 1024},
      {"Ascend910B4", 192 * 1024},
      {"Ascend910_9362", 192 * 1024}, {"Ascend910_9372", 192 * 1024},
      {"Ascend910_9381", 192 * 1024}, {"Ascend910_9382", 192 * 1024},
      {"Ascend910_9391", 192 * 1024}, {"Ascend910_9392", 192 * 1024},
      {"Ascend310B1", 248 * 1024},
      {"Ascend310B2", 248 * 1024},
      {"Ascend310B3", 248 * 1024},
      {"Ascend310B4", 248 * 1024},
      {"Ascend910_95", 256 * 1024},
      {"Ascend950", 256 * 1024},
      {"Ascend910_9579", 256 * 1024}, {"Ascend910_9581", 256 * 1024},
      {"Ascend910_9589", 256 * 1024}, {"Ascend910_9599", 256 * 1024},
  };
  const auto capacity = capacities.find(targetArch);
  if (capacity == capacities.end())
    return std::nullopt;
  return capacity->second;
}

} // namespace mlir::triton::ascend::ub
