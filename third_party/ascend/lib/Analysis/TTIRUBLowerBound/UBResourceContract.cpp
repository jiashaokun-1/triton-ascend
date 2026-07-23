#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "llvm/ADT/STLExtras.h"

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

bool haveEqualOptions(const StringMap<std::string> &lhs,
                      const StringMap<std::string> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const auto &option : lhs) {
    auto other = rhs.find(option.getKey());
    if (other == rhs.end() || other->second != option.getValue())
      return false;
  }
  return true;
}

bool areEqual(const PipelineStageContext &lhs,
              const PipelineStageContext &rhs) {
  return lhs.stageName == rhs.stageName &&
         haveEqualOptions(lhs.options, rhs.options);
}

bool areEqual(const PipelineIdentity &lhs, const PipelineIdentity &rhs) {
  return lhs.openSourcePipeline == rhs.openSourcePipeline &&
         lhs.canonicalTtirSha256 == rhs.canonicalTtirSha256 &&
         lhs.relevantOptionsJson == rhs.relevantOptionsJson &&
         lhs.targetArch == rhs.targetArch &&
         lhs.tritonVersion == rhs.tritonVersion &&
         lhs.cannVersionHash == rhs.cannVersionHash &&
         lhs.sha256 == rhs.sha256;
}

LogicalResult applyDisposition(MandatoryUBResourceGraph &graph,
                               const UBResourceContract &contract,
                               const PipelineStageContext &context) {
  switch (contract.apply(graph, context)) {
  case ContractDisposition::Preserve:
  case ContractDisposition::Transform:
    return success();
  case ContractDisposition::Invalidate:
    invalidateAll(graph, contract.id());
    return success();
  case ContractDisposition::InternalError:
    return failure();
  }
  return failure();
}

class InvalidateContract final : public UBResourceContract {
public:
  explicit InvalidateContract(const PipelineStageContext &stage)
      : stageName(stage.stageName) {
    for (const auto &option : stage.options)
      options[option.getKey()] = option.getValue();
  }

  StringRef id() const override { return "invalidate-unmodeled-stage"; }
  StringRef version() const override { return "1"; }
  bool matches(const PipelineStageContext &context) const override {
    return context.stageName == stageName &&
           haveEqualOptions(context.options, options);
  }
  ContractDisposition
  apply(MandatoryUBResourceGraph &,
        const PipelineStageContext &) const override {
    return ContractDisposition::Invalidate;
  }

private:
  std::string stageName;
  StringMap<std::string> options;
};

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

class DirectCopyContractBase : public UBResourceContract {
public:
  DirectCopyContractBase(const PipelineStageContext &stage,
                         int64_t expectedResourceCount,
                         int64_t expectedSourceElements,
                         unsigned expectedElementBitWidth,
                         int64_t expectedInputPayloadBytes)
      : stageName(stage.stageName),
        expectedResourceCount(expectedResourceCount),
        expectedSourceElements(expectedSourceElements),
        expectedElementBitWidth(expectedElementBitWidth),
        expectedInputPayloadBytes(expectedInputPayloadBytes) {
    for (const auto &option : stage.options)
      options[option.getKey()] = option.getValue();
  }

  StringRef version() const final { return "1"; }

  bool matches(const PipelineStageContext &context) const final {
    return context.stageName == stageName &&
           haveEqualOptions(context.options, options);
  }

protected:
  bool matchesResources(const MandatoryUBResourceGraph &graph) const {
    if (expectedResourceCount <= 0 || expectedSourceElements <= 0 ||
        expectedElementBitWidth == 0 || expectedInputPayloadBytes <= 0 ||
        graph.resources().size() !=
            static_cast<size_t>(expectedResourceCount))
      return false;
    return llvm::all_of(graph.resources(), [&](const auto &resource) {
      return resource.validity == ValidityState::Valid &&
             resource.origin == "tt.load" &&
             resource.kind == MaterializationKind::GMToUBLoad &&
             resource.minInstances == 1 &&
             resource.consumer == "tt.store" &&
             resource.sourceElements == expectedSourceElements &&
             resource.elementBitWidth == expectedElementBitWidth &&
             resource.minPayloadBytes == expectedInputPayloadBytes;
    });
  }

  LogicalResult appendTrace(MandatoryUBResourceGraph &graph,
                            StringRef contractId) const {
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal)
      if (failed(graph.appendResourceTrace(static_cast<ResourceId>(ordinal),
                                           contractId)))
        return failure();
    return success();
  }

private:
  std::string stageName;
  StringMap<std::string> options;
  int64_t expectedResourceCount;
  int64_t expectedSourceElements;
  unsigned expectedElementBitWidth;
  int64_t expectedInputPayloadBytes;
};

class DirectCopyPreserveContract final : public DirectCopyContractBase {
public:
  using DirectCopyContractBase::DirectCopyContractBase;

  StringRef id() const override { return "direct-copy-preserve"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (!matchesResources(graph))
      return ContractDisposition::Invalidate;
    if (failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Preserve;
  }
};

class DirectCopyMaxTilesContract final : public DirectCopyContractBase {
public:
  DirectCopyMaxTilesContract(const PipelineStageContext &stage,
                             int64_t expectedResourceCount,
                             int64_t expectedSourceElements,
                             unsigned expectedElementBitWidth,
                             int64_t expectedInputPayloadBytes,
                             int64_t maxTiles)
      : DirectCopyContractBase(stage, expectedResourceCount,
                               expectedSourceElements,
                               expectedElementBitWidth,
                               expectedInputPayloadBytes),
        maxTiles(maxTiles) {}

  StringRef id() const override { return "direct-copy-max-tiles"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (maxTiles <= 0)
      return ContractDisposition::InternalError;
    if (!matchesResources(graph))
      return ContractDisposition::Invalidate;
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal) {
      const int64_t payload = graph.resources()[ordinal].minPayloadBytes;
      int64_t transformed = payload / maxTiles;
      if (payload % maxTiles != 0)
        ++transformed;
      if (failed(graph.lowerResourcePayload(static_cast<ResourceId>(ordinal),
                                            transformed, id())))
        return ContractDisposition::InternalError;
    }
    return ContractDisposition::Transform;
  }

private:
  int64_t maxTiles;
};

class BinaryAddContractBase : public UBResourceContract {
public:
  BinaryAddContractBase(const PipelineStageContext &stage,
                        int64_t expectedResourceCount,
                        int64_t expectedSourceElements,
                        unsigned expectedElementBitWidth,
                        int64_t expectedInputPayloadBytes)
      : stageName(stage.stageName),
        expectedResourceCount(expectedResourceCount),
        expectedSourceElements(expectedSourceElements),
        expectedElementBitWidth(expectedElementBitWidth),
        expectedInputPayloadBytes(expectedInputPayloadBytes) {
    for (const auto &option : stage.options)
      options[option.getKey()] = option.getValue();
  }

  StringRef version() const final { return "1"; }

  bool matches(const PipelineStageContext &context) const final {
    return context.stageName == stageName &&
           haveEqualOptions(context.options, options);
  }

protected:
  bool matchesResources(const MandatoryUBResourceGraph &graph,
                        bool requireDistinct) const {
    if (expectedResourceCount != 2 || expectedSourceElements <= 0 ||
        expectedElementBitWidth == 0 || expectedInputPayloadBytes <= 0 ||
        graph.resources().size() != 2 || graph.witnesses().size() != 1)
      return false;
    const CoexistenceWitness &witness = graph.witnesses().front();
    if (witness.resources != SmallVector<ResourceId>({0, 1}))
      return false;
    if (requireDistinct ? !graph.hasPairwiseDistinctWitness(0)
                        : !graph.hasPairwiseMayAliasWitness(0))
      return false;
    const MandatoryUBResource &lhs = graph.resources()[0];
    const MandatoryUBResource &rhs = graph.resources()[1];
    if (lhs.birth.ordinal == rhs.birth.ordinal ||
        lhs.lastRequiredUse.ordinal != rhs.lastRequiredUse.ordinal ||
        lhs.birth.ordinal >= lhs.lastRequiredUse.ordinal ||
        rhs.birth.ordinal >= rhs.lastRequiredUse.ordinal)
      return false;
    return llvm::all_of(graph.resources(), [&](const auto &resource) {
      return resource.validity == ValidityState::Valid &&
             resource.origin == "tt.load" &&
             resource.kind == MaterializationKind::GMToUBLoad &&
             resource.minInstances == 1 &&
             resource.consumer == "arith.addf" &&
             resource.sourceElements == expectedSourceElements &&
             resource.elementBitWidth == expectedElementBitWidth &&
             resource.minPayloadBytes == expectedInputPayloadBytes;
    });
  }

  LogicalResult appendTrace(MandatoryUBResourceGraph &graph,
                            StringRef contractId) const {
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal)
      if (failed(graph.appendResourceTrace(static_cast<ResourceId>(ordinal),
                                           contractId)))
        return failure();
    return success();
  }

private:
  std::string stageName;
  StringMap<std::string> options;
  int64_t expectedResourceCount;
  int64_t expectedSourceElements;
  unsigned expectedElementBitWidth;
  int64_t expectedInputPayloadBytes;
};

class BinaryAddPreserveContract final : public BinaryAddContractBase {
public:
  using BinaryAddContractBase::BinaryAddContractBase;

  StringRef id() const override { return "binary-add-preserve"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    // Preserve is used on both sides of the materialization contract.  Before
    // materialization, the matcher can only prove that the two inputs may
    // alias; afterwards, BinaryAddMaxTilesContract monotonically refines that
    // relation to must-distinct.  Accept either sound state here.  The witness
    // solver still requires must-distinct before it is allowed to add the two
    // payloads, so this does not turn a pre-materialization may-alias fact into
    // a lower-bound certificate.
    if (!matchesResources(graph, /*requireDistinct=*/false) &&
        !matchesResources(graph, /*requireDistinct=*/true))
      return ContractDisposition::Invalidate;
    if (failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Preserve;
  }
};

class BinaryAddMaxTilesContract final : public BinaryAddContractBase {
public:
  BinaryAddMaxTilesContract(const PipelineStageContext &stage,
                            int64_t expectedResourceCount,
                            int64_t expectedSourceElements,
                            unsigned expectedElementBitWidth,
                            int64_t expectedInputPayloadBytes,
                            int64_t maxTiles)
      : BinaryAddContractBase(stage, expectedResourceCount,
                              expectedSourceElements,
                              expectedElementBitWidth,
                              expectedInputPayloadBytes),
        maxTiles(maxTiles) {}

  StringRef id() const override { return "binary-add-max-tiles"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (maxTiles <= 0)
      return ContractDisposition::InternalError;
    if (!matchesResources(graph, /*requireDistinct=*/false))
      return ContractDisposition::Invalidate;
    if (failed(graph.refineWitnessToMustDistinct(0, id())))
      return ContractDisposition::InternalError;
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal) {
      const int64_t payload = graph.resources()[ordinal].minPayloadBytes;
      int64_t transformed = payload / maxTiles;
      if (payload % maxTiles != 0)
        ++transformed;
      if (failed(graph.lowerResourcePayload(static_cast<ResourceId>(ordinal),
                                            transformed, id())))
        return ContractDisposition::InternalError;
    }
    return ContractDisposition::Transform;
  }

private:
  int64_t maxTiles;
};

class LoopCarriedAddContractBase : public UBResourceContract {
public:
  LoopCarriedAddContractBase(const PipelineStageContext &stage,
                             int64_t expectedResourceCount,
                             int64_t expectedSourceElements,
                             unsigned expectedElementBitWidth,
                             int64_t expectedInputPayloadBytes)
      : stageName(stage.stageName),
        expectedResourceCount(expectedResourceCount),
        expectedSourceElements(expectedSourceElements),
        expectedElementBitWidth(expectedElementBitWidth),
        expectedInputPayloadBytes(expectedInputPayloadBytes) {
    for (const auto &option : stage.options)
      options[option.getKey()] = option.getValue();
  }

  StringRef version() const final { return "1"; }

  bool matches(const PipelineStageContext &context) const final {
    return context.stageName == stageName &&
           haveEqualOptions(context.options, options);
  }

protected:
  bool matchesResources(const MandatoryUBResourceGraph &graph,
                        bool requireDistinct) const {
    if (expectedResourceCount != 2 || expectedSourceElements <= 0 ||
        expectedElementBitWidth == 0 || expectedInputPayloadBytes <= 0 ||
        graph.resources().size() != 2 || graph.witnesses().size() != 1)
      return false;
    const CoexistenceWitness &witness = graph.witnesses().front();
    if (witness.resources != SmallVector<ResourceId>({0, 1}))
      return false;
    if (requireDistinct ? !graph.hasPairwiseDistinctWitness(0)
                        : !graph.hasPairwiseMayAliasWitness(0))
      return false;
    const MandatoryUBResource &accumulator = graph.resources()[0];
    const MandatoryUBResource &stepInput = graph.resources()[1];
    auto matchesCommon = [&](const MandatoryUBResource &resource) {
      return resource.validity == ValidityState::Valid &&
             resource.origin == "tt.load" &&
             resource.kind == MaterializationKind::GMToUBLoad &&
             resource.minInstances == 1 &&
             resource.sourceElements == expectedSourceElements &&
             resource.elementBitWidth == expectedElementBitWidth &&
             resource.minPayloadBytes == expectedInputPayloadBytes;
    };
    return matchesCommon(accumulator) && matchesCommon(stepInput) &&
           accumulator.debugName == "loop-carried-accumulator" &&
           accumulator.consumer == "scf.for" &&
           stepInput.debugName == "loop-step-input" &&
           stepInput.consumer == "arith.addf" &&
           accumulator.birth.ordinal < stepInput.birth.ordinal &&
           stepInput.birth.ordinal == stepInput.lastRequiredUse.ordinal &&
           stepInput.lastRequiredUse.ordinal <
               accumulator.lastRequiredUse.ordinal;
  }

  LogicalResult appendTrace(MandatoryUBResourceGraph &graph,
                            StringRef contractId) const {
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal)
      if (failed(graph.appendResourceTrace(static_cast<ResourceId>(ordinal),
                                           contractId)))
        return failure();
    return success();
  }

private:
  std::string stageName;
  StringMap<std::string> options;
  int64_t expectedResourceCount;
  int64_t expectedSourceElements;
  unsigned expectedElementBitWidth;
  int64_t expectedInputPayloadBytes;
};

class LoopCarriedAddPreserveContract final
    : public LoopCarriedAddContractBase {
public:
  using LoopCarriedAddContractBase::LoopCarriedAddContractBase;

  StringRef id() const override { return "loop-carried-add-preserve"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (!matchesResources(graph, /*requireDistinct=*/false) &&
        !matchesResources(graph, /*requireDistinct=*/true))
      return ContractDisposition::Invalidate;
    if (failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Preserve;
  }
};

class LoopCarriedAddMaxTilesContract final
    : public LoopCarriedAddContractBase {
public:
  LoopCarriedAddMaxTilesContract(const PipelineStageContext &stage,
                                 int64_t expectedResourceCount,
                                 int64_t expectedSourceElements,
                                 unsigned expectedElementBitWidth,
                                 int64_t expectedInputPayloadBytes,
                                 int64_t maxTiles)
      : LoopCarriedAddContractBase(
            stage, expectedResourceCount, expectedSourceElements,
            expectedElementBitWidth, expectedInputPayloadBytes),
        maxTiles(maxTiles) {}

  StringRef id() const override { return "loop-carried-add-max-tiles"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    // Only the observed tile=1 lowering is admitted: one allocation carries
    // the accumulator across iterations while a second allocation is loaded
    // inside each iteration.  Multi-tile loop liveness needs a new profile.
    if (maxTiles != 1 ||
        !matchesResources(graph, /*requireDistinct=*/false))
      return ContractDisposition::Invalidate;
    if (failed(graph.refineWitnessToMustDistinct(0, id())) ||
        failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Transform;
  }

private:
  int64_t maxTiles;
};

class LoopCarriedAddMultiBufferContract final
    : public LoopCarriedAddContractBase {
public:
  LoopCarriedAddMultiBufferContract(
      const PipelineStageContext &stage, int64_t expectedResourceCount,
      int64_t expectedSourceElements, unsigned expectedElementBitWidth,
      int64_t expectedInputPayloadBytes, int64_t expectedStepInputInstances)
      : LoopCarriedAddContractBase(
            stage, expectedResourceCount, expectedSourceElements,
            expectedElementBitWidth, expectedInputPayloadBytes),
        expectedStepInputInstances(expectedStepInputInstances) {}

  StringRef id() const override { return "loop-carried-add-multibuffer"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    // MarkMultiBuffer gives the loop-local GM->UB load two physical buffers.
    // PlanMemory expands that storage entry into two simultaneously reserved
    // instances.  The loop-external accumulator remains single-buffered.
    if (expectedStepInputInstances != 2 ||
        !matchesResources(graph, /*requireDistinct=*/true))
      return ContractDisposition::Invalidate;
    if (failed(graph.raiseResourceInstances(
            /*id=*/1, expectedStepInputInstances, id())) ||
        failed(graph.appendResourceTrace(/*id=*/0, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Transform;
  }

private:
  int64_t expectedStepInputInstances;
};

class ReshapeCopyContractBase : public UBResourceContract {
public:
  ReshapeCopyContractBase(const PipelineStageContext &stage,
                          int64_t expectedResourceCount,
                          int64_t expectedSourceElements,
                          unsigned expectedElementBitWidth,
                          int64_t expectedInputPayloadBytes)
      : stageName(stage.stageName),
        expectedResourceCount(expectedResourceCount),
        expectedSourceElements(expectedSourceElements),
        expectedElementBitWidth(expectedElementBitWidth),
        expectedInputPayloadBytes(expectedInputPayloadBytes) {
    for (const auto &option : stage.options)
      options[option.getKey()] = option.getValue();
  }

  StringRef version() const final { return "1"; }

  bool matches(const PipelineStageContext &context) const final {
    return context.stageName == stageName &&
           haveEqualOptions(context.options, options);
  }

protected:
  bool matchesResources(const MandatoryUBResourceGraph &graph,
                        bool requireMustAlias) const {
    if (expectedResourceCount != 2 || expectedSourceElements <= 0 ||
        expectedElementBitWidth == 0 || expectedInputPayloadBytes <= 0 ||
        graph.resources().size() != 2 || !graph.witnesses().empty() ||
        (requireMustAlias ? !graph.hasMustAlias(0, 1)
                          : !graph.hasMayAlias(0, 1)))
      return false;
    const MandatoryUBResource &source = graph.resources()[0];
    const MandatoryUBResource &view = graph.resources()[1];
    auto matchesCommon = [&](const MandatoryUBResource &resource) {
      return resource.validity == ValidityState::Valid &&
             resource.minInstances == 1 &&
             resource.sourceElements == expectedSourceElements &&
             resource.elementBitWidth == expectedElementBitWidth &&
             resource.minPayloadBytes == expectedInputPayloadBytes;
    };
    return matchesCommon(source) && matchesCommon(view) &&
           source.origin == "tt.load" &&
           source.kind == MaterializationKind::GMToUBLoad &&
           source.consumer == "tt.reshape" && view.origin == "tt.reshape" &&
           view.kind == MaterializationKind::ViewAlias &&
           (view.consumer == "tt.store" ||
            view.consumer == "tt.reshape");
  }

  LogicalResult appendTrace(MandatoryUBResourceGraph &graph,
                            StringRef contractId) const {
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal)
      if (failed(graph.appendResourceTrace(static_cast<ResourceId>(ordinal),
                                           contractId)))
        return failure();
    return success();
  }

private:
  std::string stageName;
  StringMap<std::string> options;
  int64_t expectedResourceCount;
  int64_t expectedSourceElements;
  unsigned expectedElementBitWidth;
  int64_t expectedInputPayloadBytes;
};

class ReshapeCopyPreserveContract final : public ReshapeCopyContractBase {
public:
  using ReshapeCopyContractBase::ReshapeCopyContractBase;

  StringRef id() const override { return "reshape-copy-preserve"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    // As with binary-add, preserve spans both sides of materialization.  The
    // source matcher starts with may-alias and ReshapeCopyMaxTilesContract
    // later refines it to must-alias.  Merely preserving may-alias is not
    // enough for the solver to collapse the two resources into one class.
    if (!matchesResources(graph, /*requireMustAlias=*/false) &&
        !matchesResources(graph, /*requireMustAlias=*/true))
      return ContractDisposition::Invalidate;
    if (failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Preserve;
  }
};

class ReshapeCopyMaxTilesContract final : public ReshapeCopyContractBase {
public:
  ReshapeCopyMaxTilesContract(const PipelineStageContext &stage,
                              int64_t expectedResourceCount,
                              int64_t expectedSourceElements,
                              unsigned expectedElementBitWidth,
                              int64_t expectedInputPayloadBytes,
                              int64_t maxTiles)
      : ReshapeCopyContractBase(stage, expectedResourceCount,
                                expectedSourceElements,
                                expectedElementBitWidth,
                                expectedInputPayloadBytes),
        maxTiles(maxTiles) {}

  StringRef id() const override { return "reshape-copy-max-tiles"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (maxTiles <= 0)
      return ContractDisposition::InternalError;
    if (!matchesResources(graph, /*requireMustAlias=*/false))
      return ContractDisposition::Invalidate;
    if (failed(graph.refineMayAliasToMustAlias(0, 1, id())))
      return ContractDisposition::InternalError;
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal) {
      const int64_t payload = graph.resources()[ordinal].minPayloadBytes;
      int64_t transformed = payload / maxTiles;
      if (payload % maxTiles != 0)
        ++transformed;
      if (failed(graph.lowerResourcePayload(static_cast<ResourceId>(ordinal),
                                            transformed, id())))
        return ContractDisposition::InternalError;
    }
    return ContractDisposition::Transform;
  }

private:
  int64_t maxTiles;
};

class ReductionSumContractBase : public UBResourceContract {
public:
  ReductionSumContractBase(const PipelineStageContext &stage,
                           int64_t expectedResourceCount,
                           int64_t expectedSourceElements,
                           unsigned expectedElementBitWidth,
                           int64_t expectedInputPayloadBytes,
                           int64_t expectedScratchPayloadBytes,
                           int64_t expectedAccumulatorPayloadBytes)
      : stageName(stage.stageName),
        expectedResourceCount(expectedResourceCount),
        expectedSourceElements(expectedSourceElements),
        expectedElementBitWidth(expectedElementBitWidth),
        expectedInputPayloadBytes(expectedInputPayloadBytes),
        expectedScratchPayloadBytes(expectedScratchPayloadBytes),
        expectedAccumulatorPayloadBytes(expectedAccumulatorPayloadBytes) {
    for (const auto &option : stage.options)
      options[option.getKey()] = option.getValue();
  }

  StringRef version() const final { return "1"; }

  bool matches(const PipelineStageContext &context) const final {
    return context.stageName == stageName &&
           haveEqualOptions(context.options, options);
  }

protected:
  bool matchesResources(const MandatoryUBResourceGraph &graph,
                        bool requireDistinct) const {
    if (expectedResourceCount != 3 || expectedSourceElements <= 0 ||
        expectedElementBitWidth != 32 || expectedInputPayloadBytes <= 0 ||
        expectedScratchPayloadBytes <= 0 ||
        expectedAccumulatorPayloadBytes <= 0 ||
        graph.resources().size() != 3 || graph.witnesses().size() != 1)
      return false;
    const CoexistenceWitness &witness = graph.witnesses().front();
    if (witness.resources != SmallVector<ResourceId>({0, 1, 2}) ||
        (requireDistinct ? !graph.hasPairwiseDistinctWitness(0)
                         : !graph.hasPairwiseMayAliasWitness(0)))
      return false;
    const MandatoryUBResource &input = graph.resources()[0];
    const MandatoryUBResource &scratch = graph.resources()[1];
    const MandatoryUBResource &accumulator = graph.resources()[2];
    auto matchesCommon = [&](const MandatoryUBResource &resource) {
      return resource.validity == ValidityState::Valid &&
             resource.minInstances == 1 &&
             resource.sourceElements == expectedSourceElements &&
             resource.elementBitWidth == expectedElementBitWidth;
    };
    return matchesCommon(input) && matchesCommon(scratch) &&
           matchesCommon(accumulator) && input.origin == "tt.load" &&
           input.kind == MaterializationKind::GMToUBLoad &&
           input.consumer == "tt.reduce" &&
           input.minPayloadBytes == expectedInputPayloadBytes &&
           scratch.origin == "tt.reduce" &&
           scratch.kind == MaterializationKind::ReductionScratch &&
           scratch.consumer == "tt.reduce" &&
           scratch.minPayloadBytes == expectedScratchPayloadBytes &&
           accumulator.origin == "tt.reduce" &&
           accumulator.kind == MaterializationKind::ReductionAccumulator &&
           accumulator.consumer == "tt.store" &&
           accumulator.minPayloadBytes == expectedAccumulatorPayloadBytes &&
           input.lastRequiredUse.ordinal == scratch.birth.ordinal &&
           scratch.birth.ordinal == scratch.lastRequiredUse.ordinal &&
           accumulator.birth.ordinal == scratch.birth.ordinal &&
           accumulator.lastRequiredUse.ordinal >
               accumulator.birth.ordinal;
  }

  LogicalResult appendTrace(MandatoryUBResourceGraph &graph,
                            StringRef contractId) const {
    for (size_t ordinal = 0; ordinal < graph.resources().size(); ++ordinal)
      if (failed(graph.appendResourceTrace(static_cast<ResourceId>(ordinal),
                                           contractId)))
        return failure();
    return success();
  }

private:
  std::string stageName;
  StringMap<std::string> options;
  int64_t expectedResourceCount;
  int64_t expectedSourceElements;
  unsigned expectedElementBitWidth;
  int64_t expectedInputPayloadBytes;
  int64_t expectedScratchPayloadBytes;
  int64_t expectedAccumulatorPayloadBytes;
};

class ReductionSumPreserveContract final : public ReductionSumContractBase {
public:
  using ReductionSumContractBase::ReductionSumContractBase;

  StringRef id() const override { return "reduction-sum-preserve"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (!matchesResources(graph, /*requireDistinct=*/false) &&
        !matchesResources(graph, /*requireDistinct=*/true))
      return ContractDisposition::Invalidate;
    if (failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Preserve;
  }
};

class ReductionSumMaxTilesContract final : public ReductionSumContractBase {
public:
  ReductionSumMaxTilesContract(const PipelineStageContext &stage,
                               int64_t expectedResourceCount,
                               int64_t expectedSourceElements,
                               unsigned expectedElementBitWidth,
                               int64_t expectedInputPayloadBytes,
                               int64_t expectedScratchPayloadBytes,
                               int64_t expectedAccumulatorPayloadBytes,
                               int64_t maxTiles)
      : ReductionSumContractBase(
            stage, expectedResourceCount, expectedSourceElements,
            expectedElementBitWidth, expectedInputPayloadBytes,
            expectedScratchPayloadBytes, expectedAccumulatorPayloadBytes),
        maxTiles(maxTiles) {}

  StringRef id() const override { return "reduction-sum-max-tiles"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    // The P3 slice is intentionally restricted to the observed tile=1
    // lowering.  A later profile must supply a separately validated formula
    // before multi-tile reductions can be admitted.
    if (maxTiles != 1 || !matchesResources(graph, /*requireDistinct=*/false))
      return ContractDisposition::Invalidate;
    if (failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Transform;
  }

private:
  int64_t maxTiles;
};

class ReductionSumExtraBufferContract final
    : public ReductionSumContractBase {
public:
  using ReductionSumContractBase::ReductionSumContractBase;

  StringRef id() const override { return "reduction-sum-extra-buffer"; }

  ContractDisposition
  apply(MandatoryUBResourceGraph &graph,
        const PipelineStageContext &) const override {
    if (!matchesResources(graph, /*requireDistinct=*/false))
      return ContractDisposition::Invalidate;
    if (failed(graph.refineWitnessToMustDistinct(0, id())) ||
        failed(appendTrace(graph, id())))
      return ContractDisposition::InternalError;
    return ContractDisposition::Transform;
  }
};

} // namespace

void PipelineContractRegistry::setProfileIdentity(PipelineIdentity identity) {
  profileIdentity = std::move(identity);
  profileContracts.clear();
}

LogicalResult PipelineContractRegistry::addProfileContract(
    PipelineContractBinding binding,
    std::unique_ptr<UBResourceContract> contract) {
  if (!profileIdentity || !contract ||
      contract->id() != binding.contractId ||
      contract->version() != binding.contractVersion ||
      !contract->matches(binding.stage))
    return failure();
  profileContracts.push_back(
      {.binding = std::move(binding), .contract = std::move(contract)});
  return success();
}

bool PipelineContractRegistry::matchesProfile(
    const PipelineIdentity &identity,
    ArrayRef<PipelineStageContext> stages) const {
  if (!profileIdentity || !areEqual(*profileIdentity, identity) ||
      profileContracts.size() != stages.size())
    return false;
  for (size_t ordinal = 0; ordinal < stages.size(); ++ordinal) {
    const ProfileContractEntry &entry = profileContracts[ordinal];
    if (!areEqual(entry.binding.stage, stages[ordinal]) ||
        entry.contract->id() != entry.binding.contractId ||
        entry.contract->version() != entry.binding.contractVersion ||
        !entry.contract->matches(stages[ordinal]))
      return false;
  }
  return true;
}

LogicalResult PipelineContractRegistry::applyProfileStage(
    MandatoryUBResourceGraph &graph, const PipelineStageContext &context,
    size_t stageOrdinal) const {
  if (stageOrdinal >= profileContracts.size())
    return failure();
  const ProfileContractEntry &entry = profileContracts[stageOrdinal];
  if (!areEqual(entry.binding.stage, context) ||
      !entry.contract->matches(context))
    return failure();
  return applyDisposition(graph, *entry.contract, context);
}

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

  return applyDisposition(graph, *matchedContract, context);
}

std::unique_ptr<UBResourceContract> makeFixedTileContract(StringRef stageName,
                                                          int64_t maxTiles) {
  return std::make_unique<FixedTileContract>(stageName, maxTiles);
}

std::unique_ptr<UBResourceContract>
makeInvalidateContract(const PipelineStageContext &stage) {
  return std::make_unique<InvalidateContract>(stage);
}

std::unique_ptr<UBResourceContract> makeDirectCopyPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes) {
  return std::make_unique<DirectCopyPreserveContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes);
}

std::unique_ptr<UBResourceContract> makeDirectCopyMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles) {
  return std::make_unique<DirectCopyMaxTilesContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes, maxTiles);
}

std::unique_ptr<UBResourceContract> makeBinaryAddPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes) {
  return std::make_unique<BinaryAddPreserveContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes);
}

std::unique_ptr<UBResourceContract> makeBinaryAddMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles) {
  return std::make_unique<BinaryAddMaxTilesContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes, maxTiles);
}

std::unique_ptr<UBResourceContract> makeLoopCarriedAddPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes) {
  return std::make_unique<LoopCarriedAddPreserveContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes);
}

std::unique_ptr<UBResourceContract> makeLoopCarriedAddMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles) {
  return std::make_unique<LoopCarriedAddMaxTilesContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes, maxTiles);
}

std::unique_ptr<UBResourceContract> makeLoopCarriedAddMultiBufferContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedStepInputInstances) {
  return std::make_unique<LoopCarriedAddMultiBufferContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes,
      expectedStepInputInstances);
}

std::unique_ptr<UBResourceContract> makeReshapeCopyPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes) {
  return std::make_unique<ReshapeCopyPreserveContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes);
}

std::unique_ptr<UBResourceContract> makeReshapeCopyMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t maxTiles) {
  return std::make_unique<ReshapeCopyMaxTilesContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes, maxTiles);
}

std::unique_ptr<UBResourceContract> makeReductionSumPreserveContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedScratchPayloadBytes,
    int64_t expectedAccumulatorPayloadBytes) {
  return std::make_unique<ReductionSumPreserveContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes,
      expectedScratchPayloadBytes, expectedAccumulatorPayloadBytes);
}

std::unique_ptr<UBResourceContract> makeReductionSumMaxTilesContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedScratchPayloadBytes,
    int64_t expectedAccumulatorPayloadBytes, int64_t maxTiles) {
  return std::make_unique<ReductionSumMaxTilesContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes,
      expectedScratchPayloadBytes, expectedAccumulatorPayloadBytes, maxTiles);
}

std::unique_ptr<UBResourceContract> makeReductionSumExtraBufferContract(
    const PipelineStageContext &stage, int64_t expectedResourceCount,
    int64_t expectedSourceElements, unsigned expectedElementBitWidth,
    int64_t expectedInputPayloadBytes, int64_t expectedScratchPayloadBytes,
    int64_t expectedAccumulatorPayloadBytes) {
  return std::make_unique<ReductionSumExtraBufferContract>(
      stage, expectedResourceCount, expectedSourceElements,
      expectedElementBitWidth, expectedInputPayloadBytes,
      expectedScratchPayloadBytes, expectedAccumulatorPayloadBytes);
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
