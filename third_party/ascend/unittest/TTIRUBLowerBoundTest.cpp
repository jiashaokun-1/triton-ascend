#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace mlir::triton::ascend::ub {
namespace {

class FixedDispositionContract final : public UBResourceContract {
public:
  FixedDispositionContract(StringRef stageName,
                           ContractDisposition disposition)
      : stageName(stageName.str()), disposition(disposition) {}

  StringRef id() const override { return "fixed-disposition"; }
  StringRef version() const override { return "1"; }
  bool matches(const PipelineStageContext &context) const override {
    return context.stageName == stageName;
  }
  ContractDisposition apply(MandatoryUBResourceGraph &,
                            const PipelineStageContext &) const override {
    return disposition;
  }

private:
  std::string stageName;
  ContractDisposition disposition;
};

TEST(MandatoryUBResourceGraph, SingletonUsesLargestMandatoryResource) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 64 * 1024, 1});
  graph.addResource({"load1", 192 * 1024 + 4, 1});
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 192 * 1024 + 4);
  EXPECT_EQ(result->resourceIds.size(), 1u);
}

TEST(MandatoryUBResourceGraph, InvalidResourceCannotContribute) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource({"load0", 256 * 1024, 1});
  graph.invalidate(id, "unknown-stage");
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 0);
}

TEST(MandatoryUBResourceGraph, ArithmeticOverflowFailsClosed) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", INT64_MAX, 2});
  EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
}

TEST(MandatoryUBResourceGraph, PairwiseOverlapIsNotAThreeWayWitness) {
  MandatoryUBResourceGraph graph;
  auto a = graph.addResource({"a", 32, 1});
  auto b = graph.addResource({"b", 64, 1});
  auto c = graph.addResource({"c", 128, 1});
  graph.addMustDistinct(a, b);
  graph.addMustDistinct(b, c);
  graph.addMustDistinct(a, c);
  graph.addWitness({a, b});
  graph.addWitness({b, c});
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 192);
}

TEST(MandatoryUBResourceGraph, PossibleAliasCannotBeSummed) {
  MandatoryUBResourceGraph graph;
  auto a = graph.addResource({"a", 64, 1});
  auto b = graph.addResource({"b", 128, 1});
  graph.addMayAlias(a, b);
  graph.addWitness({a, b});
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 128);
}

TEST(MandatoryUBResourceGraph,
     InvalidWitnessMemberDoesNotCreateSubsetWitness) {
  MandatoryUBResourceGraph graph;
  auto a = graph.addResource({"a", 100, 1});
  auto b = graph.addResource({"b", 100, 1});
  auto c = graph.addResource({"c", 1, 1});
  graph.addMustDistinct(a, b);
  graph.addMustDistinct(a, c);
  graph.addMustDistinct(b, c);
  graph.addWitness({a, b, c});
  graph.invalidate(c, "invalid-contract");

  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 100);
  EXPECT_EQ(result->kind, "singleton");
}

TEST(MandatoryUBResourceGraph, ResourceIdExhaustionFailsClosed) {
  MandatoryUBResourceGraph graph(StableIdLimits{1, 1});
  auto first = graph.addResource({"a", 64, 1});
  EXPECT_EQ(first, 0u);

  EXPECT_EQ(graph.addResource({"b", 128, 1}), InvalidResourceId);
  EXPECT_EQ(graph.resources().size(), 1u);
  EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
  EXPECT_TRUE(failed(graph.solveWitnessLowerBound()));
}

TEST(MandatoryUBResourceGraph, WitnessIdExhaustionFailsClosed) {
  MandatoryUBResourceGraph graph(StableIdLimits{1, 1});
  auto resource = graph.addResource({"a", 64, 1});

  auto first = graph.addWitness({resource});
  EXPECT_EQ(first, 0u);

  EXPECT_EQ(graph.addWitness({resource}), InvalidWitnessId);
  EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
  EXPECT_TRUE(failed(graph.solveWitnessLowerBound()));
}

TEST(MandatoryUBResourceGraph, InvalidRelationIdsFailClosed) {
  using Relation = void (MandatoryUBResourceGraph::*)(ResourceId, ResourceId);
  for (Relation relation : {&MandatoryUBResourceGraph::addMayAlias,
                            &MandatoryUBResourceGraph::addMustAlias,
                            &MandatoryUBResourceGraph::addMustDistinct}) {
    MandatoryUBResourceGraph graph;
    auto resource = graph.addResource({"a", 64, 1});
    (graph.*relation)(resource, InvalidResourceId);
    EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
    EXPECT_TRUE(failed(graph.solveWitnessLowerBound()));
  }
}

TEST(MandatoryUBResourceGraph, FixedInsertionApiSupportsPlannedCallSites) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource({"load0", 64, 1});
  EXPECT_EQ(graph.resources()[id].debugName, "load0");
  graph.invalidate(id, "test");

  graph.addResource({"load1", 32, 1});
  auto witness = graph.addWitness({id});
  graph.addWitness({id});

  EXPECT_NE(id, InvalidResourceId);
  EXPECT_NE(witness, InvalidWitnessId);
}

TEST(UBResourceContract, UnknownStageInvalidatesResources) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  EXPECT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "unknown-pass"})));
  EXPECT_EQ(graph.solveSingletonLowerBound()->bytes, 0);
}

TEST(UBResourceContract, MismatchedContractInvalidatesResources) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.addForTesting(std::make_unique<FixedDispositionContract>(
      "known-pass", ContractDisposition::Preserve));
  EXPECT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "other-pass"})));
  EXPECT_EQ(graph.solveSingletonLowerBound()->bytes, 0);
}

TEST(UBResourceContract, PreserveKeepsResourcesValid) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.addForTesting(std::make_unique<FixedDispositionContract>(
      "preserve", ContractDisposition::Preserve));
  EXPECT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "preserve"})));
  EXPECT_EQ(graph.solveSingletonLowerBound()->bytes, 262144);
}

TEST(UBResourceContract, TransformCanOnlyLowerToProvenMinimum) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.addForTesting(makeFixedTileContract("tile", 2));
  EXPECT_TRUE(
      succeeded(registry.applyOrInvalidateAll(graph, {.stageName = "tile"})));
  EXPECT_EQ(graph.resources()[id].minPayloadBytes, 131072);
}

TEST(UBResourceContract, ExplicitInvalidateInvalidatesResources) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.addForTesting(std::make_unique<FixedDispositionContract>(
      "invalidate", ContractDisposition::Invalidate));
  EXPECT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "invalidate"})));
  EXPECT_EQ(graph.solveSingletonLowerBound()->bytes, 0);
}

TEST(UBResourceContract, InternalErrorReturnsFailure) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.addForTesting(std::make_unique<FixedDispositionContract>(
      "broken", ContractDisposition::InternalError));
  EXPECT_TRUE(failed(
      registry.applyOrInvalidateAll(graph, {.stageName = "broken"})));
}

TEST(UBResourceContract, CapacityHasNoUnknownDefault) {
  EXPECT_EQ(*getUBCapacityBytes("Ascend910B"), 192 * 1024);
  EXPECT_EQ(*getUBCapacityBytes("Ascend910_95"), 256 * 1024);
  EXPECT_EQ(*getUBCapacityBytes("Ascend950"), 256 * 1024);
  EXPECT_FALSE(getUBCapacityBytes("future-chip").has_value());
}

} // namespace
} // namespace mlir::triton::ascend::ub
