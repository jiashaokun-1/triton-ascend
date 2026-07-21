#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace mlir::triton::ascend::ub {
namespace {

ResourceId requireResourceId(FailureOr<ResourceId> id) {
  EXPECT_TRUE(succeeded(id));
  return succeeded(id) ? *id : ResourceId{0};
}

WitnessId requireWitnessId(FailureOr<WitnessId> id) {
  EXPECT_TRUE(succeeded(id));
  return succeeded(id) ? *id : WitnessId{0};
}

TEST(MandatoryUBResourceGraph, SingletonUsesLargestMandatoryResource) {
  MandatoryUBResourceGraph graph;
  requireResourceId(graph.addResource({"load0", 64 * 1024, 1}));
  requireResourceId(graph.addResource({"load1", 192 * 1024 + 4, 1}));
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 192 * 1024 + 4);
  EXPECT_EQ(result->resourceIds.size(), 1u);
}

TEST(MandatoryUBResourceGraph, InvalidResourceCannotContribute) {
  MandatoryUBResourceGraph graph;
  auto id = requireResourceId(graph.addResource({"load0", 256 * 1024, 1}));
  graph.invalidate(id, "unknown-stage");
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 0);
}

TEST(MandatoryUBResourceGraph, ArithmeticOverflowFailsClosed) {
  MandatoryUBResourceGraph graph;
  requireResourceId(graph.addResource({"load0", INT64_MAX, 2}));
  EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
}

TEST(MandatoryUBResourceGraph, PairwiseOverlapIsNotAThreeWayWitness) {
  MandatoryUBResourceGraph graph;
  auto a = requireResourceId(graph.addResource({"a", 32, 1}));
  auto b = requireResourceId(graph.addResource({"b", 64, 1}));
  auto c = requireResourceId(graph.addResource({"c", 128, 1}));
  graph.addMustDistinct(a, b);
  graph.addMustDistinct(b, c);
  graph.addMustDistinct(a, c);
  requireWitnessId(graph.addWitness({a, b}));
  requireWitnessId(graph.addWitness({b, c}));
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 192);
}

TEST(MandatoryUBResourceGraph, PossibleAliasCannotBeSummed) {
  MandatoryUBResourceGraph graph;
  auto a = requireResourceId(graph.addResource({"a", 64, 1}));
  auto b = requireResourceId(graph.addResource({"b", 128, 1}));
  graph.addMayAlias(a, b);
  requireWitnessId(graph.addWitness({a, b}));
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 128);
}

TEST(MandatoryUBResourceGraph,
     InvalidWitnessMemberDoesNotCreateSubsetWitness) {
  MandatoryUBResourceGraph graph;
  auto a = requireResourceId(graph.addResource({"a", 100, 1}));
  auto b = requireResourceId(graph.addResource({"b", 100, 1}));
  auto c = requireResourceId(graph.addResource({"c", 1, 1}));
  graph.addMustDistinct(a, b);
  graph.addMustDistinct(a, c);
  graph.addMustDistinct(b, c);
  requireWitnessId(graph.addWitness({a, b, c}));
  graph.invalidate(c, "invalid-contract");

  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 100);
  EXPECT_EQ(result->kind, "singleton");
}

TEST(MandatoryUBResourceGraph, ResourceIdExhaustionFailsClosed) {
  MandatoryUBResourceGraph graph(StableIdLimits{1, 1});
  auto first = graph.addResource({"a", 64, 1});
  ASSERT_TRUE(succeeded(first));
  EXPECT_EQ(*first, 0u);

  EXPECT_TRUE(failed(graph.addResource({"b", 128, 1})));
  EXPECT_EQ(graph.resources().size(), 1u);
  EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
}

TEST(MandatoryUBResourceGraph, WitnessIdExhaustionFailsClosed) {
  MandatoryUBResourceGraph graph(StableIdLimits{1, 1});
  auto resource = graph.addResource({"a", 64, 1});
  ASSERT_TRUE(succeeded(resource));

  auto first = graph.addWitness({*resource});
  ASSERT_TRUE(succeeded(first));
  EXPECT_EQ(*first, 0u);

  EXPECT_TRUE(failed(graph.addWitness({*resource})));
  EXPECT_TRUE(failed(graph.solveWitnessLowerBound()));
}

} // namespace
} // namespace mlir::triton::ascend::ub
