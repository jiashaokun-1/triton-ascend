#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace mlir::triton::ascend::ub {
namespace {

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

} // namespace
} // namespace mlir::triton::ascend::ub
