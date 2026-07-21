#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"
#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <string>

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

constexpr StringLiteral kDirectLoadCopy = R"mlir(
module {
  tt.func public @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
)mlir";

std::string replaceOnce(StringRef source, StringRef from, StringRef to) {
  std::string result = source.str();
  size_t position = result.find(from.str());
  EXPECT_NE(position, std::string::npos);
  if (position != std::string::npos)
    result.replace(position, from.size(), to.str());
  return result;
}

std::string replaceAll(StringRef source, StringRef from, StringRef to) {
  std::string result = source.str();
  size_t position = 0;
  while ((position = result.find(from.str(), position)) != std::string::npos) {
    result.replace(position, from.size(), to.str());
    position += to.size();
  }
  return result;
}

class TTIRUBLowerBoundAnalysisTest : public ::testing::Test {
protected:
  TTIRUBLowerBoundAnalysisTest() {
    context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
    registry.addForTesting(std::make_unique<FixedDispositionContract>(
        "preserve", ContractDisposition::Preserve));
  }

  OwningOpRef<ModuleOp> parse(StringRef source = kDirectLoadCopy) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  TTIRUBAnalysisOptions options(StringRef targetArch = "Ascend910B") {
    TTIRUBAnalysisOptions result;
    result.targetArch = targetArch.str();
    result.compileMode = "aiv";
    result.pipelineIdentity = {
        .openSourcePipeline = "synthetic-all-preserve",
        .relevantOptionsJson = "{}",
        .targetArch = targetArch.str(),
        .tritonVersion = "test",
        .cannVersionHash = "test",
        .sha256 = "synthetic-all-preserve-v1",
    };
    result.stages.push_back({.stageName = "preserve"});
    return result;
  }

  TTIRUBAnalysisResult analyze(StringRef source,
                               TTIRUBAnalysisOptions analysisOptions) {
    OwningOpRef<ModuleOp> module = parse(source);
    EXPECT_TRUE(module);
    if (!module)
      return {};
    return analyzeTTIRUBLowerBound(*module, analysisOptions, registry);
  }

  static bool hasReason(const TTIRUBAnalysisResult &result, StringRef reason) {
    return llvm::any_of(result.unsupportedReasons, [&](const std::string &item) {
      return item == reason;
    });
  }

  MLIRContext context;
  PipelineContractRegistry registry;
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

TEST_F(TTIRUBLowerBoundAnalysisTest, DirectLoadOverflowIsRejected) {
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, options());

  EXPECT_EQ(result.lowerBoundBytes, 262144);
  ASSERT_TRUE(result.capacityBytes.has_value());
  EXPECT_EQ(*result.capacityBytes, 192 * 1024);
  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  EXPECT_TRUE(result.unsupportedReasons.empty());
}

TEST_F(TTIRUBLowerBoundAnalysisTest, MaskedLoadDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy, "%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>",
      "%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, "
      "%mask: tensor<65536xi1>");
  source = replaceOnce(
      source,
      "%value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>",
      "%value = tt.load %src_ptrs, %mask : tensor<65536x!tt.ptr<f32>>");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "masked-load"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, DynamicShapeDefersWithNamedReason) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::LoadOp load;
  module->walk([&](triton::LoadOp op) { load = op; });
  ASSERT_TRUE(load);
  auto originalType = cast<RankedTensorType>(load.getType());
  load.getResult().setType(RankedTensorType::get(
      {ShapedType::kDynamic}, originalType.getElementType()));

  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, options(), registry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "dynamic-shape"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       NonContiguousPointerDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy, "%value = tt.load %src_ptrs",
      "%value = tt.load %srcs");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "non-contiguous-pointer"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoadNotReachingStoreDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>\n",
      "");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "load-not-reaching-store"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, NestedRegionDefersWithNamedReason) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::LoadOp load;
  module->walk([&](triton::LoadOp op) { load = op; });
  ASSERT_TRUE(load);

  OperationState nestedState(load.getLoc(), "test.nested");
  nestedState.addRegion();
  Operation *nested = Operation::create(nestedState);
  load->getBlock()->getOperations().insert(load->getIterator(), nested);
  nested->getRegion(0).push_back(new Block());
  load->moveBefore(&nested->getRegion(0).front(),
                   nested->getRegion(0).front().end());

  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, options(), registry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "nested-region"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ReductionUseDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>",
      R"mlir(    %sum = "tt.reduce" (%value) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<65536xf32>) -> f32)mlir");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op-reduction"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       SubByteElementTypeDefersWithNamedReason) {
  std::string source = replaceAll(kDirectLoadCopy, "f32", "i1");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "sub-byte-element-type"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       UnknownPipelineProfileDefersWithNamedReason) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.pipelineIdentity.sha256.clear();

  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, CapacityEqualityDoesNotReject) {
  std::string source = replaceAll(kDirectLoadCopy, "65536", "49152");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.lowerBoundBytes, 192 * 1024);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(result.unsupportedReasons.empty());
}

} // namespace
} // namespace mlir::triton::ascend::ub
