#include "Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h"
#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"
#include "Analysis/TTIRUBLowerBound/VerifierSafety.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace mlir::triton::ascend::ub {
namespace {

class FixedDispositionContract final : public UBResourceContract {
public:
  FixedDispositionContract(StringRef stageName,
                           ContractDisposition disposition,
                           StringRef contractId = "fixed-disposition",
                           StringRef contractVersion = "1")
      : stageName(stageName.str()), disposition(disposition),
        contractId(contractId.str()), contractVersion(contractVersion.str()) {}

  StringRef id() const override { return contractId; }
  StringRef version() const override { return contractVersion; }
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
  std::string contractId;
  std::string contractVersion;
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

constexpr StringLiteral kBinaryAdd = R"mlir(
module {
  tt.func public @add(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %lhs_value = tt.load %lhs_ptrs : tensor<65536x!tt.ptr<f32>>
    %rhs_value = tt.load %rhs_ptrs : tensor<65536x!tt.ptr<f32>>
    %sum = arith.addf %lhs_value, %rhs_value : tensor<65536xf32>
    tt.store %dst_ptrs, %sum : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
)mlir";

constexpr StringLiteral kReshapeCopy = R"mlir(
module {
  tt.func public @reshape_copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %view = tt.reshape %value : tensor<65536xf32> -> tensor<256x256xf32>
    %dst_view = tt.reshape %dst_ptrs : tensor<65536x!tt.ptr<f32>> -> tensor<256x256x!tt.ptr<f32>>
    tt.store %dst_view, %view : tensor<256x256x!tt.ptr<f32>>
    tt.return
  }
}
)mlir";

constexpr StringLiteral kReshapeCopyRoundTrip = R"mlir(
module {
  tt.func public @reshape_copy_round_trip(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %view = tt.reshape %value : tensor<65536xf32> -> tensor<256x256xf32>
    %flat = tt.reshape %view : tensor<256x256xf32> -> tensor<65536xf32>
    tt.store %dst_ptrs, %flat : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
)mlir";

constexpr StringLiteral kReductionSum = R"mlir(
module {
  tt.func public @reduction_sum(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %sum = "tt.reduce" (%value) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<65536xf32>) -> f32
    tt.store %dst, %sum : !tt.ptr<f32>
    tt.return
  }
}
)mlir";

constexpr StringLiteral kLoopCarriedAdd = R"mlir(
module {
  tt.func public @loop_carried_add(%init_src: !tt.ptr<f32>, %step_src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %init_splat = tt.splat %init_src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %init_ptrs = tt.addptr %init_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %step_splat = tt.splat %step_src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %step_ptrs = tt.addptr %step_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %init = tt.load %init_ptrs : tensor<65536x!tt.ptr<f32>>
    %result = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<65536xf32> {
      %step = tt.load %step_ptrs : tensor<65536x!tt.ptr<f32>>
      %next = arith.addf %acc, %step : tensor<65536xf32>
      scf.yield %next : tensor<65536xf32>
    }
    tt.store %dst_ptrs, %result : tensor<65536x!tt.ptr<f32>>
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
    context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                        triton::TritonDialect>();
    TTIRUBAnalysisOptions analysisOptions = options();
    registry.setProfileIdentity(analysisOptions.pipelineIdentity);
    EXPECT_TRUE(succeeded(registry.addProfileContract(
        {.stage = {.stageName = "preserve"},
         .contractId = "fixed-disposition",
         .contractVersion = "1"},
        std::make_unique<FixedDispositionContract>(
            "preserve", ContractDisposition::Preserve))));
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
        .canonicalTtirSha256 = "synthetic-canonical-ttir-v1",
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

  TTIRUBAnalysisResult analyzeModule(
      ModuleOp module, TTIRUBAnalysisOptions analysisOptions) {
    return analyzeTTIRUBLowerBound(module, analysisOptions, registry);
  }

  std::pair<TTIRUBAnalysisResult, unsigned>
  analyzeModuleCapturingDiagnostics(ModuleOp module,
                                    TTIRUBAnalysisOptions analysisOptions) {
    unsigned diagnosticCount = 0;
    TTIRUBAnalysisResult result;
    {
      ScopedDiagnosticHandler handler(
          &context, [&](Diagnostic &) { ++diagnosticCount; });
      result = analyzeTTIRUBLowerBound(module, analysisOptions, registry);
    }
    return {std::move(result), diagnosticCount};
  }

  static bool hasReason(const TTIRUBAnalysisResult &result, StringRef reason) {
    return llvm::any_of(result.unsupportedReasons, [&](const std::string &item) {
      return item == reason;
    });
  }

  MLIRContext context;
  PipelineContractRegistry registry;
};

template <typename OpTy> OpTy findOnlyOp(ModuleOp module) {
  OpTy result;
  module.walk([&](OpTy op) { result = op; });
  EXPECT_TRUE(result);
  return result;
}

Operation *replaceWithMalformedOperation(Operation *original,
                                         TypeRange resultTypes,
                                         bool addRegion = false,
                                         Block *successor = nullptr) {
  OperationState state(original->getLoc(), original->getName().getStringRef());
  state.addOperands(original->getOperands());
  state.addTypes(resultTypes);
  if (addRegion)
    state.addRegion();
  if (successor)
    state.addSuccessors(successor);
  Operation *replacement = Operation::create(state);
  original->getBlock()->getOperations().insert(original->getIterator(),
                                                replacement);
  for (auto [oldResult, newResult] :
       llvm::zip(original->getResults(), replacement->getResults().take_front(
                                             original->getNumResults())))
    oldResult.replaceAllUsesWith(newResult);
  original->erase();
  return replacement;
}

TEST(MandatoryUBResourceGraph, SingletonUsesLargestMandatoryResource) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 64 * 1024, 1});
  MandatoryUBResource largest{"load1", 192 * 1024 + 4, 1};
  largest.contractTrace = {"source", "materialize"};
  graph.addResource(std::move(largest));
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 192 * 1024 + 4);
  EXPECT_EQ(result->resourceIds.size(), 1u);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"source", "materialize"}));
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

TEST(MandatoryUBResourceGraph, ResourceInstancesAreMonotonicLowerBounds) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource({"loop-step", 64, 1});
  ASSERT_TRUE(
      succeeded(graph.raiseResourceInstances(id, 2, "multibuffer-factor-2")));
  EXPECT_EQ(graph.resources()[id].minInstances, 2);
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 128);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"multibuffer-factor-2"}));

  MandatoryUBResourceGraph invalid;
  auto invalidId = invalid.addResource({"loop-step", 64, 2});
  EXPECT_TRUE(
      failed(invalid.raiseResourceInstances(invalidId, 1, "illegal-lowering")));
  EXPECT_TRUE(failed(invalid.solveSingletonLowerBound()));
}

TEST(MandatoryUBResourceGraph, PairwiseOverlapIsNotAThreeWayWitness) {
  MandatoryUBResourceGraph graph;
  auto a = graph.addResource({"a", 32, 1});
  MandatoryUBResource bResource{"b", 64, 1};
  bResource.contractTrace = {"b-source"};
  auto b = graph.addResource(std::move(bResource));
  MandatoryUBResource cResource{"c", 128, 1};
  cResource.contractTrace = {"c-source"};
  auto c = graph.addResource(std::move(cResource));
  graph.addMustDistinct(a, b);
  graph.addMustDistinct(b, c);
  graph.addMustDistinct(a, c);
  graph.addWitness({a, b});
  CoexistenceWitness witness;
  witness.resources = {b, c};
  witness.contractTrace = {"lifetime-overlap"};
  graph.addWitness(std::move(witness));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 192);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"lifetime-overlap", "b-source",
                                      "c-source"}));
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
     MaterializationCanRefineMayAliasWitnessToMustDistinct) {
  MandatoryUBResourceGraph graph;
  MandatoryUBResource lhs{"lhs", 64, 1};
  lhs.contractTrace = {"binary-elementwise-source"};
  auto lhsId = graph.addResource(std::move(lhs));
  MandatoryUBResource rhs{"rhs", 128, 1};
  rhs.contractTrace = {"binary-elementwise-source"};
  auto rhsId = graph.addResource(std::move(rhs));
  graph.addMayAlias(lhsId, rhsId);
  CoexistenceWitness witness;
  witness.resources = {lhsId, rhsId};
  witness.contractTrace = {"binary-elementwise-source"};
  WitnessId witnessId = graph.addWitness(std::move(witness));

  ASSERT_FALSE(graph.hasPairwiseDistinctWitness(witnessId));
  ASSERT_TRUE(succeeded(graph.refineWitnessToMustDistinct(
      witnessId, "binary-elementwise-materialize")));
  ASSERT_TRUE(graph.hasPairwiseDistinctWitness(witnessId));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->kind, "witness");
  EXPECT_EQ(result->bytes, 192);
  EXPECT_EQ(result->resourceIds,
            SmallVector<ResourceId>({lhsId, rhsId}));
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"binary-elementwise-source",
                                      "binary-elementwise-materialize"}));
}

TEST(MandatoryUBResourceGraph,
     DistinctRefinementRequiresExplicitMayAliasWitnessFacts) {
  MandatoryUBResourceGraph graph;
  auto lhs = graph.addResource({"lhs", 64, 1});
  auto rhs = graph.addResource({"rhs", 128, 1});
  WitnessId witness = graph.addWitness({lhs, rhs});

  EXPECT_TRUE(failed(
      graph.refineWitnessToMustDistinct(witness, "unproven-distinct")));
  EXPECT_TRUE(failed(graph.solveWitnessLowerBound()));
}

TEST(MandatoryUBResourceGraph,
     MaterializationCanRefineMayAliasToMustAlias) {
  MandatoryUBResourceGraph graph;
  auto source = graph.addResource({"source", 128, 1});
  auto view = graph.addResource({"view", 128, 1});
  graph.addMayAlias(source, view);

  ASSERT_TRUE(graph.hasMayAlias(source, view));
  ASSERT_TRUE(succeeded(graph.refineMayAliasToMustAlias(
      source, view, "view-materialize")));
  EXPECT_FALSE(graph.hasMayAlias(source, view));
  EXPECT_TRUE(graph.hasMustAlias(source, view));
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 128);
}

TEST(MandatoryUBResourceGraph,
     AliasRefinementRequiresExplicitMayAliasFact) {
  MandatoryUBResourceGraph graph;
  auto source = graph.addResource({"source", 128, 1});
  auto view = graph.addResource({"view", 128, 1});

  EXPECT_TRUE(failed(graph.refineMayAliasToMustAlias(
      source, view, "unproven-alias")));
  EXPECT_TRUE(failed(graph.solveWitnessLowerBound()));
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

MandatoryUBResource directCopyResource(int64_t payloadBytes = 262144) {
  MandatoryUBResource resource{"load0", payloadBytes, 1};
  resource.origin = "tt.load";
  resource.kind = MaterializationKind::GMToUBLoad;
  resource.sourceElements = 65536;
  resource.elementBitWidth = 32;
  resource.consumer = "tt.store";
  return resource;
}

TEST(UBResourceContract, DirectCopyPreserveRequiresExactSourceFacts) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource(directCopyResource());
  PipelineContractRegistry registry;
  registry.addForTesting(makeDirectCopyPreserveContract(
      {.stageName = "canonicalize"}, 1, 65536, 32, 262144));

  EXPECT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "canonicalize"})));
  ASSERT_EQ(graph.resources()[id].validity, ValidityState::Valid);
  ASSERT_EQ(graph.resources()[id].contractTrace.size(), 1u);
  EXPECT_EQ(graph.resources()[id].contractTrace.back(),
            "direct-copy-preserve");
}

TEST(UBResourceContract, DirectCopyPreserveInvalidatesShapeDrift) {
  MandatoryUBResourceGraph graph;
  auto resource = directCopyResource();
  resource.sourceElements = 32768;
  auto id = graph.addResource(std::move(resource));
  PipelineContractRegistry registry;
  registry.addForTesting(makeDirectCopyPreserveContract(
      {.stageName = "canonicalize"}, 1, 65536, 32, 262144));

  EXPECT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "canonicalize"})));
  EXPECT_EQ(graph.resources()[id].validity, ValidityState::Invalid);
  EXPECT_EQ(graph.resources()[id].invalidReason, "direct-copy-preserve");
}

TEST(UBResourceContract, DirectCopyMaxTilesProducesCeilingLowerBound) {
  MandatoryUBResourceGraph graph;
  auto resource = directCopyResource(262145);
  auto id = graph.addResource(std::move(resource));
  PipelineContractRegistry registry;
  registry.addForTesting(makeDirectCopyMaxTilesContract(
      {.stageName = "materialize"}, 1, 65536, 32, 262145, 64));

  EXPECT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "materialize"})));
  EXPECT_EQ(graph.resources()[id].minPayloadBytes, 4097);
  EXPECT_EQ(graph.resources()[id].contractTrace.back(),
            "direct-copy-max-tiles");
}

TEST(UBResourceContract, DirectCopyMaxTilesRejectsResourceCountDrift) {
  MandatoryUBResourceGraph graph;
  auto first = graph.addResource(directCopyResource());
  graph.addResource(directCopyResource());
  PipelineContractRegistry registry;
  registry.addForTesting(makeDirectCopyMaxTilesContract(
      {.stageName = "materialize"}, 1, 65536, 32, 262144, 64));

  EXPECT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "materialize"})));
  EXPECT_EQ(graph.resources()[first].validity, ValidityState::Invalid);
  EXPECT_EQ(graph.resources()[first].invalidReason,
            "direct-copy-max-tiles");
}

MandatoryUBResource binaryAddResource(int64_t payloadBytes = 262144,
                                      uint64_t birth = 1) {
  MandatoryUBResource resource{"binary-input", payloadBytes, 1};
  resource.origin = "tt.load";
  resource.kind = MaterializationKind::GMToUBLoad;
  resource.sourceElements = 65536;
  resource.elementBitWidth = 32;
  resource.consumer = "arith.addf";
  resource.birth.ordinal = birth;
  resource.lastRequiredUse.ordinal = 3;
  resource.contractTrace = {"ttir-binary-add-v1"};
  return resource;
}

TEST(UBResourceContract, BinaryAddMaxTilesProvesDistinctCoexistence) {
  MandatoryUBResourceGraph graph;
  ResourceId lhs = graph.addResource(binaryAddResource());
  ResourceId rhs = graph.addResource(binaryAddResource(262144, 2));
  graph.addMayAlias(lhs, rhs);
  CoexistenceWitness witness;
  witness.resources = {lhs, rhs};
  witness.contractTrace = {"ttir-binary-add-v1"};
  graph.addWitness(std::move(witness));
  PipelineContractRegistry registry;
  registry.addForTesting(makeBinaryAddMaxTilesContract(
      {.stageName = "materialize"}, 2, 65536, 32, 262144, 64));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "materialize"})));
  EXPECT_EQ(graph.resources()[lhs].minPayloadBytes, 4096);
  EXPECT_EQ(graph.resources()[rhs].minPayloadBytes, 4096);
  ASSERT_TRUE(graph.hasPairwiseDistinctWitness(0));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->kind, "witness");
  EXPECT_EQ(result->bytes, 8192);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"ttir-binary-add-v1",
                                      "binary-add-max-tiles"}));
}

TEST(UBResourceContract, BinaryAddPreserveKeepsPreMaterializationMayAlias) {
  MandatoryUBResourceGraph graph;
  ResourceId lhs = graph.addResource(binaryAddResource(4096));
  ResourceId rhs = graph.addResource(binaryAddResource(4096, 2));
  graph.addMayAlias(lhs, rhs);
  graph.addWitness({lhs, rhs});
  PipelineContractRegistry registry;
  registry.addForTesting(makeBinaryAddPreserveContract(
      {.stageName = "suffix"}, 2, 65536, 32, 4096));

  ASSERT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "suffix"})));
  EXPECT_EQ(graph.resources()[lhs].validity, ValidityState::Valid);
  EXPECT_EQ(graph.resources()[rhs].validity, ValidityState::Valid);
  EXPECT_TRUE(graph.hasPairwiseMayAliasWitness(0));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->kind, "singleton");
  EXPECT_EQ(result->bytes, 4096);
}

TEST(UBResourceContract, BinaryAddContractRejectsLifetimeDrift) {
  MandatoryUBResourceGraph graph;
  ResourceId lhs = graph.addResource(binaryAddResource());
  ResourceId rhs = graph.addResource(binaryAddResource());
  graph.addMayAlias(lhs, rhs);
  CoexistenceWitness witness;
  witness.resources = {lhs, rhs};
  witness.contractTrace = {"ttir-binary-add-v1"};
  graph.addWitness(std::move(witness));
  PipelineContractRegistry registry;
  registry.addForTesting(makeBinaryAddMaxTilesContract(
      {.stageName = "materialize"}, 2, 65536, 32, 262144, 64));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "materialize"})));
  EXPECT_EQ(graph.resources()[lhs].validity, ValidityState::Invalid);
  EXPECT_EQ(graph.resources()[rhs].validity, ValidityState::Invalid);
}

std::pair<ResourceId, ResourceId>
addReshapeCopyResources(MandatoryUBResourceGraph &graph,
                        int64_t payloadBytes = 262144,
                        StringRef viewConsumer = "tt.store") {
  MandatoryUBResource source{"reshape-source", payloadBytes, 1};
  source.origin = "tt.load";
  source.kind = MaterializationKind::GMToUBLoad;
  source.sourceElements = 65536;
  source.elementBitWidth = 32;
  source.consumer = "tt.reshape";
  source.contractTrace = {"ttir-reshape-copy-v1"};
  ResourceId sourceId = graph.addResource(std::move(source));

  MandatoryUBResource view{"reshape-view", payloadBytes, 1};
  view.origin = "tt.reshape";
  view.kind = MaterializationKind::ViewAlias;
  view.sourceElements = 65536;
  view.elementBitWidth = 32;
  view.consumer = viewConsumer.str();
  view.contractTrace = {"ttir-reshape-copy-v1"};
  ResourceId viewId = graph.addResource(std::move(view));
  graph.addMayAlias(sourceId, viewId);
  return {sourceId, viewId};
}

TEST(UBResourceContract, ReshapeCopyMaxTilesProvesSingleAllocationAlias) {
  MandatoryUBResourceGraph graph;
  auto [source, view] = addReshapeCopyResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeReshapeCopyMaxTilesContract(
      {.stageName = "materialize"}, 2, 65536, 32, 262144, 64));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "materialize"})));
  EXPECT_TRUE(graph.hasMustAlias(source, view));
  EXPECT_EQ(graph.resources()[source].minPayloadBytes, 4096);
  EXPECT_EQ(graph.resources()[view].minPayloadBytes, 4096);
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 4096);
}

TEST(UBResourceContract, ReshapeCopyMaxTilesAcceptsStrictInverseView) {
  MandatoryUBResourceGraph graph;
  auto [source, view] =
      addReshapeCopyResources(graph, 262144, "tt.reshape");
  PipelineContractRegistry registry;
  registry.addForTesting(makeReshapeCopyMaxTilesContract(
      {.stageName = "materialize"}, 2, 65536, 32, 262144, 64));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "materialize"})));
  EXPECT_TRUE(graph.hasMustAlias(source, view));
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 4096);
}

TEST(UBResourceContract, ReshapeCopyPreserveKeepsPreMaterializationMayAlias) {
  MandatoryUBResourceGraph graph;
  auto [source, view] = addReshapeCopyResources(graph, 4096);
  PipelineContractRegistry registry;
  registry.addForTesting(makeReshapeCopyPreserveContract(
      {.stageName = "suffix"}, 2, 65536, 32, 4096));

  ASSERT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "suffix"})));
  EXPECT_EQ(graph.resources()[source].validity, ValidityState::Valid);
  EXPECT_EQ(graph.resources()[view].validity, ValidityState::Valid);
  EXPECT_TRUE(graph.hasMayAlias(source, view));
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 4096);
}

void addReductionSumResources(MandatoryUBResourceGraph &graph) {
  MandatoryUBResource input{"reduction-input", 262144, 1};
  input.origin = "tt.load";
  input.kind = MaterializationKind::GMToUBLoad;
  input.birth.ordinal = 1;
  input.lastRequiredUse.ordinal = 3;
  input.sourceElements = 65536;
  input.elementBitWidth = 32;
  input.consumer = "tt.reduce";
  input.contractTrace = {"ttir-reduction-sum-v1"};
  ResourceId inputId = graph.addResource(std::move(input));

  MandatoryUBResource scratch{"reduction-scratch", 131072, 1};
  scratch.origin = "tt.reduce";
  scratch.kind = MaterializationKind::ReductionScratch;
  scratch.birth.ordinal = 3;
  scratch.lastRequiredUse.ordinal = 3;
  scratch.sourceElements = 65536;
  scratch.elementBitWidth = 32;
  scratch.consumer = "tt.reduce";
  scratch.contractTrace = {"ttir-reduction-sum-v1"};
  ResourceId scratchId = graph.addResource(std::move(scratch));

  MandatoryUBResource accumulator{"reduction-accumulator", 4, 1};
  accumulator.origin = "tt.reduce";
  accumulator.kind = MaterializationKind::ReductionAccumulator;
  accumulator.birth.ordinal = 3;
  accumulator.lastRequiredUse.ordinal = 4;
  accumulator.sourceElements = 65536;
  accumulator.elementBitWidth = 32;
  accumulator.consumer = "tt.store";
  accumulator.contractTrace = {"ttir-reduction-sum-v1"};
  ResourceId accumulatorId = graph.addResource(std::move(accumulator));

  for (auto [lhs, rhs] :
       {std::pair{inputId, scratchId}, std::pair{inputId, accumulatorId},
        std::pair{scratchId, accumulatorId}})
    graph.addMayAlias(lhs, rhs);
  CoexistenceWitness witness;
  witness.resources = {inputId, scratchId, accumulatorId};
  witness.contractTrace = {"ttir-reduction-sum-v1"};
  graph.addWitness(std::move(witness));
}

TEST(UBResourceContract, ReductionSumExtraBufferProvesThreeWayPeak) {
  MandatoryUBResourceGraph graph;
  addReductionSumResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeReductionSumExtraBufferContract(
      {.stageName = "suffix"}, 3, 65536, 32, 262144, 131072, 4));

  ASSERT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "suffix"})));
  EXPECT_TRUE(graph.hasPairwiseDistinctWitness(0));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->kind, "witness");
  EXPECT_EQ(result->bytes, 393220);
  EXPECT_EQ(result->resourceIds, SmallVector<ResourceId>({0, 1, 2}));
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"ttir-reduction-sum-v1",
                                      "reduction-sum-extra-buffer"}));
}

TEST(UBResourceContract, ReductionSumPreserveCannotInventDistinctness) {
  MandatoryUBResourceGraph graph;
  addReductionSumResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeReductionSumPreserveContract(
      {.stageName = "preserve"}, 3, 65536, 32, 262144, 131072, 4));

  ASSERT_TRUE(succeeded(
      registry.applyOrInvalidateAll(graph, {.stageName = "preserve"})));
  EXPECT_TRUE(graph.hasPairwiseMayAliasWitness(0));
  EXPECT_EQ(graph.solveWitnessLowerBound()->kind, "singleton");
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 262144);
}

std::pair<ResourceId, ResourceId>
addDynamicCVResources(MandatoryUBResourceGraph &graph) {
  MandatoryUBResource fixpipe{"dynamic-cv-fixpipe-output", 1024, 1};
  fixpipe.origin = "tt.dot";
  fixpipe.kind = MaterializationKind::DynamicCVFixpipeOutput;
  fixpipe.birth.ordinal = 20;
  fixpipe.lastRequiredUse.ordinal = 24;
  fixpipe.sourceElements = 256;
  fixpipe.elementBitWidth = 32;
  fixpipe.consumer = "math.exp";
  fixpipe.contractTrace = {"ttir-dynamic-cv-dot-exp-v1"};
  ResourceId fixpipeId = graph.addResource(std::move(fixpipe));

  MandatoryUBResource vector{"dynamic-cv-vector-output", 1024, 1};
  vector.origin = "math.exp";
  vector.kind = MaterializationKind::DynamicCVVectorOutput;
  vector.birth.ordinal = 24;
  vector.lastRequiredUse.ordinal = 25;
  vector.sourceElements = 256;
  vector.elementBitWidth = 32;
  vector.consumer = "tt.store";
  vector.contractTrace = {"ttir-dynamic-cv-dot-exp-v1"};
  ResourceId vectorId = graph.addResource(std::move(vector));

  graph.addMayAlias(fixpipeId, vectorId);
  CoexistenceWitness witness;
  witness.resources = {fixpipeId, vectorId};
  witness.contractTrace = {"ttir-dynamic-cv-dot-exp-v1"};
  graph.addWitness(std::move(witness));
  return {fixpipeId, vectorId};
}

TEST(UBResourceContract, DynamicCVReplayTransfersObservedPhysicalFacts) {
  MandatoryUBResourceGraph graph;
  auto [fixpipe, vector] = addDynamicCVResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeDynamicCVReplayContract(
      {.stageName = "ttir.dynamic-cv-pipeline"}, 2, 256, 32, 1024,
      /*projectedPayloadBytes=*/512, /*fixpipeMinInstances=*/2,
      /*vectorMinInstances=*/3));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "ttir.dynamic-cv-pipeline"})));
  EXPECT_EQ(graph.resources()[fixpipe].minPayloadBytes, 512);
  EXPECT_EQ(graph.resources()[fixpipe].minInstances, 2);
  EXPECT_EQ(graph.resources()[vector].minPayloadBytes, 512);
  EXPECT_EQ(graph.resources()[vector].minInstances, 3);
  EXPECT_TRUE(graph.hasPairwiseDistinctWitness(0));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 2560);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({"ttir-dynamic-cv-dot-exp-v1",
                                      "dynamic-cv-replay"}));
}

TEST(UBResourceContract, DynamicCVReplayInvalidatesLifetimeDrift) {
  MandatoryUBResourceGraph graph;
  auto [fixpipe, vector] = addDynamicCVResources(graph);
  // Rebuild with a gap between the CUBE result's last use and the VECTOR
  // result's birth.  The replay contract may only assert coexistence for the
  // exact adjacent boundary observed by the oracle.
  MandatoryUBResourceGraph drifted;
  MandatoryUBResource first = graph.resources()[fixpipe];
  MandatoryUBResource second = graph.resources()[vector];
  second.birth.ordinal += 1;
  ResourceId firstId = drifted.addResource(std::move(first));
  ResourceId secondId = drifted.addResource(std::move(second));
  drifted.addMayAlias(firstId, secondId);
  drifted.addWitness({firstId, secondId});

  PipelineContractRegistry registry;
  registry.addForTesting(makeDynamicCVReplayContract(
      {.stageName = "ttir.dynamic-cv-pipeline"}, 2, 256, 32, 1024, 1024, 1,
      1));
  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      drifted, {.stageName = "ttir.dynamic-cv-pipeline"})));
  EXPECT_EQ(drifted.resources()[firstId].validity, ValidityState::Invalid);
  EXPECT_EQ(drifted.resources()[secondId].validity, ValidityState::Invalid);
}

TEST(UBResourceContract, DynamicCVContractsPreserveARealStageChain) {
  MandatoryUBResourceGraph graph;
  addDynamicCVResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeDynamicCVSourcePreserveContract(
      {.stageName = "ttir.auto-blockify"}, 2, 256, 32, 1024, 512, 2, 3));
  registry.addForTesting(makeDynamicCVReplayContract(
      {.stageName = "ttir.dynamic-cv-pipeline"}, 2, 256, 32, 1024, 512, 2,
      3));
  registry.addForTesting(makeDynamicCVResultPreserveContract(
      {.stageName = "bisheng.ub-affecting-suffix"}, 2, 256, 32, 1024, 512, 2,
      3));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "ttir.auto-blockify"})));
  EXPECT_TRUE(graph.hasPairwiseMayAliasWitness(0));
  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "ttir.dynamic-cv-pipeline"})));
  EXPECT_TRUE(graph.hasPairwiseDistinctWitness(0));
  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "bisheng.ub-affecting-suffix"})));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 2560);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({
                "ttir-dynamic-cv-dot-exp-v1",
                "dynamic-cv-source-preserve",
                "dynamic-cv-replay",
                "dynamic-cv-result-preserve",
            }));
}

std::pair<ResourceId, ResourceId>
addIrregularMemoryResources(MandatoryUBResourceGraph &graph) {
  MandatoryUBResource index{"irregular-index", 64, 1};
  index.origin = "tt.load";
  index.kind = MaterializationKind::IrregularIndex;
  index.birth.ordinal = 3;
  index.lastRequiredUse.ordinal = 6;
  index.sourceElements = 8;
  index.elementBitWidth = 64;
  index.consumer = "tt.addptr";
  index.contractTrace = {"ttir-irregular-indirect-add-v1"};
  ResourceId indexId = graph.addResource(std::move(index));

  MandatoryUBResource value{"irregular-gather", 32, 1};
  value.origin = "tt.load";
  value.kind = MaterializationKind::IrregularGather;
  value.birth.ordinal = 6;
  value.lastRequiredUse.ordinal = 10;
  value.sourceElements = 8;
  value.elementBitWidth = 32;
  value.consumer = "arith.addf";
  value.contractTrace = {"ttir-irregular-indirect-add-v1"};
  ResourceId valueId = graph.addResource(std::move(value));

  graph.addMayAlias(indexId, valueId);
  CoexistenceWitness witness;
  witness.resources = {indexId, valueId};
  witness.contractTrace = {"ttir-irregular-indirect-add-v1"};
  graph.addWitness(std::move(witness));
  return {indexId, valueId};
}

TEST(UBResourceContract, IrregularReplayUsesCeilingTileProjection) {
  MandatoryUBResourceGraph graph;
  auto [index, value] = addIrregularMemoryResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeIrregularMemoryReplayContract(
      {.stageName = "ttir.triton-to-linalg"}, 2, 8, 64, 32, 64, 32,
      /*maxTiles=*/3));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "ttir.triton-to-linalg"})));
  EXPECT_EQ(graph.resources()[index].minPayloadBytes, 22);
  EXPECT_EQ(graph.resources()[value].minPayloadBytes, 11);
  EXPECT_TRUE(graph.hasPairwiseDistinctWitness(0));
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 33);
}

TEST(UBResourceContract, IrregularReplayInvalidatesResourceKindDrift) {
  MandatoryUBResourceGraph graph;
  auto [index, value] = addIrregularMemoryResources(graph);
  (void)value;
  PipelineContractRegistry registry;
  registry.addForTesting(makeIrregularMemoryReplayContract(
      {.stageName = "ttir.triton-to-linalg"}, 2, 8, 64, 32, 64, 32, 1));

  // The contract must not reinterpret a regular GM load as an index buffer.
  auto regular = graph.resources()[index];
  MandatoryUBResourceGraph drifted;
  regular.kind = MaterializationKind::GMToUBLoad;
  ResourceId first = drifted.addResource(std::move(regular));
  ResourceId second = drifted.addResource(graph.resources()[1]);
  drifted.addMayAlias(first, second);
  drifted.addWitness({first, second});
  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      drifted, {.stageName = "ttir.triton-to-linalg"})));
  EXPECT_EQ(drifted.resources()[first].validity, ValidityState::Invalid);
  EXPECT_EQ(drifted.resources()[second].validity, ValidityState::Invalid);
}

TEST(UBResourceContract, IrregularContractsPreserveARealStageChain) {
  MandatoryUBResourceGraph graph;
  addIrregularMemoryResources(graph);
  PipelineContractRegistry registry;
  registry.addForTesting(makeIrregularMemorySourcePreserveContract(
      {.stageName = "ttir.auto-blockify"}, 2, 8, 64, 32, 64, 32, 3));
  registry.addForTesting(makeIrregularMemoryReplayContract(
      {.stageName = "ttir.triton-to-linalg"}, 2, 8, 64, 32, 64, 32, 3));
  registry.addForTesting(makeIrregularMemoryResultPreserveContract(
      {.stageName = "bisheng.ub-affecting-suffix"}, 2, 8, 64, 32, 64, 32,
      3));

  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "ttir.auto-blockify"})));
  EXPECT_TRUE(graph.hasPairwiseMayAliasWitness(0));
  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "ttir.triton-to-linalg"})));
  EXPECT_TRUE(graph.hasPairwiseDistinctWitness(0));
  ASSERT_TRUE(succeeded(registry.applyOrInvalidateAll(
      graph, {.stageName = "bisheng.ub-affecting-suffix"})));
  auto result = graph.solveWitnessLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 33);
  EXPECT_EQ(result->contractTrace,
            SmallVector<std::string>({
                "ttir-irregular-indirect-add-v1",
                "irregular-memory-source-preserve",
                "irregular-memory-replay",
                "irregular-memory-result-preserve",
            }));
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
  EXPECT_EQ(graph.resources()[0].contractTrace,
            SmallVector<std::string>({"fixed-disposition"}));
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
  auto expectCapacity = [](StringRef arch, int64_t expected) {
    std::optional<int64_t> capacity = getUBCapacityBytes(arch);
    ASSERT_TRUE(capacity.has_value()) << arch.str();
    EXPECT_EQ(*capacity, expected) << arch.str();
  };

  for (const char *arch : {"Ascend910B", "Ascend910_93", "Ascend910B1",
                           "Ascend910B2", "Ascend910B3", "Ascend910B4",
                           "Ascend910_9362", "Ascend910_9372",
                           "Ascend910_9381", "Ascend910_9382",
                           "Ascend910_9391", "Ascend910_9392"})
    expectCapacity(arch, 192 * 1024);

  for (const char *arch : {"Ascend310B1", "Ascend310B2", "Ascend310B3",
                           "Ascend310B4"})
    expectCapacity(arch, 248 * 1024);

  for (const char *arch : {"Ascend910_95", "Ascend950", "Ascend910_9579",
                           "Ascend910_9581", "Ascend910_9589",
                           "Ascend910_9599"})
    expectCapacity(arch, 256 * 1024);

  for (const char *arch : {"future-chip", "Ascend910B-future",
                           "Ascend910BLAH", "Ascend910_93future",
                           "Ascend910_95future", "Ascend950Future",
                           "Ascend310B", "Ascend310B5"})
    EXPECT_FALSE(getUBCapacityBytes(arch).has_value()) << arch;
}

TEST(VerifierSafety, ShortLoadSegmentVectorIsRejected) {
  EXPECT_FALSE(detail::hasValidLoadOperandSegments({1, 0}, 1));
}

TEST(VerifierSafety, LongLoadSegmentVectorIsRejected) {
  EXPECT_FALSE(detail::hasValidLoadOperandSegments({1, 0, 0, 0}, 1));
}

TEST(VerifierSafety, ShortStoreSegmentVectorIsRejected) {
  EXPECT_FALSE(detail::hasValidStoreOperandSegments({1, 1}, 2));
}

TEST(VerifierSafety, LongStoreSegmentVectorIsRejected) {
  EXPECT_FALSE(detail::hasValidStoreOperandSegments({1, 1, 0, 0}, 2));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, DirectLoadOverflowIsRejected) {
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, options());

  EXPECT_EQ(result.lowerBoundBytes, 262144);
  ASSERT_TRUE(result.capacityBytes.has_value());
  EXPECT_EQ(*result.capacityBytes, 192 * 1024);
  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  EXPECT_TRUE(result.unsupportedReasons.empty());
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       BinaryAddUsesDistinctCoexistenceWitnessAfterMaterialization) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages = {{.stageName = "materialize"}};
  PipelineContractRegistry binaryRegistry;
  binaryRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(binaryRegistry.addProfileContract(
      {.stage = analysisOptions.stages.front(),
       .contractId = "binary-add-max-tiles",
       .contractVersion = "1"},
      makeBinaryAddMaxTilesContract(analysisOptions.stages.front(), 2, 65536,
                                    32, 262144, 2))));
  OwningOpRef<ModuleOp> module = parse(kBinaryAdd);
  ASSERT_TRUE(module);

  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, binaryRegistry);
  EXPECT_EQ(result.lowerBoundBytes, 262144);
  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "witness");
  EXPECT_EQ(result.certificates[0].resourceIds,
            SmallVector<ResourceId>({0, 1}));
  EXPECT_EQ(result.certificates[0].contractTrace,
            SmallVector<std::string>({"ttir-binary-add-v1",
                                      "binary-add-max-tiles"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       BinaryAddProfilePreservesProofBeforeAndAfterMaterialization) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages = {{.stageName = "source"},
                            {.stageName = "materialize"},
                            {.stageName = "suffix"}};
  PipelineContractRegistry registry;
  registry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(registry.addProfileContract(
      {.stage = analysisOptions.stages[0],
       .contractId = "binary-add-preserve",
       .contractVersion = "1"},
      makeBinaryAddPreserveContract(analysisOptions.stages[0], 2, 65536, 32,
                                    262144))));
  ASSERT_TRUE(succeeded(registry.addProfileContract(
      {.stage = analysisOptions.stages[1],
       .contractId = "binary-add-max-tiles",
       .contractVersion = "1"},
      makeBinaryAddMaxTilesContract(analysisOptions.stages[1], 2, 65536, 32,
                                    262144, 1))));
  ASSERT_TRUE(succeeded(registry.addProfileContract(
      {.stage = analysisOptions.stages[2],
       .contractId = "binary-add-preserve",
       .contractVersion = "1"},
      makeBinaryAddPreserveContract(analysisOptions.stages[2], 2, 65536, 32,
                                    262144))));

  OwningOpRef<ModuleOp> module = parse(kBinaryAdd);
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, analysisOptions, registry);

  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  EXPECT_EQ(result.lowerBoundBytes, 524288);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "witness");
  EXPECT_EQ(result.certificates[0].contractTrace,
            SmallVector<std::string>({"ttir-binary-add-v1",
                                      "binary-add-preserve",
                                      "binary-add-max-tiles",
                                      "binary-add-preserve"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       BinaryAddWithoutDistinctContractCannotSumInputs) {
  TTIRUBAnalysisResult result = analyze(kBinaryAdd, options());

  EXPECT_EQ(result.lowerBoundBytes, 262144);
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "singleton");
  EXPECT_EQ(result.certificates[0].resourceIds.size(), 1u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoopCarriedAddProvesAccumulatorAndStepInputCoexist) {
  TTIRUBAnalysisOptions analysisOptions = options("Ascend910B1");
  analysisOptions.stages = {{.stageName = "source"},
                            {.stageName = "materialize"},
                            {.stageName = "suffix"}};
  PipelineContractRegistry loopRegistry;
  loopRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(loopRegistry.addProfileContract(
      {.stage = analysisOptions.stages[0],
       .contractId = "loop-carried-add-preserve",
       .contractVersion = "1"},
      makeLoopCarriedAddPreserveContract(analysisOptions.stages[0], 2, 65536,
                                         32, 262144))));
  ASSERT_TRUE(succeeded(loopRegistry.addProfileContract(
      {.stage = analysisOptions.stages[1],
       .contractId = "loop-carried-add-max-tiles",
       .contractVersion = "1"},
      makeLoopCarriedAddMaxTilesContract(analysisOptions.stages[1], 2, 65536,
                                         32, 262144, 1))));
  ASSERT_TRUE(succeeded(loopRegistry.addProfileContract(
      {.stage = analysisOptions.stages[2],
       .contractId = "loop-carried-add-preserve",
       .contractVersion = "1"},
      makeLoopCarriedAddPreserveContract(analysisOptions.stages[2], 2, 65536,
                                         32, 262144))));
  OwningOpRef<ModuleOp> module = parse(kLoopCarriedAdd);
  ASSERT_TRUE(module);

  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, analysisOptions, loopRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  EXPECT_EQ(result.lowerBoundBytes, 524288);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "witness");
  EXPECT_EQ(result.certificates[0].resourceIds,
            SmallVector<ResourceId>({0, 1}));
  EXPECT_EQ(
      result.certificates[0].contractTrace,
      SmallVector<std::string>({"ttir-loop-carried-add-v1",
                                "loop-carried-add-preserve",
                                "loop-carried-add-max-tiles",
                                "loop-carried-add-preserve"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoopCarriedAddAcceptsAnyStaticPositiveTripCountAtLeastTwo) {
  std::string source =
      replaceOnce(kLoopCarriedAdd, "%c2 = arith.constant 2 : index",
                  "%c2 = arith.constant 3 : index");
  TTIRUBAnalysisResult result = analyze(source, options());

  EXPECT_EQ(result.lowerBoundBytes, 262144);
  EXPECT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "singleton");
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoopCarriedAddRejectsSingleIterationForMultiBufferProof) {
  std::string source =
      replaceOnce(kLoopCarriedAdd, "%c2 = arith.constant 2 : index",
                  "%c2 = arith.constant 1 : index");
  TTIRUBAnalysisResult result = analyze(source, options());

  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-loop-trip-count"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoopCarriedAddRejectsNonPositiveStep) {
  std::string source =
      replaceOnce(kLoopCarriedAdd, "%c1 = arith.constant 1 : index",
                  "%c1 = arith.constant 0 : index");
  TTIRUBAnalysisResult result = analyze(source, options());

  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-loop-bounds"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoopCarriedAddMultiBufferRaisesStepInputInstances) {
  TTIRUBAnalysisOptions analysisOptions = options("Ascend910B1");
  analysisOptions.stages = {{.stageName = "materialize"},
                            {.stageName = "suffix"}};
  PipelineContractRegistry loopRegistry;
  loopRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(loopRegistry.addProfileContract(
      {.stage = analysisOptions.stages[0],
       .contractId = "loop-carried-add-max-tiles",
       .contractVersion = "1"},
      makeLoopCarriedAddMaxTilesContract(analysisOptions.stages[0], 2, 65536,
                                         32, 262144, 1))));
  ASSERT_TRUE(succeeded(loopRegistry.addProfileContract(
      {.stage = analysisOptions.stages[1],
       .contractId = "loop-carried-add-multibuffer",
       .contractVersion = "1"},
      makeLoopCarriedAddMultiBufferContract(
          analysisOptions.stages[1], 2, 65536, 32, 262144, 2))));
  OwningOpRef<ModuleOp> module = parse(kLoopCarriedAdd);
  ASSERT_TRUE(module);

  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, analysisOptions, loopRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  EXPECT_EQ(result.lowerBoundBytes, 786432);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].resourceIds,
            SmallVector<ResourceId>({0, 1}));
  EXPECT_EQ(
      result.certificates[0].contractTrace,
      SmallVector<std::string>({"ttir-loop-carried-add-v1",
                                "loop-carried-add-max-tiles",
                                "loop-carried-add-multibuffer"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ReshapeCopyUsesOneAliasClassAfterMaterialization) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages = {{.stageName = "materialize"}};
  PipelineContractRegistry reshapeRegistry;
  reshapeRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(reshapeRegistry.addProfileContract(
      {.stage = analysisOptions.stages.front(),
       .contractId = "reshape-copy-max-tiles",
       .contractVersion = "1"},
      makeReshapeCopyMaxTilesContract(analysisOptions.stages.front(), 2,
                                      65536, 32, 262144, 64))));
  OwningOpRef<ModuleOp> module = parse(kReshapeCopy);
  ASSERT_TRUE(module);

  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, reshapeRegistry);
  EXPECT_EQ(result.lowerBoundBytes, 4096);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "singleton");
  EXPECT_EQ(result.certificates[0].resourceIds.size(), 1u);
  EXPECT_EQ(result.certificates[0].contractTrace,
            SmallVector<std::string>({"ttir-reshape-copy-v1",
                                      "reshape-copy-max-tiles"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ReshapeCopyRoundTripUsesOneAliasClassWithoutPointerReshape) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages = {{.stageName = "materialize"}};
  PipelineContractRegistry reshapeRegistry;
  reshapeRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(reshapeRegistry.addProfileContract(
      {.stage = analysisOptions.stages.front(),
       .contractId = "reshape-copy-max-tiles",
       .contractVersion = "1"},
      makeReshapeCopyMaxTilesContract(analysisOptions.stages.front(), 2,
                                      65536, 32, 262144, 64))));
  OwningOpRef<ModuleOp> module = parse(kReshapeCopyRoundTrip);
  ASSERT_TRUE(module);

  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, reshapeRegistry);
  EXPECT_EQ(result.lowerBoundBytes, 4096);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "singleton");
  EXPECT_EQ(result.certificates[0].resourceIds.size(), 1u);
  EXPECT_EQ(result.certificates[0].contractTrace,
            SmallVector<std::string>({"ttir-reshape-copy-v1",
                                      "reshape-copy-max-tiles"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ReductionSumUsesInputScratchAndAccumulatorWitness) {
  TTIRUBAnalysisOptions analysisOptions = options("Ascend910B1");
  analysisOptions.stages = {{.stageName = "materialize"},
                            {.stageName = "suffix"}};
  PipelineContractRegistry reductionRegistry;
  reductionRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  ASSERT_TRUE(succeeded(reductionRegistry.addProfileContract(
      {.stage = analysisOptions.stages[0],
       .contractId = "reduction-sum-max-tiles",
       .contractVersion = "1"},
      makeReductionSumMaxTilesContract(analysisOptions.stages[0], 3, 65536,
                                       32, 262144, 131072, 4, 1))));
  ASSERT_TRUE(succeeded(reductionRegistry.addProfileContract(
      {.stage = analysisOptions.stages[1],
       .contractId = "reduction-sum-extra-buffer",
       .contractVersion = "1"},
      makeReductionSumExtraBufferContract(analysisOptions.stages[1], 3, 65536,
                                          32, 262144, 131072, 4))));
  OwningOpRef<ModuleOp> module = parse(kReductionSum);
  ASSERT_TRUE(module);

  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, reductionRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Reject);
  EXPECT_EQ(result.lowerBoundBytes, 393220);
  ASSERT_TRUE(result.unsupportedReasons.empty());
  ASSERT_EQ(result.certificates.size(), 1u);
  EXPECT_EQ(result.certificates[0].kind, "witness");
  EXPECT_EQ(result.certificates[0].resourceIds,
            SmallVector<ResourceId>({0, 1, 2}));
  EXPECT_EQ(result.certificates[0].contractTrace,
            SmallVector<std::string>({"ttir-reduction-sum-v1",
                                      "reduction-sum-max-tiles",
                                      "reduction-sum-extra-buffer"}));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ReductionSumOddInputDefers) {
  std::string source = replaceAll(kReductionSum, "65536", "65535");
  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-reduction-shape"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ReductionSumFastMathDefers) {
  std::string source = replaceOnce(
      kReductionSum, "%add = arith.addf %lhs, %rhs : f32",
      "%add = arith.addf %lhs, %rhs fastmath<fast> : f32");
  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-reduction-combiner"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ReshapeCopyWithReorderDefers) {
  std::string source = replaceOnce(
      kReshapeCopy, "%view = tt.reshape %value :",
      "%view = tt.reshape %value allow_reorder :");
  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-view-dataflow"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, BroadcastDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>",
      R"mlir(    %expanded = tt.expand_dims %value {axis = 1 : i32} : tensor<65536xf32> -> tensor<65536x1xf32>
    %broadcast = tt.broadcast %expanded : tensor<65536x1xf32> -> tensor<65536x2xf32>
    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>)mlir");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op-broadcast"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ExpandDimsDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>",
      R"mlir(    %expanded = tt.expand_dims %value {axis = 1 : i32} : tensor<65536xf32> -> tensor<65536x1xf32>
    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>)mlir");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op-expand-dims"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, BitcastDefersWithNamedReason) {
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>",
      R"mlir(    %bits = tt.bitcast %value : tensor<65536xf32> -> tensor<65536xi32>
    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>)mlir");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op-bitcast"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       BinaryAddRejectsOneLoadUsedForBothOperands) {
  std::string source = replaceOnce(
      kBinaryAdd, "%sum = arith.addf %lhs_value, %rhs_value",
      "%sum = arith.addf %lhs_value, %lhs_value");

  TTIRUBAnalysisResult result = analyze(source, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-elementwise-dataflow"));
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
  constexpr StringLiteral source = R"mlir(
module {
  tt.func public @copy(%src: tensor<?x!tt.ptr<f32>>) {
    %value = tt.load %src : tensor<?x!tt.ptr<f32>>
    tt.return
  }
}
)mlir";

  TTIRUBAnalysisResult result = analyze(source, options());
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
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>",
      R"mlir(    %value = scf.execute_region -> tensor<65536xf32> {
      %nested = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
      scf.yield %nested : tensor<65536xf32>
    })mlir");

  TTIRUBAnalysisResult result = analyze(source, options());
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

TEST_F(TTIRUBLowerBoundAnalysisTest, NullModuleDefersAsMalformedIR) {
  TTIRUBAnalysisResult result = analyzeModule(ModuleOp(), options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       UnknownOpWithSuccessorDefersBeforeVerifier) {
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::ReturnOp returnOp = findOnlyOp<triton::ReturnOp>(*module);
  OperationState state(returnOp.getLoc(), "test.malformed_unknown");
  state.addSuccessors(returnOp->getBlock());
  Operation *unknown = Operation::create(state);
  returnOp->getBlock()->getOperations().insert(returnOp->getIterator(),
                                                unknown);

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       P4OperationFamiliesDeferWithStableNamedReasons) {
  const std::pair<StringRef, StringRef> cases[] = {
      {"tt.descriptor_gather", "unsupported-op-descriptor-memory"},
      {"tt.gather", "unsupported-op-irregular-memory"},
      {"tt.trans", "unsupported-op-layout-transform"},
      {"tt.scan", "unsupported-op-scan"},
      {"tt.cat", "unsupported-op-shape-construction"},
      {"tt.atomic_rmw", "unsupported-op-atomic"},
      {"tt.print", "unsupported-op-launch-or-diagnostics"},
      {"scf.while", "unsupported-op-control-flow"},
  };
  for (auto [operationName, reason] : cases) {
    context.allowUnregisteredDialects();
    OwningOpRef<ModuleOp> module = parse();
    ASSERT_TRUE(module);
    triton::ReturnOp returnOp = findOnlyOp<triton::ReturnOp>(*module);
    OperationState state(returnOp.getLoc(), operationName);
    Operation *unsupported = Operation::create(state);
    returnOp->getBlock()->getOperations().insert(returnOp->getIterator(),
                                                  unsupported);

    auto [result, diagnosticCount] =
        analyzeModuleCapturingDiagnostics(*module, options());
    EXPECT_EQ(result.decision, TTIRUBDecision::Defer) << operationName.str();
    EXPECT_TRUE(hasReason(result, reason)) << operationName.str();
    EXPECT_EQ(diagnosticCount, 0u) << operationName.str();
  }
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ArithmeticOpWithMalformedRegionDefersBeforeVerifier) {
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::ReturnOp returnOp = findOnlyOp<triton::ReturnOp>(*module);
  OperationState state(returnOp.getLoc(), "arith.malformed");
  Region *region = state.addRegion();
  auto block = std::make_unique<Block>();
  OperationState innerState(returnOp.getLoc(), "test.inner");
  block->push_back(Operation::create(innerState));
  region->push_back(block.release());
  Operation *arithmetic = Operation::create(state);
  returnOp->getBlock()->getOperations().insert(returnOp->getIterator(),
                                                arithmetic);

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op-arithmetic"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ReductionWithMissingPropertyDefersBeforeVerifier) {
  std::string source = replaceOnce(
      kDirectLoadCopy,
      "    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>",
      R"mlir(    %sum = "tt.reduce" (%value) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<65536xf32>) -> f32)mlir");
  OwningOpRef<ModuleOp> module = parse(source);
  ASSERT_TRUE(module);
  triton::ReduceOp reduce = findOnlyOp<triton::ReduceOp>(*module);
  reduce.getProperties().axis = {};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unsupported-op-reduction"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ZeroRegionModuleDefersAsMalformedIR) {
  OperationState state(UnknownLoc::get(&context), ModuleOp::getOperationName());
  Operation *rawModule = Operation::create(state);
  ModuleOp malformedModule = cast<ModuleOp>(rawModule);

  TTIRUBAnalysisResult result = analyzeModule(malformedModule, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  rawModule->destroy();
}

TEST_F(TTIRUBLowerBoundAnalysisTest, MissingRangeAttrsDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::MakeRangeOp range = findOnlyOp<triton::MakeRangeOp>(*module);
  replaceWithMalformedOperation(range, range->getResultTypes());

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, CorruptLoadSegmentsDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::LoadOp load = findOnlyOp<triton::LoadOp>(*module);
  load.getProperties().operandSegmentSizes = {0, 1, 0};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       NegativeLoadSegmentDefersBeforeVerifier) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::LoadOp load = findOnlyOp<triton::LoadOp>(*module);
  load.getProperties().operandSegmentSizes = {-1, 2, 0};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoadSegmentSumMismatchDefersBeforeVerifier) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::LoadOp load = findOnlyOp<triton::LoadOp>(*module);
  load.getProperties().operandSegmentSizes = {1, 1, 0};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LoadOtherWithoutMaskDefersBeforeVerifier) {
  std::string source = replaceOnce(
      kDirectLoadCopy, "%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>",
      "%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, "
      "%mask: tensor<65536xi1>, %other: tensor<65536xf32>");
  source = replaceOnce(
      source,
      "%value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>",
      "%value = tt.load %src_ptrs, %mask, %other : "
      "tensor<65536x!tt.ptr<f32>>");
  OwningOpRef<ModuleOp> module = parse(source);
  ASSERT_TRUE(module);
  triton::LoadOp load = findOnlyOp<triton::LoadOp>(*module);
  load->eraseOperand(1);
  load.getProperties().operandSegmentSizes = {1, 0, 1};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ShortStoreOperandGroupsDeferBeforeVerifier) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::StoreOp store = findOnlyOp<triton::StoreOp>(*module);
  store->eraseOperand(1);

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       LongStoreOperandGroupsDeferBeforeVerifier) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::StoreOp store = findOnlyOp<triton::StoreOp>(*module);
  Value value = store.getValue();
  store->insertOperands(store->getNumOperands(), {value, value});

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       MissingLoadDefaultPropertyDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::LoadOp load = findOnlyOp<triton::LoadOp>(*module);
  load.getProperties().cache = {};

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       MissingStoreDefaultPropertyDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::StoreOp store = findOnlyOp<triton::StoreOp>(*module);
  store.getProperties().cache = {};

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ZeroRegionFunctionDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::FuncOp function = findOnlyOp<triton::FuncOp>(*module);
  OperationState state(function.getLoc(), triton::FuncOp::getOperationName());
  Operation *malformedFunction = Operation::create(state);
  function->getBlock()->getOperations().insert(function->getIterator(),
                                                malformedFunction);
  function.erase();

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       MissingFunctionNameDefersBeforeVerifier) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::FuncOp function = findOnlyOp<triton::FuncOp>(*module);
  function.getProperties().sym_name = {};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       MissingFunctionTypeDefersBeforeVerifier) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::FuncOp function = findOnlyOp<triton::FuncOp>(*module);
  function.getProperties().function_type = {};

  auto [result, diagnosticCount] =
      analyzeModuleCapturingDiagnostics(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
  EXPECT_EQ(diagnosticCount, 0u);
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       FunctionSignatureBodyMismatchDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::FuncOp function = findOnlyOp<triton::FuncOp>(*module);
  function.setFunctionType(FunctionType::get(&context, {}, {}));

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ExtraMatchedResultDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::AddPtrOp addPtr = findOnlyOp<triton::AddPtrOp>(*module);
  SmallVector<Type> resultTypes(addPtr->getResultTypes());
  resultTypes.push_back(addPtr.getType());
  replaceWithMalformedOperation(addPtr, resultTypes);

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ExtraMatchedRegionDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::AddPtrOp addPtr = findOnlyOp<triton::AddPtrOp>(*module);
  replaceWithMalformedOperation(addPtr, addPtr->getResultTypes(), true);

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ExtraAddFResultDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse(kBinaryAdd);
  ASSERT_TRUE(module);
  arith::AddFOp add = findOnlyOp<arith::AddFOp>(*module);
  SmallVector<Type> resultTypes(add->getResultTypes());
  resultTypes.push_back(add.getType());
  replaceWithMalformedOperation(add, resultTypes);

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ExtraReshapeResultDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse(kReshapeCopy);
  ASSERT_TRUE(module);
  triton::ReshapeOp reshape = findOnlyOp<triton::ReshapeOp>(*module);
  SmallVector<Type> resultTypes(reshape->getResultTypes());
  resultTypes.push_back(reshape.getType());
  replaceWithMalformedOperation(reshape, resultTypes);

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ExtraMatchedSuccessorDefersAsMalformedIR) {
  MLIRContext malformedContext;
  malformedContext.allowUnregisteredDialects();
  Location location = UnknownLoc::get(&malformedContext);
  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  Block &body = module->getBodyRegion().front();

  OperationState sourceState(location, "test.source");
  Type i32 = IntegerType::get(&malformedContext, 32);
  sourceState.addTypes({i32, i32});
  Operation *source = Operation::create(sourceState);
  body.push_back(source);

  OperationState addPtrState(location, "tt.addptr");
  addPtrState.addOperands(source->getResults());
  addPtrState.addTypes(i32);
  addPtrState.addSuccessors(&body);
  body.push_back(Operation::create(addPtrState));

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, MultipleReturnsDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::ReturnOp returnOp = findOnlyOp<triton::ReturnOp>(*module);
  Operation *duplicate = returnOp->clone();
  returnOp->getBlock()->getOperations().insert(returnOp->getIterator(),
                                                duplicate);

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, InvalidSSAPlacementDefersAsMalformedIR) {
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  triton::SplatOp splat = findOnlyOp<triton::SplatOp>(*module);
  triton::LoadOp load = findOnlyOp<triton::LoadOp>(*module);
  splat->moveAfter(load);

  TTIRUBAnalysisResult result = analyzeModule(*module, options());
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "malformed-ir"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       RegistryExposesExactSyntheticContractProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  EXPECT_TRUE(registry.matchesProfile(analysisOptions.pipelineIdentity,
                                      analysisOptions.stages));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, OmittedStageDefersAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages.clear();
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, DuplicateStageDefersAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages.push_back({.stageName = "preserve"});
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, AdditionalStageDefersAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages.push_back({.stageName = "unknown"});
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ReorderedStagesDeferAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages.clear();
  analysisOptions.stages.push_back({.stageName = "unknown"});
  analysisOptions.stages.push_back({.stageName = "preserve"});
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, UnknownStageDefersAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages.front().stageName = "unknown";
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, StageOptionsDeferAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.stages.front().options["tile"] = "2";
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, ContractIdMismatchDefersAsUnknownProfile) {
  PipelineContractRegistry mismatchedRegistry;
  TTIRUBAnalysisOptions analysisOptions = options();
  mismatchedRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  EXPECT_TRUE(failed(mismatchedRegistry.addProfileContract(
      {.stage = {.stageName = "preserve"},
       .contractId = "fixed-disposition",
       .contractVersion = "1"},
      std::make_unique<FixedDispositionContract>(
          "preserve", ContractDisposition::Preserve, "other", "1"))));
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, mismatchedRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ContractVersionMismatchDefersAsUnknownProfile) {
  PipelineContractRegistry mismatchedRegistry;
  TTIRUBAnalysisOptions analysisOptions = options();
  mismatchedRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  EXPECT_TRUE(failed(mismatchedRegistry.addProfileContract(
      {.stage = {.stageName = "preserve"},
       .contractId = "fixed-disposition",
       .contractVersion = "1"},
      std::make_unique<FixedDispositionContract>(
          "preserve", ContractDisposition::Preserve, "fixed-disposition",
          "2"))));
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, mismatchedRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       DuplicateContractsDeferAsUnknownProfile) {
  PipelineContractRegistry duplicateRegistry;
  TTIRUBAnalysisOptions analysisOptions = options();
  duplicateRegistry.setProfileIdentity(analysisOptions.pipelineIdentity);
  EXPECT_TRUE(succeeded(duplicateRegistry.addProfileContract(
      {.stage = {.stageName = "preserve"},
       .contractId = "fixed-disposition",
       .contractVersion = "1"},
      std::make_unique<FixedDispositionContract>(
          "preserve", ContractDisposition::Preserve))));
  EXPECT_TRUE(succeeded(duplicateRegistry.addProfileContract(
      {.stage = {.stageName = "preserve"},
       .contractId = "fixed-disposition",
       .contractVersion = "1"},
      std::make_unique<FixedDispositionContract>(
          "preserve", ContractDisposition::Preserve))));
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result = analyzeTTIRUBLowerBound(
      *module, analysisOptions, duplicateRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest, TargetMismatchDefersAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.pipelineIdentity.targetArch = "Ascend950";
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       CanonicalTTIRHashMismatchDefersAsUnknownProfile) {
  TTIRUBAnalysisOptions analysisOptions = options();
  analysisOptions.pipelineIdentity.canonicalTtirSha256 = "other-ttir";
  TTIRUBAnalysisResult result = analyze(kDirectLoadCopy, analysisOptions);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

} // namespace
} // namespace mlir::triton::ascend::ub
