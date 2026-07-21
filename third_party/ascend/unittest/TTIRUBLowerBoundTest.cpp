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
  EXPECT_TRUE(registry.hasExactlyOneMatchingContract(
      {.stageName = "preserve"}, "fixed-disposition", "1"));
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
  mismatchedRegistry.addForTesting(std::make_unique<FixedDispositionContract>(
      "preserve", ContractDisposition::Preserve, "other", "1"));
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, options(), mismatchedRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       ContractVersionMismatchDefersAsUnknownProfile) {
  PipelineContractRegistry mismatchedRegistry;
  mismatchedRegistry.addForTesting(std::make_unique<FixedDispositionContract>(
      "preserve", ContractDisposition::Preserve, "fixed-disposition", "2"));
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, options(), mismatchedRegistry);
  EXPECT_EQ(result.decision, TTIRUBDecision::Defer);
  EXPECT_TRUE(hasReason(result, "unknown-pipeline-profile"));
}

TEST_F(TTIRUBLowerBoundAnalysisTest,
       DuplicateContractsDeferAsUnknownProfile) {
  PipelineContractRegistry duplicateRegistry;
  duplicateRegistry.addForTesting(std::make_unique<FixedDispositionContract>(
      "preserve", ContractDisposition::Preserve));
  duplicateRegistry.addForTesting(std::make_unique<FixedDispositionContract>(
      "preserve", ContractDisposition::Preserve));
  OwningOpRef<ModuleOp> module = parse();
  ASSERT_TRUE(module);
  TTIRUBAnalysisResult result =
      analyzeTTIRUBLowerBound(*module, options(), duplicateRegistry);
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

} // namespace
} // namespace mlir::triton::ascend::ub
