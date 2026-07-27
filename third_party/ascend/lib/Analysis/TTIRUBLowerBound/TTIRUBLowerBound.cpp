#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/VerifierSafety.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <array>

namespace mlir::triton::ascend::ub {

namespace detail {

bool hasValidLoadOperandSegments(ArrayRef<int32_t> segments,
                                 unsigned numOperands) {
  if (segments.size() != 3 ||
      llvm::any_of(segments, [](int32_t size) { return size < 0; }))
    return false;
  if (segments[0] != 1 || segments[1] > 1 || segments[2] > 1 ||
      (segments[2] != 0 && segments[1] == 0))
    return false;
  int64_t total = static_cast<int64_t>(segments[0]) + segments[1] +
                  segments[2];
  return total == numOperands;
}

bool hasValidStoreOperandSegments(ArrayRef<int32_t> segments,
                                  unsigned numOperands) {
  if (segments.size() != 3 ||
      llvm::any_of(segments, [](int32_t size) { return size < 0; }))
    return false;
  if (segments[0] != 1 || segments[1] != 1 || segments[2] > 1)
    return false;
  int64_t total = static_cast<int64_t>(segments[0]) + segments[1] +
                  segments[2];
  return total == numOperands;
}

} // namespace detail

LogicalResult materializeDirectTensorLoads(
    ModuleOp module, MandatoryUBResourceGraph &graph,
    SmallVectorImpl<std::string> &unsupportedReasons);

namespace {

void addReason(SmallVectorImpl<std::string> &reasons, StringRef reason) {
  if (!llvm::is_contained(reasons, reason))
    reasons.push_back(reason.str());
}

template <typename OpTy> bool hasExactRegisteredType(Operation *operation) {
  return operation->getName().isRegistered() &&
         operation->getName().getTypeID() == TypeID::get<OpTy>();
}

bool hasExactShape(Operation *operation, unsigned numOperands,
                   unsigned numResults, unsigned numRegions,
                   unsigned numSuccessors) {
  return operation->getNumOperands() == numOperands &&
         operation->getNumResults() == numResults &&
         operation->getNumRegions() == numRegions &&
         operation->getNumSuccessors() == numSuccessors;
}

StringRef classifyUnsupportedOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == triton::ReduceOp::getOperationName())
    return "unsupported-op-reduction";
  if (name == "tt.scan" || name == "tt.scan.return")
    return "unsupported-op-scan";
  if (name == triton::BroadcastOp::getOperationName())
    return "unsupported-op-broadcast";
  if (name == triton::ExpandDimsOp::getOperationName())
    return "unsupported-op-expand-dims";
  if (name == triton::BitcastOp::getOperationName())
    return "unsupported-op-bitcast";
  if (name == "tt.make_tensor_ptr" || name == "tt.advance" ||
      name == "tt.descriptor_load" || name == "tt.descriptor_store" ||
      name == "tt.descriptor_reduce" || name == "tt.descriptor_gather" ||
      name == "tt.descriptor_scatter")
    return "unsupported-op-descriptor-memory";
  if (name == "tt.gather" || name == "tt.scatter" ||
      name == "tt.index_select" || name == "tt.index_put" ||
      name == "tt.load_unstructured" || name == "tt.store_unstructured")
    return "unsupported-op-irregular-memory";
  if (name == "tt.trans" || name == "tt.flip" || name == "tt.sort" ||
      name == "ttascend.flip" || name == "ttascend.sort")
    return "unsupported-op-layout-transform";
  if (name == "tt.cat" || name == "tt.join" || name == "tt.split")
    return "unsupported-op-shape-construction";
  if (name == "tt.atomic_rmw" || name == "tt.atomic_cas")
    return "unsupported-op-atomic";
  if (name == "tt.dot_scaled")
    return "unsupported-dot-scaled-requires-full-boundary";
  if (name.starts_with("ttascend.") || name.starts_with("hivm.custom_"))
    return "unsupported-op-custom";
  if (name == "tt.get_program_id" || name == "tt.get_num_programs" ||
      name == "tt.assert" || name == "tt.print")
    return "unsupported-op-launch-or-diagnostics";
  if (name == "scf.while" || name == "scf.if")
    return "unsupported-op-control-flow";
  if (name.split('.').first == "arith")
    return "unsupported-op-arithmetic";
  if (operation->getNumRegions() != 0)
    return "nested-region";
  return "unsupported-op";
}

StringRef getVerifierPreflightFailure(Operation *root) {
  unsigned functionCount = 0;
  SmallVector<Operation *> worklist{root};
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    StringRef name = operation->getName().getStringRef();
    if (name == ModuleOp::getOperationName()) {
      if (operation != root || !hasExactRegisteredType<ModuleOp>(operation) ||
          !hasExactShape(operation, 0, 0, 1, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
      // ModuleOp has no mandatory inherent properties in this MLIR pin.
    } else if (name == triton::FuncOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::FuncOp>(operation) ||
          !hasExactShape(operation, 0, 0, 1, 0) ||
          operation->getParentOp() != root || ++functionCount != 1)
        return "malformed-ir";
      OpaqueProperties storage = operation->getPropertiesStorage();
      if (!storage)
        return "malformed-ir";
      const auto *properties = storage.as<triton::FuncOp::Properties *>();
      Attribute rawName = properties->sym_name;
      Attribute rawType = properties->function_type;
      auto functionTypeAttr = dyn_cast_or_null<TypeAttr>(rawType);
      if (!isa_and_nonnull<StringAttr>(rawName) || !functionTypeAttr ||
          !isa<FunctionType>(functionTypeAttr.getValue()))
        return "malformed-ir";
    } else if (name == triton::ReturnOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::ReturnOp>(operation) ||
          !hasExactShape(operation, 0, 0, 0, 0))
        return "malformed-ir";
      // ReturnOp uses EmptyProperties and has no mandatory attributes.
    } else if (name == triton::MakeRangeOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::MakeRangeOp>(operation) ||
          !hasExactShape(operation, 0, 1, 0, 0))
        return "malformed-ir";
      OpaqueProperties storage = operation->getPropertiesStorage();
      if (!storage)
        return "malformed-ir";
      const auto *properties = storage.as<triton::MakeRangeOp::Properties *>();
      Attribute rawStart = properties->start;
      Attribute rawEnd = properties->end;
      if (!isa_and_nonnull<IntegerAttr>(rawStart) ||
          !isa_and_nonnull<IntegerAttr>(rawEnd))
        return "malformed-ir";
    } else if (name == triton::SplatOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::SplatOp>(operation) ||
          !hasExactShape(operation, 1, 1, 0, 0))
        return "malformed-ir";
    } else if (name == triton::ReshapeOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::ReshapeOp>(operation) ||
          !hasExactShape(operation, 1, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
    } else if (name == triton::AddPtrOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::AddPtrOp>(operation) ||
          !hasExactShape(operation, 2, 1, 0, 0))
        return "malformed-ir";
    } else if (name == triton::LoadOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::LoadOp>(operation) ||
          operation->getNumResults() != 1 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return "malformed-ir";
      OpaqueProperties storage = operation->getPropertiesStorage();
      if (!storage)
        return "malformed-ir";
      const auto *properties = storage.as<triton::LoadOp::Properties *>();
      if (!properties->boundaryCheck || !properties->cache ||
          !properties->evict || !properties->isVolatile ||
          !detail::hasValidLoadOperandSegments(
              properties->operandSegmentSizes,
              operation->getNumOperands()))
        return "malformed-ir";
    } else if (name == triton::StoreOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::StoreOp>(operation) ||
          operation->getNumResults() != 0 ||
          operation->getNumRegions() != 0 ||
          operation->getNumSuccessors() != 0)
        return "malformed-ir";
      OpaqueProperties storage = operation->getPropertiesStorage();
      if (!storage)
        return "malformed-ir";
      const auto *properties = storage.as<triton::StoreOp::Properties *>();
      if (!properties->boundaryCheck || !properties->cache ||
          !properties->evict)
        return "malformed-ir";
      // StoreOp has no AttrSizedOperandSegments property in this checkout;
      // its ODS groups are uniquely derived from the total operand count.
      int64_t maskSize = static_cast<int64_t>(operation->getNumOperands()) - 2;
      if (maskSize < 0 || maskSize > 1)
        return "malformed-ir";
      std::array<int32_t, 3> segments = {
          1, 1, static_cast<int32_t>(maskSize)};
      if (!detail::hasValidStoreOperandSegments(
              segments, operation->getNumOperands()))
        return "malformed-ir";
    } else if (name == triton::ReduceOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::ReduceOp>(operation) ||
          !hasExactShape(operation, 1, 1, 1, 0) ||
          !operation->getPropertiesStorage())
        return "unsupported-op-reduction";
      const auto *properties =
          operation->getPropertiesStorage().as<triton::ReduceOp::Properties *>();
      if (!isa_and_nonnull<IntegerAttr>(properties->axis) ||
          !operation->getRegion(0).hasOneBlock() ||
          operation->getRegion(0).front().getNumArguments() != 2)
        return "unsupported-op-reduction";
    } else if (name == triton::ReduceReturnOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::ReduceReturnOp>(operation) ||
          !hasExactShape(operation, 1, 0, 0, 0))
        return "unsupported-op-reduction";
    } else if (name == arith::ConstantOp::getOperationName()) {
      if (!hasExactRegisteredType<arith::ConstantOp>(operation) ||
          !hasExactShape(operation, 0, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
      const auto *properties =
          operation->getPropertiesStorage().as<arith::ConstantOp::Properties *>();
      if (!isa_and_nonnull<IntegerAttr, DenseElementsAttr>(properties->value))
        return "malformed-ir";
    } else if (name == arith::MulIOp::getOperationName()) {
      if (!hasExactRegisteredType<arith::MulIOp>(operation) ||
          !hasExactShape(operation, 2, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
    } else if (name == triton::ExpandDimsOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::ExpandDimsOp>(operation) ||
          !hasExactShape(operation, 1, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
      const auto *properties = operation->getPropertiesStorage()
                                   .as<triton::ExpandDimsOp::Properties *>();
      if (!isa_and_nonnull<IntegerAttr>(properties->axis))
        return "malformed-ir";
    } else if (name == triton::BroadcastOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::BroadcastOp>(operation) ||
          !hasExactShape(operation, 1, 1, 0, 0))
        return "malformed-ir";
    } else if (name == triton::DotOp::getOperationName()) {
      if (!hasExactRegisteredType<triton::DotOp>(operation) ||
          !hasExactShape(operation, 3, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
      const auto *properties =
          operation->getPropertiesStorage().as<triton::DotOp::Properties *>();
      if (!properties->inputPrecision || !properties->maxNumImpreciseAcc)
        return "malformed-ir";
    } else if (name == math::ExpOp::getOperationName()) {
      if (!hasExactRegisteredType<math::ExpOp>(operation) ||
          !hasExactShape(operation, 1, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
    } else if (name == scf::ForOp::getOperationName()) {
      if (!hasExactRegisteredType<scf::ForOp>(operation) ||
          !hasExactShape(operation, 4, 1, 1, 0) ||
          !operation->getRegion(0).hasOneBlock() ||
          operation->getRegion(0).front().getNumArguments() != 2)
        return "unsupported-op-loop";
    } else if (name == scf::YieldOp::getOperationName()) {
      if (!hasExactRegisteredType<scf::YieldOp>(operation) ||
          !hasExactShape(operation, 1, 0, 0, 0))
        return "unsupported-op-loop";
    } else if (name == arith::AddFOp::getOperationName()) {
      if (!hasExactRegisteredType<arith::AddFOp>(operation) ||
          !hasExactShape(operation, 2, 1, 0, 0) ||
          !operation->getPropertiesStorage())
        return "malformed-ir";
    } else {
      // Raw names are used only to reject unsupported operations. Never walk
      // into them or expose their structure/properties to the verifier.
      return classifyUnsupportedOperation(operation);
    }

    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          worklist.push_back(&nested);
  }
  return functionCount == 1 ? StringRef() : StringRef("malformed-ir");
}

bool hasKnownPipelineProfile(const TTIRUBAnalysisOptions &options,
                             const PipelineContractRegistry &registry) {
  return options.pipelineIdentity.targetArch == options.targetArch &&
         registry.matchesProfile(options.pipelineIdentity, options.stages);
}

} // namespace

TTIRUBAnalysisResult
analyzeTTIRUBLowerBound(ModuleOp module, const TTIRUBAnalysisOptions &options,
                        const PipelineContractRegistry &registry) {
  TTIRUBAnalysisResult result;
  if (!module) {
    addReason(result.unsupportedReasons, "malformed-ir");
    return result;
  }
  StringRef preflightFailure =
      getVerifierPreflightFailure(module.getOperation());
  if (!preflightFailure.empty()) {
    addReason(result.unsupportedReasons, preflightFailure);
    return result;
  }
  if (failed(verify(module.getOperation()))) {
    addReason(result.unsupportedReasons, "malformed-ir");
    return result;
  }
  result.capacityBytes = getUBCapacityBytes(options.targetArch);
  if (!result.capacityBytes) {
    addReason(result.unsupportedReasons, "unknown-target");
    return result;
  }
  if (options.compileMode != "aiv") {
    addReason(result.unsupportedReasons, "unsupported-compile-mode");
    return result;
  }
  if (!hasKnownPipelineProfile(options, registry)) {
    addReason(result.unsupportedReasons, "unknown-pipeline-profile");
    return result;
  }

  MandatoryUBResourceGraph graph;
  if (failed(materializeDirectTensorLoads(module, graph,
                                          result.unsupportedReasons)))
    return result;

  for (size_t ordinal = 0; ordinal < options.stages.size(); ++ordinal) {
    if (failed(registry.applyProfileStage(graph, options.stages[ordinal],
                                          ordinal))) {
      addReason(result.unsupportedReasons, "pipeline-contract-internal-error");
      return result;
    }
  }

  for (const MandatoryUBResource &resource : graph.resources()) {
    if (resource.validity == ValidityState::Invalid) {
      addReason(result.unsupportedReasons, resource.invalidReason);
      for (const std::string &trace : resource.contractTrace)
        addReason(result.deferTrace, trace);
    }
  }
  if (!result.unsupportedReasons.empty())
    return result;

  FailureOr<LowerBoundCertificate> certificate =
      graph.solveWitnessLowerBound();
  if (failed(certificate)) {
    addReason(result.unsupportedReasons, "malformed-resource-graph");
    return result;
  }

  result.lowerBoundBytes = certificate->bytes;
  result.certificates.push_back(*certificate);
  if (result.lowerBoundBytes > *result.capacityBytes)
    result.decision = TTIRUBDecision::Reject;
  return result;
}

} // namespace mlir::triton::ascend::ub
