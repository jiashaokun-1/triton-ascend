#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <climits>

namespace mlir::triton::ascend::ub {
namespace {

void addReason(SmallVectorImpl<std::string> &reasons, StringRef reason) {
  if (!llvm::is_contained(reasons, reason))
    reasons.push_back(reason.str());
}

LogicalResult defer(SmallVectorImpl<std::string> &reasons, StringRef reason) {
  addReason(reasons, reason);
  return failure();
}

FailureOr<int64_t> getPayloadBytes(RankedTensorType type,
                                   SmallVectorImpl<std::string> &reasons) {
  if (!type.hasStaticShape()) {
    addReason(reasons, "dynamic-shape");
    return failure();
  }
  if (type.getRank() != 1) {
    addReason(reasons, "unsupported-shape");
    return failure();
  }

  Type elementType = type.getElementType();
  if (!isa<IntegerType, FloatType>(elementType)) {
    addReason(reasons, "unsupported-element-type");
    return failure();
  }
  unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
  if (elementBitWidth < 8) {
    addReason(reasons, "sub-byte-element-type");
    return failure();
  }
  if (elementBitWidth != 8 && elementBitWidth != 16 &&
      elementBitWidth != 32 && elementBitWidth != 64) {
    addReason(reasons, "unsupported-element-type");
    return failure();
  }

  int64_t numElements = type.getNumElements();
  int64_t bytesPerElement = llvm::divideCeil(elementBitWidth, 8u);
  if (numElements < 0 ||
      (bytesPerElement != 0 && numElements > INT64_MAX / bytesPerElement)) {
    addReason(reasons, "arithmetic-overflow");
    return failure();
  }
  return numElements * bytesPerElement;
}

struct ContiguousPointerChain {
  triton::MakeRangeOp range;
  triton::SplatOp splat;
  triton::AddPtrOp addPtr;
};

FailureOr<ContiguousPointerChain>
matchContiguousPointer(Value pointer, Block &entryBlock, int64_t numElements,
                       Type elementType,
                       SmallVectorImpl<std::string> &reasons) {
  auto pointerTensorType = dyn_cast<RankedTensorType>(pointer.getType());
  auto tensorPointerType =
      pointerTensorType
          ? dyn_cast<triton::PointerType>(pointerTensorType.getElementType())
          : triton::PointerType();
  if (!pointerTensorType || !pointerTensorType.hasStaticShape() ||
      pointerTensorType.getRank() != 1 ||
      pointerTensorType.getNumElements() != numElements ||
      !tensorPointerType || tensorPointerType.getPointeeType() != elementType ||
      tensorPointerType.getAddressSpace() != 1) {
    addReason(reasons, "non-contiguous-pointer");
    return failure();
  }

  auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>();
  if (!addPtr || addPtr->getNumOperands() != 2 ||
      addPtr->getNumResults() != 1) {
    addReason(reasons, "non-contiguous-pointer");
    return failure();
  }
  auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
  auto range = addPtr.getOffset().getDefiningOp<triton::MakeRangeOp>();
  if (!splat || splat->getNumOperands() != 1 ||
      splat->getNumResults() != 1 || !range ||
      range->getNumOperands() != 0 || range->getNumResults() != 1 ||
      !range.getStartAttr() || !range.getEndAttr()) {
    addReason(reasons, "non-contiguous-pointer");
    return failure();
  }
  auto rangeType =
      dyn_cast<RankedTensorType>(range.getResult().getType());
  if (range.getStartAttr().getInt() != 0 ||
      range.getEndAttr().getInt() != numElements || !rangeType ||
      !rangeType.hasStaticShape() || rangeType.getRank() != 1 ||
      rangeType.getNumElements() != numElements ||
      !rangeType.getElementType().isInteger(32) ||
      splat.getResult().getType() != pointer.getType()) {
    addReason(reasons, "non-contiguous-pointer");
    return failure();
  }

  auto blockArgument = dyn_cast<BlockArgument>(splat.getSrc());
  auto pointerType = blockArgument
                         ? dyn_cast<triton::PointerType>(blockArgument.getType())
                         : triton::PointerType();
  if (!blockArgument || blockArgument.getOwner() != &entryBlock ||
      !pointerType || isa<RankedTensorType>(pointerType.getPointeeType()) ||
      pointerType.getPointeeType() != elementType ||
      pointerType.getAddressSpace() != 1 || !blockArgument.hasOneUse() ||
      !splat.getResult().hasOneUse()) {
    addReason(reasons, "non-contiguous-pointer");
    return failure();
  }

  return ContiguousPointerChain{range, splat, addPtr};
}

bool isDirectlyInEntryBlock(Operation *operation, triton::FuncOp function) {
  return operation->getParentOp() == function.getOperation() &&
         operation->getBlock() == &function.getBody().front();
}

LogicalResult rejectUnsupportedOperations(
    ModuleOp module, const llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  WalkResult result = module.walk([&](Operation *operation) {
    if (isa<triton::ReduceOp>(operation)) {
      addReason(reasons, "unsupported-op-reduction");
      return WalkResult::interrupt();
    }
    if (isa<ModuleOp>(operation))
      return WalkResult::advance();
    if (auto function = dyn_cast<triton::FuncOp>(operation)) {
      if (function->getParentOp() == module.getOperation())
        return WalkResult::advance();
      addReason(reasons, "nested-region");
      return WalkResult::interrupt();
    }
    if (auto returnOp = dyn_cast<triton::ReturnOp>(operation)) {
      triton::FuncOp function = returnOp->getParentOfType<triton::FuncOp>();
      if (function && function->getNumRegions() == 1 &&
          function.getBody().hasOneBlock() &&
          isDirectlyInEntryBlock(returnOp, function) &&
          returnOp->getNumOperands() == 0)
        return WalkResult::advance();
      addReason(reasons, "nested-region");
      return WalkResult::interrupt();
    }
    if (isa<triton::MakeRangeOp, triton::SplatOp, triton::AddPtrOp,
            triton::LoadOp, triton::StoreOp>(operation)) {
      if (matched.contains(operation))
        return WalkResult::advance();
      addReason(reasons, "unmatched-operation");
      return WalkResult::interrupt();
    }
    if (operation->getDialect() &&
        operation->getDialect()->getNamespace() == "arith") {
      addReason(reasons, "unsupported-op-arithmetic");
      return WalkResult::interrupt();
    }
    addReason(reasons, operation->getNumRegions() ? "nested-region"
                                                  : "unsupported-op");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult materializeLoad(triton::LoadOp load,
                              MandatoryUBResourceGraph &graph,
                              llvm::DenseSet<Operation *> &matched,
                              SmallVectorImpl<std::string> &reasons) {
  if (load->getNumOperands() == 0 || load->getNumResults() != 1)
    return defer(reasons, "malformed-load");
  triton::FuncOp function = load->getParentOfType<triton::FuncOp>();
  if (!function || function->getNumRegions() != 1 ||
      !function.getBody().hasOneBlock() ||
      !isDirectlyInEntryBlock(load, function))
    return defer(reasons, "nested-region");
  if (load->getNumOperands() != 1)
    return defer(reasons, "masked-load");
  if (load.getOther() || !load.getBoundaryCheck().empty() ||
      load.getPadding() || load.getIsVolatile() ||
      load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-load-semantics");

  auto resultType = dyn_cast<RankedTensorType>(load.getType());
  if (!resultType)
    return defer(reasons, "unsupported-shape");
  FailureOr<int64_t> payloadBytes = getPayloadBytes(resultType, reasons);
  if (failed(payloadBytes))
    return failure();

  FailureOr<ContiguousPointerChain> sourceChain = matchContiguousPointer(
      load.getPtr(), function.getBody().front(), resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(sourceChain))
    return failure();
  if (!sourceChain->addPtr.getResult().hasOneUse())
    return defer(reasons, "load-pointer-has-extra-use");

  if (load.getResult().use_empty())
    return defer(reasons, "load-not-reaching-store");
  if (!load.getResult().hasOneUse())
    return defer(reasons, "load-has-extra-use");
  Operation *user = *load.getResult().getUsers().begin();
  if (isa<triton::ReduceOp>(user))
    return defer(reasons, "unsupported-op-reduction");
  auto store = dyn_cast<triton::StoreOp>(user);
  if (!store || store->getNumOperands() < 2 ||
      store->getNumResults() != 0)
    return defer(reasons, "malformed-store");
  if (store.getValue() != load.getResult() ||
      store.getValue().getType() != resultType)
    return defer(reasons, "load-not-reaching-store");
  if (!isDirectlyInEntryBlock(store, function))
    return defer(reasons, "nested-region");
  if (store->getNumOperands() != 2)
    return defer(reasons, "masked-store");
  if (!store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-store-semantics");

  FailureOr<ContiguousPointerChain> destinationChain = matchContiguousPointer(
      store.getPtr(), function.getBody().front(), resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(destinationChain))
    return failure();
  if (!destinationChain->addPtr.getResult().hasOneUse() ||
      sourceChain->range != destinationChain->range ||
      std::distance(sourceChain->range.getResult().use_begin(),
                    sourceChain->range.getResult().use_end()) != 2)
    return defer(reasons, "non-contiguous-pointer");

  MandatoryUBResource resource;
  resource.debugName = "direct-tt-load";
  resource.minPayloadBytes = *payloadBytes;
  resource.minInstances = 1;
  resource.origin = "tt.load";
  resource.kind = MaterializationKind::GMToUBLoad;
  resource.contractTrace.push_back("ttir-direct-load-v1");
  if (graph.addResource(std::move(resource)) == InvalidResourceId)
    return defer(reasons, "malformed-resource-graph");

  matched.insert(sourceChain->range);
  matched.insert(sourceChain->splat);
  matched.insert(sourceChain->addPtr);
  matched.insert(destinationChain->splat);
  matched.insert(destinationChain->addPtr);
  matched.insert(load);
  matched.insert(store);
  return success();
}

} // namespace

LogicalResult materializeDirectTensorLoads(
    ModuleOp module, MandatoryUBResourceGraph &graph,
    SmallVectorImpl<std::string> &unsupportedReasons) {
  SmallVector<triton::FuncOp> functions;
  for (triton::FuncOp function : module.getOps<triton::FuncOp>())
    functions.push_back(function);
  if (functions.size() != 1 || functions.front()->getNumRegions() != 1 ||
      !functions.front().isPublic() ||
      !functions.front().getBody().hasOneBlock())
    return defer(unsupportedReasons, "unsupported-module-shape");

  SmallVector<triton::LoadOp> loads;
  module.walk([&](triton::LoadOp load) { loads.push_back(load); });
  if (loads.empty())
    return defer(unsupportedReasons, "no-mandatory-load");

  llvm::DenseSet<Operation *> matched;
  for (triton::LoadOp load : loads) {
    if (failed(materializeLoad(load, graph, matched, unsupportedReasons)))
      return failure();
  }
  for (BlockArgument argument :
       functions.front().getBody().front().getArguments()) {
    if (!argument.hasOneUse() ||
        !matched.contains(*argument.getUsers().begin()))
      return defer(unsupportedReasons, "unsupported-function-argument");
  }
  return rejectUnsupportedOperations(module, matched, unsupportedReasons);
}

} // namespace mlir::triton::ascend::ub
