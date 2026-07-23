#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

bool isDirectlyInEntryBlock(Operation *operation, triton::FuncOp function);

FailureOr<ContiguousPointerChain>
matchContiguousPointer(Value pointer, triton::FuncOp function,
                       int64_t numElements, Type elementType,
                       SmallVectorImpl<std::string> &reasons) {
  Block &entryBlock = function.getBody().front();
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
  if (!isDirectlyInEntryBlock(range, function) ||
      !isDirectlyInEntryBlock(splat, function) ||
      !isDirectlyInEntryBlock(addPtr, function) ||
      range->getNumRegions() != 0 || range->getNumSuccessors() != 0 ||
      splat->getNumRegions() != 0 || splat->getNumSuccessors() != 0 ||
      addPtr->getNumRegions() != 0 || addPtr->getNumSuccessors() != 0 ||
      !range->isBeforeInBlock(addPtr) || !splat->isBeforeInBlock(addPtr)) {
    addReason(reasons, "invalid-chain-placement");
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
            triton::LoadOp, triton::StoreOp, triton::ReshapeOp,
            arith::AddFOp>(operation)) {
      triton::FuncOp function = operation->getParentOfType<triton::FuncOp>();
      if (matched.contains(operation) && function &&
          isDirectlyInEntryBlock(operation, function) &&
          operation->getNumRegions() == 0 &&
          operation->getNumSuccessors() == 0)
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
      !isDirectlyInEntryBlock(load, function) ||
      load->getNumRegions() != 0 || load->getNumSuccessors() != 0)
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
      load.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(sourceChain))
    return failure();
  if (!sourceChain->addPtr.getResult().hasOneUse() ||
      !sourceChain->addPtr->isBeforeInBlock(load))
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
  if (!isDirectlyInEntryBlock(store, function) ||
      store->getNumRegions() != 0 || store->getNumSuccessors() != 0 ||
      !load->isBeforeInBlock(store))
    return defer(reasons, "nested-region");
  if (store->getNumOperands() != 2)
    return defer(reasons, "masked-store");
  if (!store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-store-semantics");

  FailureOr<ContiguousPointerChain> destinationChain = matchContiguousPointer(
      store.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(destinationChain))
    return failure();
  if (!destinationChain->addPtr.getResult().hasOneUse() ||
      !destinationChain->addPtr->isBeforeInBlock(store) ||
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
  resource.sourceElements = resultType.getNumElements();
  resource.elementBitWidth = resultType.getElementType().getIntOrFloatBitWidth();
  resource.consumer = "tt.store";
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

bool isStrictNoReorderReshape(triton::ReshapeOp reshape, Value source,
                              int64_t numElements, Type elementType,
                              unsigned resultRank, triton::FuncOp function) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto resultType = dyn_cast<RankedTensorType>(reshape.getType());
  return reshape && reshape.getSrc() == source && sourceType && resultType &&
         sourceType.hasStaticShape() && resultType.hasStaticShape() &&
         sourceType.getNumElements() == numElements &&
         resultType.getNumElements() == numElements &&
         sourceType.getElementType() == elementType &&
         resultType.getElementType() == elementType &&
         resultType.getRank() == resultRank && !reshape.getAllowReorder() &&
         !reshape.getEfficientLayout() && reshape->getNumOperands() == 1 &&
         reshape->getNumResults() == 1 && reshape->getNumRegions() == 0 &&
         reshape->getNumSuccessors() == 0 &&
         isDirectlyInEntryBlock(reshape, function) &&
         reshape.getResult().hasOneUse();
}

LogicalResult materializeReshapeCopy(
    triton::LoadOp load, triton::ReshapeOp valueReshape,
    MandatoryUBResourceGraph &graph, llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  triton::FuncOp function = load->getParentOfType<triton::FuncOp>();
  if (!function || !isDirectlyInEntryBlock(load, function) ||
      load->getNumOperands() != 1 || load->getNumResults() != 1 ||
      load->getNumRegions() != 0 || load->getNumSuccessors() != 0)
    return defer(reasons, "unsupported-view-load");
  if (load.getOther() || !load.getBoundaryCheck().empty() ||
      load.getPadding() || load.getIsVolatile() ||
      load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-load-semantics");

  auto sourceType = dyn_cast<RankedTensorType>(load.getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      sourceType.getRank() != 1)
    return defer(reasons, "unsupported-view-shape");
  FailureOr<int64_t> payloadBytes = getPayloadBytes(sourceType, reasons);
  if (failed(payloadBytes))
    return failure();
  const int64_t numElements = sourceType.getNumElements();
  Type elementType = sourceType.getElementType();
  if (!load.getResult().hasOneUse() ||
      !isStrictNoReorderReshape(valueReshape, load.getResult(), numElements,
                                elementType, 2, function) ||
      !load->isBeforeInBlock(valueReshape))
    return defer(reasons, "unsupported-view-dataflow");

  FailureOr<ContiguousPointerChain> sourceChain = matchContiguousPointer(
      load.getPtr(), function, numElements, elementType, reasons);
  if (failed(sourceChain))
    return failure();
  if (!sourceChain->addPtr.getResult().hasOneUse() ||
      !sourceChain->addPtr->isBeforeInBlock(load))
    return defer(reasons, "load-pointer-has-extra-use");

  auto store = dyn_cast<triton::StoreOp>(
      *valueReshape.getResult().getUsers().begin());
  if (!store || store->getNumOperands() != 2 || store->getNumResults() != 0 ||
      store->getNumRegions() != 0 || store->getNumSuccessors() != 0 ||
      store.getValue() != valueReshape.getResult() ||
      !isDirectlyInEntryBlock(store, function) ||
      !valueReshape->isBeforeInBlock(store) ||
      !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-view-store");

  auto pointerReshape = store.getPtr().getDefiningOp<triton::ReshapeOp>();
  auto pointerResultType =
      dyn_cast<RankedTensorType>(store.getPtr().getType());
  auto pointerElementType =
      pointerResultType
          ? dyn_cast<triton::PointerType>(pointerResultType.getElementType())
          : triton::PointerType();
  auto valueResultType = dyn_cast<RankedTensorType>(valueReshape.getType());
  if (!pointerReshape || !pointerResultType || !pointerElementType ||
      !valueResultType ||
      pointerResultType.getShape() != valueResultType.getShape() ||
      pointerElementType.getPointeeType() != elementType ||
      pointerElementType.getAddressSpace() != 1 ||
      !isStrictNoReorderReshape(
          pointerReshape, pointerReshape.getSrc(), numElements,
          pointerResultType.getElementType(), 2, function) ||
      !pointerReshape->isBeforeInBlock(store))
    return defer(reasons, "unsupported-view-pointer");
  FailureOr<ContiguousPointerChain> destinationChain = matchContiguousPointer(
      pointerReshape.getSrc(), function, numElements, elementType, reasons);
  if (failed(destinationChain))
    return failure();
  if (!destinationChain->addPtr.getResult().hasOneUse() ||
      sourceChain->range != destinationChain->range ||
      std::distance(sourceChain->range.getResult().use_begin(),
                    sourceChain->range.getResult().use_end()) != 2)
    return defer(reasons, "non-contiguous-pointer");

  MandatoryUBResource sourceResource;
  sourceResource.debugName = "reshape-copy-input";
  sourceResource.minPayloadBytes = *payloadBytes;
  sourceResource.minInstances = 1;
  sourceResource.origin = "tt.load";
  sourceResource.kind = MaterializationKind::GMToUBLoad;
  sourceResource.birth.ordinal =
      std::distance(function.getBody().front().begin(), load->getIterator());
  sourceResource.lastRequiredUse.ordinal = std::distance(
      function.getBody().front().begin(), valueReshape->getIterator());
  sourceResource.sourceElements = numElements;
  sourceResource.elementBitWidth = elementType.getIntOrFloatBitWidth();
  sourceResource.consumer = "tt.reshape";
  sourceResource.contractTrace.push_back("ttir-reshape-copy-v1");
  ResourceId sourceId = graph.addResource(std::move(sourceResource));

  MandatoryUBResource viewResource;
  viewResource.debugName = "reshape-copy-view";
  viewResource.minPayloadBytes = *payloadBytes;
  viewResource.minInstances = 1;
  viewResource.origin = "tt.reshape";
  viewResource.kind = MaterializationKind::ViewAlias;
  viewResource.birth.ordinal = std::distance(
      function.getBody().front().begin(), valueReshape->getIterator());
  viewResource.lastRequiredUse.ordinal =
      std::distance(function.getBody().front().begin(), store->getIterator());
  viewResource.sourceElements = numElements;
  viewResource.elementBitWidth = elementType.getIntOrFloatBitWidth();
  viewResource.consumer = "tt.store";
  viewResource.contractTrace.push_back("ttir-reshape-copy-v1");
  ResourceId viewId = graph.addResource(std::move(viewResource));
  if (sourceId == InvalidResourceId || viewId == InvalidResourceId)
    return defer(reasons, "malformed-resource-graph");
  graph.addMayAlias(sourceId, viewId);

  matched.insert(sourceChain->range);
  for (const ContiguousPointerChain &chain :
       {*sourceChain, *destinationChain}) {
    matched.insert(chain.splat);
    matched.insert(chain.addPtr);
  }
  matched.insert(load);
  matched.insert(valueReshape);
  matched.insert(pointerReshape);
  matched.insert(store);
  return success();
}

FailureOr<ContiguousPointerChain> matchBinaryInputLoad(
    triton::LoadOp load, arith::AddFOp add, triton::FuncOp function,
    RankedTensorType resultType, SmallVectorImpl<std::string> &reasons) {
  if (!isDirectlyInEntryBlock(load, function) ||
      load->getNumOperands() != 1 || load->getNumResults() != 1 ||
      load->getNumRegions() != 0 || load->getNumSuccessors() != 0)
    return defer(reasons, "unsupported-elementwise-load");
  if (load.getOther() || !load.getBoundaryCheck().empty() ||
      load.getPadding() || load.getIsVolatile() ||
      load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-load-semantics");
  if (load.getType() != resultType || !load.getResult().hasOneUse() ||
      *load.getResult().getUsers().begin() != add.getOperation() ||
      !load->isBeforeInBlock(add))
    return defer(reasons, "unsupported-elementwise-dataflow");

  FailureOr<ContiguousPointerChain> chain = matchContiguousPointer(
      load.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(chain))
    return failure();
  if (!chain->addPtr.getResult().hasOneUse() ||
      !chain->addPtr->isBeforeInBlock(load))
    return defer(reasons, "load-pointer-has-extra-use");
  return chain;
}

LogicalResult materializeBinaryAdd(
    ArrayRef<triton::LoadOp> loads, arith::AddFOp add,
    MandatoryUBResourceGraph &graph, llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  if (loads.size() != 2 || !add || add->getNumOperands() != 2 ||
      add->getNumResults() != 1 || add->getNumRegions() != 0 ||
      add->getNumSuccessors() != 0 ||
      add.getFastmath() != arith::FastMathFlags::none)
    return defer(reasons, "unsupported-elementwise-op");
  triton::FuncOp function = add->getParentOfType<triton::FuncOp>();
  if (!function || !isDirectlyInEntryBlock(add, function))
    return defer(reasons, "nested-region");
  auto resultType = dyn_cast<RankedTensorType>(add.getType());
  if (!resultType || add.getLhs().getType() != resultType ||
      add.getRhs().getType() != resultType)
    return defer(reasons, "unsupported-elementwise-shape");
  FailureOr<int64_t> payloadBytes = getPayloadBytes(resultType, reasons);
  if (failed(payloadBytes))
    return failure();

  auto lhsLoad = add.getLhs().getDefiningOp<triton::LoadOp>();
  auto rhsLoad = add.getRhs().getDefiningOp<triton::LoadOp>();
  if (!lhsLoad || !rhsLoad || lhsLoad == rhsLoad ||
      !llvm::is_contained(loads, lhsLoad) ||
      !llvm::is_contained(loads, rhsLoad))
    return defer(reasons, "unsupported-elementwise-dataflow");
  FailureOr<ContiguousPointerChain> lhsChain = matchBinaryInputLoad(
      lhsLoad, add, function, resultType, reasons);
  FailureOr<ContiguousPointerChain> rhsChain = matchBinaryInputLoad(
      rhsLoad, add, function, resultType, reasons);
  if (failed(lhsChain) || failed(rhsChain))
    return failure();

  if (!add.getResult().hasOneUse())
    return defer(reasons, "unsupported-elementwise-dataflow");
  auto store = dyn_cast<triton::StoreOp>(*add.getResult().getUsers().begin());
  if (!store || store->getNumOperands() != 2 || store->getNumResults() != 0 ||
      store->getNumRegions() != 0 || store->getNumSuccessors() != 0 ||
      store.getValue() != add.getResult() ||
      !isDirectlyInEntryBlock(store, function) ||
      !add->isBeforeInBlock(store) || !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-elementwise-store");
  FailureOr<ContiguousPointerChain> destinationChain = matchContiguousPointer(
      store.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(destinationChain))
    return failure();
  if (!destinationChain->addPtr.getResult().hasOneUse() ||
      !destinationChain->addPtr->isBeforeInBlock(store) ||
      lhsChain->range != rhsChain->range ||
      lhsChain->range != destinationChain->range ||
      std::distance(lhsChain->range.getResult().use_begin(),
                    lhsChain->range.getResult().use_end()) != 3)
    return defer(reasons, "non-contiguous-pointer");

  const uint64_t addOrdinal =
      std::distance(function.getBody().front().begin(), add->getIterator());
  SmallVector<ResourceId> resourceIds;
  for (triton::LoadOp load : {lhsLoad, rhsLoad}) {
    MandatoryUBResource resource;
    resource.debugName = "binary-add-input";
    resource.minPayloadBytes = *payloadBytes;
    resource.minInstances = 1;
    resource.origin = "tt.load";
    resource.kind = MaterializationKind::GMToUBLoad;
    resource.birth.ordinal =
        std::distance(function.getBody().front().begin(), load->getIterator());
    resource.lastRequiredUse.ordinal = addOrdinal;
    resource.sourceElements = resultType.getNumElements();
    resource.elementBitWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    resource.consumer = "arith.addf";
    resource.contractTrace.push_back("ttir-binary-add-v1");
    ResourceId id = graph.addResource(std::move(resource));
    if (id == InvalidResourceId)
      return defer(reasons, "malformed-resource-graph");
    resourceIds.push_back(id);
  }
  graph.addMayAlias(resourceIds[0], resourceIds[1]);
  CoexistenceWitness witness;
  witness.resources = resourceIds;
  witness.contractTrace.push_back("ttir-binary-add-v1");
  if (graph.addWitness(std::move(witness)) == InvalidWitnessId)
    return defer(reasons, "malformed-resource-graph");

  matched.insert(lhsChain->range);
  for (const ContiguousPointerChain &chain :
       {*lhsChain, *rhsChain, *destinationChain}) {
    matched.insert(chain.splat);
    matched.insert(chain.addPtr);
  }
  matched.insert(lhsLoad);
  matched.insert(rhsLoad);
  matched.insert(add);
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
  Block &entryBlock = functions.front().getBody().front();
  if (entryBlock.empty())
    return defer(unsupportedReasons, "unsupported-module-shape");
  auto returnOp = dyn_cast<triton::ReturnOp>(&entryBlock.back());
  if (!returnOp || returnOp->getNumOperands() != 0 ||
      returnOp->getNumRegions() != 0 || returnOp->getNumSuccessors() != 0 ||
      llvm::count_if(entryBlock, [](Operation &operation) {
        return isa<triton::ReturnOp>(operation);
      }) != 1)
    return defer(unsupportedReasons, "unsupported-module-shape");

  SmallVector<triton::LoadOp> loads;
  module.walk([&](triton::LoadOp load) { loads.push_back(load); });
  if (loads.empty())
    return defer(unsupportedReasons, "no-mandatory-load");

  llvm::DenseSet<Operation *> matched;
  SmallVector<arith::AddFOp> adds;
  module.walk([&](arith::AddFOp add) { adds.push_back(add); });
  if (!adds.empty()) {
    if (adds.size() != 1 ||
        failed(materializeBinaryAdd(loads, adds.front(), graph, matched,
                                    unsupportedReasons)))
      return failure();
  } else {
    for (triton::LoadOp load : loads) {
      triton::ReshapeOp reshape;
      if (load.getResult().hasOneUse())
        reshape = dyn_cast<triton::ReshapeOp>(
            *load.getResult().getUsers().begin());
      LogicalResult result = reshape
                                 ? materializeReshapeCopy(
                                       load, reshape, graph, matched,
                                       unsupportedReasons)
                                 : materializeLoad(load, graph, matched,
                                                   unsupportedReasons);
      if (failed(result))
        return failure();
    }
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
