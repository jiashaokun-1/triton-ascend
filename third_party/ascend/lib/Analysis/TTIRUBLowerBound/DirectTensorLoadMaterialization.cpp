#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

FailureOr<int64_t>
getStaticTensorPayloadBytes(RankedTensorType type, unsigned expectedRank,
                            SmallVectorImpl<std::string> &reasons) {
  if (!type || !type.hasStaticShape()) {
    addReason(reasons, "dynamic-shape");
    return failure();
  }
  if (type.getRank() != expectedRank) {
    addReason(reasons, "unsupported-shape");
    return failure();
  }
  Type elementType = type.getElementType();
  if (!isa<IntegerType, FloatType>(elementType)) {
    addReason(reasons, "unsupported-element-type");
    return failure();
  }
  const unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth < 8 || (bitWidth != 8 && bitWidth != 16 && bitWidth != 32 &&
                       bitWidth != 64)) {
    addReason(reasons, bitWidth < 8 ? "sub-byte-element-type"
                                   : "unsupported-element-type");
    return failure();
  }
  const int64_t elements = type.getNumElements();
  const int64_t bytesPerElement = llvm::divideCeil(bitWidth, 8u);
  if (elements < 0 ||
      (bytesPerElement != 0 && elements > INT64_MAX / bytesPerElement)) {
    addReason(reasons, "arithmetic-overflow");
    return failure();
  }
  return elements * bytesPerElement;
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
    if (!matched.contains(operation) &&
        isa<triton::ExpandDimsOp, triton::BroadcastOp>(operation)) {
      addReason(reasons, isa<triton::ExpandDimsOp>(operation)
                             ? "unsupported-op-expand-dims"
                             : "unsupported-op-broadcast");
      return WalkResult::interrupt();
    }
    if (isa<triton::MakeRangeOp, triton::SplatOp, triton::AddPtrOp,
            triton::LoadOp, triton::StoreOp, triton::ReshapeOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::DotOp,
            triton::ReduceOp, triton::ReduceReturnOp, arith::AddFOp,
            arith::MulIOp, arith::ConstantOp, math::ExpOp, scf::ForOp,
            scf::YieldOp>(operation)) {
      triton::FuncOp function = operation->getParentOfType<triton::FuncOp>();
      const bool isReduction = isa<triton::ReduceOp>(operation);
      const bool isReductionBody =
          isa<triton::ReduceReturnOp, arith::AddFOp>(operation) &&
          operation->getParentOfType<triton::ReduceOp>();
      const bool isLoop = isa<scf::ForOp>(operation);
      const bool isLoopBody =
          isa<triton::LoadOp, arith::AddFOp, scf::YieldOp>(operation) &&
          operation->getParentOfType<scf::ForOp>();
      const bool validPlacement =
          isReduction
              ? function && isDirectlyInEntryBlock(operation, function) &&
                    operation->getNumRegions() == 1
              : isLoop
                    ? function &&
                          isDirectlyInEntryBlock(operation, function) &&
                          operation->getNumRegions() == 1
                    : isReductionBody || isLoopBody ||
                    (function && isDirectlyInEntryBlock(operation, function) &&
                     operation->getNumRegions() == 0);
      if (matched.contains(operation) && validPlacement &&
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

LogicalResult materializeReductionSum(
    triton::LoadOp load, triton::ReduceOp reduce,
    MandatoryUBResourceGraph &graph, llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  triton::FuncOp function = load->getParentOfType<triton::FuncOp>();
  if (!function || !isDirectlyInEntryBlock(load, function) ||
      !isDirectlyInEntryBlock(reduce, function) ||
      load->getNumOperands() != 1 || load->getNumResults() != 1 ||
      load->getNumRegions() != 0 || load->getNumSuccessors() != 0 ||
      reduce->getNumOperands() != 1 || reduce->getNumResults() != 1 ||
      reduce->getNumRegions() != 1 || reduce->getNumSuccessors() != 0 ||
      reduce.getAxis() != 0)
    return defer(reasons, "unsupported-reduction-shape");
  if (load.getOther() || !load.getBoundaryCheck().empty() ||
      load.getPadding() || load.getIsVolatile() ||
      load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-load-semantics");

  auto inputType = dyn_cast<RankedTensorType>(load.getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 1 ||
      !inputType.getElementType().isF32() ||
      inputType.getNumElements() < 2 ||
      inputType.getNumElements() % 2 != 0 ||
      reduce.getSrcs().front() != load.getResult() ||
      reduce.getResult().front().getType() != inputType.getElementType() ||
      !load.getResult().hasOneUse() ||
      *load.getResult().getUsers().begin() != reduce.getOperation() ||
      !load->isBeforeInBlock(reduce))
    return defer(reasons, "unsupported-reduction-shape");
  FailureOr<int64_t> payloadBytes = getPayloadBytes(inputType, reasons);
  if (failed(payloadBytes))
    return failure();

  FailureOr<ContiguousPointerChain> sourceChain = matchContiguousPointer(
      load.getPtr(), function, inputType.getNumElements(),
      inputType.getElementType(), reasons);
  if (failed(sourceChain))
    return failure();
  if (!sourceChain->addPtr.getResult().hasOneUse() ||
      !sourceChain->addPtr->isBeforeInBlock(load))
    return defer(reasons, "load-pointer-has-extra-use");

  Region &combine = reduce.getCombineOp();
  if (!combine.hasOneBlock())
    return defer(reasons, "unsupported-reduction-combiner");
  Block &block = combine.front();
  if (block.getNumArguments() != 2 ||
      block.getArgument(0).getType() != inputType.getElementType() ||
      block.getArgument(1).getType() != inputType.getElementType())
    return defer(reasons, "unsupported-reduction-combiner");
  auto add = dyn_cast_or_null<arith::AddFOp>(block.empty() ? nullptr
                                                           : &block.front());
  auto reduceReturn =
      dyn_cast_or_null<triton::ReduceReturnOp>(block.empty() ? nullptr
                                                             : &block.back());
  if (!add || !reduceReturn || &block.front() == &block.back() ||
      std::distance(block.begin(), block.end()) != 2 ||
      add->getNumOperands() != 2 || add->getNumResults() != 1 ||
      add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != block.getArgument(0) ||
      add.getRhs() != block.getArgument(1) ||
      reduceReturn->getNumOperands() != 1 ||
      reduceReturn.getResult().front() != add.getResult())
    return defer(reasons, "unsupported-reduction-combiner");

  Value result = reduce.getResult().front();
  if (result.use_empty())
    return defer(reasons, "unsupported-op-reduction");
  if (!result.hasOneUse())
    return defer(reasons, "unsupported-reduction-dataflow");
  auto store = dyn_cast<triton::StoreOp>(*result.getUsers().begin());
  auto destination = dyn_cast<BlockArgument>(store ? store.getPtr() : Value());
  auto destinationType =
      destination
          ? dyn_cast<triton::PointerType>(destination.getType())
          : triton::PointerType();
  if (!store || store->getNumOperands() != 2 || store->getNumResults() != 0 ||
      store->getNumRegions() != 0 || store->getNumSuccessors() != 0 ||
      store.getValue() != result || !isDirectlyInEntryBlock(store, function) ||
      !reduce->isBeforeInBlock(store) || !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL || !destination ||
      destination.getOwner() != &function.getBody().front() ||
      !destination.hasOneUse() || !destinationType ||
      destinationType.getPointeeType() != inputType.getElementType() ||
      destinationType.getAddressSpace() != 1)
    return defer(reasons, "unsupported-reduction-store");

  const uint64_t loadOrdinal =
      std::distance(function.getBody().front().begin(), load->getIterator());
  const uint64_t reduceOrdinal =
      std::distance(function.getBody().front().begin(), reduce->getIterator());
  const uint64_t storeOrdinal =
      std::distance(function.getBody().front().begin(), store->getIterator());
  SmallVector<ResourceId> resourceIds;
  auto addResource = [&](StringRef debugName, int64_t bytes,
                         StringRef origin, MaterializationKind kind,
                         uint64_t birth, uint64_t lastUse,
                         StringRef consumer) -> LogicalResult {
    MandatoryUBResource resource;
    resource.debugName = debugName.str();
    resource.minPayloadBytes = bytes;
    resource.minInstances = 1;
    resource.origin = origin.str();
    resource.kind = kind;
    resource.birth.ordinal = birth;
    resource.lastRequiredUse.ordinal = lastUse;
    resource.sourceElements = inputType.getNumElements();
    resource.elementBitWidth = 32;
    resource.consumer = consumer.str();
    resource.contractTrace.push_back("ttir-reduction-sum-v1");
    ResourceId id = graph.addResource(std::move(resource));
    if (id == InvalidResourceId)
      return failure();
    resourceIds.push_back(id);
    return success();
  };
  if (failed(addResource("reduction-sum-input", *payloadBytes, "tt.load",
                         MaterializationKind::GMToUBLoad, loadOrdinal,
                         reduceOrdinal, "tt.reduce")) ||
      failed(addResource("reduction-sum-scratch", *payloadBytes / 2,
                         "tt.reduce", MaterializationKind::ReductionScratch,
                         reduceOrdinal, reduceOrdinal, "tt.reduce")) ||
      failed(addResource("reduction-sum-accumulator", 4, "tt.reduce",
                         MaterializationKind::ReductionAccumulator,
                         reduceOrdinal, storeOrdinal, "tt.store")))
    return defer(reasons, "malformed-resource-graph");
  for (size_t lhs = 0; lhs < resourceIds.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < resourceIds.size(); ++rhs)
      graph.addMayAlias(resourceIds[lhs], resourceIds[rhs]);
  CoexistenceWitness witness;
  witness.resources = resourceIds;
  witness.contractTrace.push_back("ttir-reduction-sum-v1");
  if (graph.addWitness(std::move(witness)) == InvalidWitnessId)
    return defer(reasons, "malformed-resource-graph");

  matched.insert(sourceChain->range);
  matched.insert(sourceChain->splat);
  matched.insert(sourceChain->addPtr);
  matched.insert(load);
  matched.insert(reduce);
  matched.insert(add);
  matched.insert(reduceReturn);
  matched.insert(store);
  return success();
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
  if (!load.getResult().hasOneUse()) {
    for (Operation *user : load.getResult().getUsers()) {
      auto expand = dyn_cast<triton::ExpandDimsOp>(user);
      if (!expand)
        continue;
      if (llvm::any_of(expand.getResult().getUsers(), [](Operation *nested) {
            return isa<triton::BroadcastOp>(nested);
          }))
        return defer(reasons, "unsupported-op-broadcast");
      return defer(reasons, "unsupported-op-expand-dims");
    }
    return defer(reasons, "load-has-extra-use");
  }
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

  Value storedValue = valueReshape.getResult();
  triton::ReshapeOp inverseReshape;
  Operation *valueUser = *valueReshape.getResult().getUsers().begin();
  if (auto candidate = dyn_cast<triton::ReshapeOp>(valueUser)) {
    if (!isStrictNoReorderReshape(candidate, valueReshape.getResult(),
                                  numElements, elementType, 1, function) ||
        candidate.getType() != sourceType ||
        !valueReshape->isBeforeInBlock(candidate))
      return defer(reasons, "unsupported-view-dataflow");
    inverseReshape = candidate;
    storedValue = candidate.getResult();
    valueUser = *candidate.getResult().getUsers().begin();
  }

  auto store = dyn_cast<triton::StoreOp>(valueUser);
  if (!store || store->getNumOperands() != 2 || store->getNumResults() != 0 ||
      store->getNumRegions() != 0 || store->getNumSuccessors() != 0 ||
      store.getValue() != storedValue ||
      !isDirectlyInEntryBlock(store, function) ||
      !(inverseReshape ? inverseReshape->isBeforeInBlock(store)
                       : valueReshape->isBeforeInBlock(store)) ||
      !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-view-store");

  auto pointerReshape = store.getPtr().getDefiningOp<triton::ReshapeOp>();
  auto valueResultType = dyn_cast<RankedTensorType>(valueReshape.getType());
  if (!valueResultType)
    return defer(reasons, "unsupported-view-shape");

  Value destinationPointer = store.getPtr();
  if (pointerReshape) {
    auto pointerResultType =
        dyn_cast<RankedTensorType>(pointerReshape.getType());
    auto pointerElementType =
        pointerResultType
            ? dyn_cast<triton::PointerType>(pointerResultType.getElementType())
            : triton::PointerType();
    if (!pointerResultType || !pointerElementType ||
        pointerResultType.getShape() != valueResultType.getShape() ||
        pointerElementType.getPointeeType() != elementType ||
        pointerElementType.getAddressSpace() != 1 ||
        !isStrictNoReorderReshape(
            pointerReshape, pointerReshape.getSrc(), numElements,
            pointerResultType.getElementType(), 2, function) ||
        !pointerReshape->isBeforeInBlock(store))
      return defer(reasons, "unsupported-view-pointer");
    destinationPointer = pointerReshape.getSrc();
  } else if (!inverseReshape) {
    // A rank-changing value must be paired either with an identically-shaped
    // pointer view or with a strict inverse value reshape before a flat store.
    return defer(reasons, "unsupported-view-pointer");
  }

  FailureOr<ContiguousPointerChain> destinationChain = matchContiguousPointer(
      destinationPointer, function, numElements, elementType, reasons);
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
  viewResource.lastRequiredUse.ordinal = std::distance(
      function.getBody().front().begin(),
      inverseReshape ? inverseReshape->getIterator() : store->getIterator());
  viewResource.sourceElements = numElements;
  viewResource.elementBitWidth = elementType.getIntOrFloatBitWidth();
  viewResource.consumer = inverseReshape ? "tt.reshape" : "tt.store";
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
  if (inverseReshape)
    matched.insert(inverseReshape);
  if (pointerReshape)
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

FailureOr<int64_t> getIndexConstant(Value value, triton::FuncOp function,
                                    SmallVectorImpl<std::string> &reasons) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  if (!constant || !integer || !constant.getType().isIndex() ||
      !isDirectlyInEntryBlock(constant, function) ||
      constant->getNumOperands() != 0 || constant->getNumResults() != 1 ||
      constant->getNumRegions() != 0 || constant->getNumSuccessors() != 0 ||
      !constant.getResult().hasOneUse())
    return defer(reasons, "unsupported-loop-bounds");
  return integer.getInt();
}

LogicalResult materializeLoopCarriedAdd(
    ArrayRef<triton::LoadOp> loads, scf::ForOp loop,
    MandatoryUBResourceGraph &graph, llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  triton::FuncOp function = loop->getParentOfType<triton::FuncOp>();
  if (!function || !isDirectlyInEntryBlock(loop, function) ||
      loads.size() != 2 || loop->getNumOperands() != 4 ||
      loop->getNumResults() != 1 || loop->getNumRegions() != 1 ||
      loop->getNumSuccessors() != 0 || loop.getInitArgs().size() != 1 ||
      !loop.getRegion().hasOneBlock())
    return defer(reasons, "unsupported-loop-shape");

  FailureOr<int64_t> lower =
      getIndexConstant(loop.getLowerBound(), function, reasons);
  FailureOr<int64_t> upper =
      getIndexConstant(loop.getUpperBound(), function, reasons);
  FailureOr<int64_t> step =
      getIndexConstant(loop.getStep(), function, reasons);
  if (failed(lower) || failed(upper) || failed(step) || *step <= 0 ||
      *lower >= *upper)
    return defer(reasons, "unsupported-loop-bounds");
  int64_t signedDistance = 0;
  if (llvm::SubOverflow(*upper, *lower, signedDistance))
    return defer(reasons, "arithmetic-overflow");
  const uint64_t distance = static_cast<uint64_t>(signedDistance);
  const uint64_t positiveStep = static_cast<uint64_t>(*step);
  const uint64_t tripCount =
      distance / positiveStep + (distance % positiveStep != 0);
  if (tripCount < 2)
    return defer(reasons, "unsupported-loop-trip-count");

  Block &body = loop.getRegion().front();
  if (body.getNumArguments() != 2 ||
      body.getArgument(0) != loop.getInductionVar() ||
      loop.getRegionIterArgs().size() != 1 ||
      body.getArgument(1) != loop.getRegionIterArgs().front() ||
      std::distance(body.begin(), body.end()) != 3)
    return defer(reasons, "unsupported-loop-shape");
  auto stepLoad = dyn_cast<triton::LoadOp>(&body.front());
  auto add =
      dyn_cast_or_null<arith::AddFOp>(stepLoad ? stepLoad->getNextNode()
                                               : nullptr);
  auto yield =
      dyn_cast_or_null<scf::YieldOp>(add ? add->getNextNode() : nullptr);
  if (!stepLoad || !add || !yield || &body.back() != yield.getOperation() ||
      add->getNumOperands() != 2 || add->getNumResults() != 1 ||
      add->getNumRegions() != 0 || add->getNumSuccessors() != 0 ||
      add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != loop.getRegionIterArgs().front() ||
      add.getRhs() != stepLoad.getResult() ||
      yield->getNumOperands() != 1 || yield->getNumResults() != 0 ||
      yield->getNumRegions() != 0 || yield->getNumSuccessors() != 0 ||
      yield.getResults().front() != add.getResult())
    return defer(reasons, "unsupported-loop-body");

  triton::LoadOp initLoad;
  for (triton::LoadOp load : loads) {
    if (load == stepLoad)
      continue;
    if (initLoad)
      return defer(reasons, "unsupported-loop-dataflow");
    initLoad = load;
  }
  if (!initLoad || !isDirectlyInEntryBlock(initLoad, function) ||
      initLoad->getNumOperands() != 1 || initLoad->getNumResults() != 1 ||
      initLoad->getNumRegions() != 0 || initLoad->getNumSuccessors() != 0 ||
      stepLoad->getNumOperands() != 1 || stepLoad->getNumResults() != 1 ||
      stepLoad->getNumRegions() != 0 || stepLoad->getNumSuccessors() != 0)
    return defer(reasons, "unsupported-loop-load");
  auto hasSupportedLoadSemantics = [](triton::LoadOp load) {
    return !load.getOther() && load.getBoundaryCheck().empty() &&
           !load.getPadding() && !load.getIsVolatile() &&
           load.getCache() == triton::CacheModifier::NONE &&
           load.getEvict() == triton::EvictionPolicy::NORMAL;
  };
  if (!hasSupportedLoadSemantics(initLoad) ||
      !hasSupportedLoadSemantics(stepLoad))
    return defer(reasons, "unsupported-load-semantics");

  auto resultType = dyn_cast<RankedTensorType>(loop.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      resultType.getRank() != 1 || !resultType.getElementType().isF32() ||
      initLoad.getType() != resultType || stepLoad.getType() != resultType ||
      loop.getInitArgs().front() != initLoad.getResult() ||
      loop.getRegionIterArgs().front().getType() != resultType ||
      add.getType() != resultType || !initLoad.getResult().hasOneUse() ||
      !stepLoad.getResult().hasOneUse() ||
      *stepLoad.getResult().getUsers().begin() != add.getOperation() ||
      !loop.getResult(0).hasOneUse())
    return defer(reasons, "unsupported-loop-dataflow");
  FailureOr<int64_t> payloadBytes = getPayloadBytes(resultType, reasons);
  if (failed(payloadBytes))
    return failure();

  auto store =
      dyn_cast<triton::StoreOp>(*loop.getResult(0).getUsers().begin());
  if (!store || store->getNumOperands() != 2 || store->getNumResults() != 0 ||
      store->getNumRegions() != 0 || store->getNumSuccessors() != 0 ||
      store.getValue() != loop.getResult(0) ||
      !isDirectlyInEntryBlock(store, function) ||
      !loop->isBeforeInBlock(store) || !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return defer(reasons, "unsupported-loop-store");

  FailureOr<ContiguousPointerChain> initChain = matchContiguousPointer(
      initLoad.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  FailureOr<ContiguousPointerChain> stepChain = matchContiguousPointer(
      stepLoad.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  FailureOr<ContiguousPointerChain> destinationChain = matchContiguousPointer(
      store.getPtr(), function, resultType.getNumElements(),
      resultType.getElementType(), reasons);
  if (failed(initChain) || failed(stepChain) || failed(destinationChain))
    return failure();
  if (!initChain->addPtr.getResult().hasOneUse() ||
      !stepChain->addPtr.getResult().hasOneUse() ||
      !destinationChain->addPtr.getResult().hasOneUse() ||
      !initChain->addPtr->isBeforeInBlock(initLoad) ||
      !stepChain->addPtr->isBeforeInBlock(loop) ||
      !destinationChain->addPtr->isBeforeInBlock(store) ||
      initChain->range != stepChain->range ||
      initChain->range != destinationChain->range ||
      std::distance(initChain->range.getResult().use_begin(),
                    initChain->range.getResult().use_end()) != 3)
    return defer(reasons, "non-contiguous-pointer");

  const uint64_t initOrdinal = std::distance(
      function.getBody().front().begin(), initLoad->getIterator());
  const uint64_t loopOrdinal = std::distance(
      function.getBody().front().begin(), loop->getIterator());
  const uint64_t storeOrdinal = std::distance(
      function.getBody().front().begin(), store->getIterator());
  if (!(initOrdinal < loopOrdinal && loopOrdinal < storeOrdinal))
    return defer(reasons, "unsupported-loop-dataflow");

  SmallVector<ResourceId> resourceIds;
  auto addResource = [&](StringRef name, uint64_t birth, uint64_t lastUse,
                         StringRef consumer) -> LogicalResult {
    MandatoryUBResource resource;
    resource.debugName = name.str();
    resource.minPayloadBytes = *payloadBytes;
    resource.minInstances = 1;
    resource.origin = "tt.load";
    resource.kind = MaterializationKind::GMToUBLoad;
    resource.birth.ordinal = birth;
    resource.lastRequiredUse.ordinal = lastUse;
    resource.sourceElements = resultType.getNumElements();
    resource.elementBitWidth = 32;
    resource.consumer = consumer.str();
    resource.contractTrace.push_back("ttir-loop-carried-add-v1");
    ResourceId id = graph.addResource(std::move(resource));
    if (id == InvalidResourceId)
      return failure();
    resourceIds.push_back(id);
    return success();
  };
  if (failed(addResource("loop-carried-accumulator", initOrdinal,
                         storeOrdinal, "scf.for")) ||
      failed(addResource("loop-step-input", loopOrdinal, loopOrdinal,
                         "arith.addf")))
    return defer(reasons, "malformed-resource-graph");
  graph.addMayAlias(resourceIds[0], resourceIds[1]);
  CoexistenceWitness witness;
  witness.resources = resourceIds;
  witness.contractTrace.push_back("ttir-loop-carried-add-v1");
  if (graph.addWitness(std::move(witness)) == InvalidWitnessId)
    return defer(reasons, "malformed-resource-graph");

  matched.insert(initChain->range);
  for (const ContiguousPointerChain &chain :
       {*initChain, *stepChain, *destinationChain}) {
    matched.insert(chain.splat);
    matched.insert(chain.addPtr);
  }
  for (Value bound : {loop.getLowerBound(), loop.getUpperBound(),
                      loop.getStep()})
    matched.insert(bound.getDefiningOp());
  matched.insert(initLoad);
  matched.insert(stepLoad);
  matched.insert(loop);
  matched.insert(add);
  matched.insert(yield);
  matched.insert(store);
  return success();
}

BlockArgument tracePointerBase(Value pointer) {
  llvm::DenseSet<Value> visited;
  while (pointer && visited.insert(pointer).second) {
    if (auto argument = dyn_cast<BlockArgument>(pointer))
      return argument;
    if (auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>()) {
      pointer = addPtr.getPtr();
      continue;
    }
    if (auto broadcast = pointer.getDefiningOp<triton::BroadcastOp>()) {
      pointer = broadcast.getSrc();
      continue;
    }
    if (auto splat = pointer.getDefiningOp<triton::SplatOp>()) {
      pointer = splat.getSrc();
      continue;
    }
    break;
  }
  return {};
}

bool hasPlainLoadSemantics(triton::LoadOp load) {
  return load && load->getNumOperands() == 1 &&
         load->getNumResults() == 1 && load->getNumRegions() == 0 &&
         load->getNumSuccessors() == 0 && !load.getOther() &&
         load.getBoundaryCheck().empty() && !load.getPadding() &&
         !load.getIsVolatile() &&
         load.getCache() == triton::CacheModifier::NONE &&
         load.getEvict() == triton::EvictionPolicy::NORMAL;
}

bool hasPlainStoreSemantics(triton::StoreOp store) {
  return store && store->getNumOperands() == 2 &&
         store->getNumResults() == 0 && store->getNumRegions() == 0 &&
         store->getNumSuccessors() == 0 &&
         store.getBoundaryCheck().empty() &&
         store.getCache() == triton::CacheModifier::NONE &&
         store.getEvict() == triton::EvictionPolicy::NORMAL;
}

template <typename OpTy>
SmallVector<OpTy> collectOperations(ModuleOp module) {
  SmallVector<OpTy> result;
  module.walk([&](OpTy operation) { result.push_back(operation); });
  return result;
}

LogicalResult materializeDynamicCVDotExp(
    ModuleOp module, triton::DotOp dot, MandatoryUBResourceGraph &graph,
    llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  triton::FuncOp function = dot->getParentOfType<triton::FuncOp>();
  if (!function || !isDirectlyInEntryBlock(dot, function) ||
      function.getBody().front().getNumArguments() != 3 ||
      dot->getNumOperands() != 3 || dot->getNumResults() != 1 ||
      dot->getNumRegions() != 0 || dot->getNumSuccessors() != 0)
    return defer(reasons, "unsupported-dynamic-cv-shape");

  auto aType = dyn_cast<RankedTensorType>(dot->getOperand(0).getType());
  auto bType = dyn_cast<RankedTensorType>(dot->getOperand(1).getType());
  auto cType = dyn_cast<RankedTensorType>(dot->getOperand(2).getType());
  auto outputType = dyn_cast<RankedTensorType>(dot->getResult(0).getType());
  if (!aType || !bType || !cType || !outputType ||
      !aType.hasStaticShape() || !bType.hasStaticShape() ||
      !cType.hasStaticShape() || !outputType.hasStaticShape() ||
      aType.getRank() != 2 || bType.getRank() != 2 ||
      cType.getRank() != 2 || outputType.getRank() != 2 ||
      !aType.getElementType().isF32() || !bType.getElementType().isF32() ||
      !cType.getElementType().isF32() ||
      !outputType.getElementType().isF32() ||
      aType.getDimSize(0) != outputType.getDimSize(0) ||
      bType.getDimSize(1) != outputType.getDimSize(1) ||
      aType.getDimSize(1) != bType.getDimSize(0) ||
      cType != outputType)
    return defer(reasons, "unsupported-dynamic-cv-shape");
  FailureOr<int64_t> payload =
      getStaticTensorPayloadBytes(outputType, 2, reasons);
  if (failed(payload))
    return failure();

  auto lhsLoad = dot->getOperand(0).getDefiningOp<triton::LoadOp>();
  auto rhsLoad = dot->getOperand(1).getDefiningOp<triton::LoadOp>();
  auto accumulator =
      dot->getOperand(2).getDefiningOp<arith::ConstantOp>();
  if (!lhsLoad || !rhsLoad || lhsLoad == rhsLoad ||
      !hasPlainLoadSemantics(lhsLoad) ||
      !hasPlainLoadSemantics(rhsLoad) ||
      lhsLoad.getType() != aType || rhsLoad.getType() != bType ||
      !lhsLoad.getResult().hasOneUse() || !rhsLoad.getResult().hasOneUse() ||
      !accumulator || accumulator.getType() != cType)
    return defer(reasons, "unsupported-dynamic-cv-dataflow");

  auto denseAccumulator =
      dyn_cast<DenseElementsAttr>(accumulator.getValue());
  if (!denseAccumulator || !denseAccumulator.isSplat() ||
      !denseAccumulator.getSplatValue<APFloat>().isZero())
    return defer(reasons, "unsupported-dynamic-cv-accumulator");

  if (!dot.getResult().hasOneUse())
    return defer(reasons, "unsupported-dynamic-cv-dataflow");
  auto exp = dyn_cast<math::ExpOp>(*dot.getResult().getUsers().begin());
  if (!exp || !isDirectlyInEntryBlock(exp, function) ||
      exp->getNumOperands() != 1 || exp->getNumResults() != 1 ||
      exp->getNumRegions() != 0 || exp->getNumSuccessors() != 0 ||
      exp.getOperand().getType() != outputType ||
      exp.getResult().getType() != outputType ||
      exp.getFastmath() != arith::FastMathFlags::none ||
      !exp.getResult().hasOneUse())
    return defer(reasons, "unsupported-dynamic-cv-vector-stage");
  auto store = dyn_cast<triton::StoreOp>(*exp.getResult().getUsers().begin());
  if (!hasPlainStoreSemantics(store) || store.getValue() != exp.getResult() ||
      !isDirectlyInEntryBlock(store, function) ||
      !exp->isBeforeInBlock(store))
    return defer(reasons, "unsupported-dynamic-cv-store");

  Block &entry = function.getBody().front();
  BlockArgument lhsBase = tracePointerBase(lhsLoad.getPtr());
  BlockArgument rhsBase = tracePointerBase(rhsLoad.getPtr());
  BlockArgument destinationBase = tracePointerBase(store.getPtr());
  if (!lhsBase || !rhsBase || !destinationBase ||
      lhsBase != entry.getArgument(0) || rhsBase != entry.getArgument(1) ||
      destinationBase != entry.getArgument(2))
    return defer(reasons, "unsupported-dynamic-cv-pointer");

  const auto constants = collectOperations<arith::ConstantOp>(module);
  const auto ranges = collectOperations<triton::MakeRangeOp>(module);
  const auto expands = collectOperations<triton::ExpandDimsOp>(module);
  const auto multiplies = collectOperations<arith::MulIOp>(module);
  const auto splats = collectOperations<triton::SplatOp>(module);
  const auto broadcasts = collectOperations<triton::BroadcastOp>(module);
  const auto addPtrs = collectOperations<triton::AddPtrOp>(module);
  const auto loads = collectOperations<triton::LoadOp>(module);
  const auto dots = collectOperations<triton::DotOp>(module);
  const auto exps = collectOperations<math::ExpOp>(module);
  const auto stores = collectOperations<triton::StoreOp>(module);
  if (constants.size() != 2 || ranges.size() != 1 || expands.size() != 2 ||
      multiplies.size() != 1 || splats.size() != 3 ||
      broadcasts.size() != 4 || addPtrs.size() != 6 || loads.size() != 2 ||
      dots.size() != 1 || exps.size() != 1 || stores.size() != 1)
    return defer(reasons, "unsupported-dynamic-cv-structure");

  for (Operation &operation : entry.without_terminator()) {
    if (!isDirectlyInEntryBlock(&operation, function))
      return defer(reasons, "unsupported-dynamic-cv-structure");
    matched.insert(&operation);
  }

  const uint64_t dotOrdinal =
      std::distance(entry.begin(), dot->getIterator());
  const uint64_t expOrdinal =
      std::distance(entry.begin(), exp->getIterator());
  const uint64_t storeOrdinal =
      std::distance(entry.begin(), store->getIterator());
  MandatoryUBResource fixpipe;
  fixpipe.debugName = "dynamic-cv-fixpipe-output";
  fixpipe.minPayloadBytes = *payload;
  fixpipe.minInstances = 1;
  fixpipe.origin = "tt.dot";
  fixpipe.executionScope = UBExecutionScope::AIC;
  fixpipe.kind = MaterializationKind::DynamicCVFixpipeOutput;
  fixpipe.birth.ordinal = dotOrdinal;
  fixpipe.lastRequiredUse.ordinal = expOrdinal;
  fixpipe.sourceElements = outputType.getNumElements();
  fixpipe.elementBitWidth = 32;
  fixpipe.consumer = "math.exp";
  fixpipe.contractTrace.push_back("ttir-dynamic-cv-dot-exp-v1");
  ResourceId fixpipeId = graph.addResource(std::move(fixpipe));

  MandatoryUBResource vector;
  vector.debugName = "dynamic-cv-vector-output";
  vector.minPayloadBytes = *payload;
  vector.minInstances = 1;
  vector.origin = "math.exp";
  vector.executionScope = UBExecutionScope::AIV;
  vector.kind = MaterializationKind::DynamicCVVectorOutput;
  vector.birth.ordinal = expOrdinal;
  vector.lastRequiredUse.ordinal = storeOrdinal;
  vector.sourceElements = outputType.getNumElements();
  vector.elementBitWidth = 32;
  vector.consumer = "tt.store";
  vector.contractTrace.push_back("ttir-dynamic-cv-dot-exp-v1");
  ResourceId vectorId = graph.addResource(std::move(vector));
  if (fixpipeId == InvalidResourceId || vectorId == InvalidResourceId)
    return defer(reasons, "malformed-resource-graph");
  // Dynamic CV splits this logical producer/consumer edge across MIX AIC and
  // AIV functions.  Their UB allocations are physically independent: there
  // is neither an alias relation nor a cross-core coexistence witness.
  return success();
}

LogicalResult materializeIrregularIndirectAdd(
    ModuleOp module, ArrayRef<triton::LoadOp> loads, arith::AddFOp add,
    MandatoryUBResourceGraph &graph, llvm::DenseSet<Operation *> &matched,
    SmallVectorImpl<std::string> &reasons) {
  triton::FuncOp function = add->getParentOfType<triton::FuncOp>();
  if (!function || !isDirectlyInEntryBlock(add, function) ||
      function.getBody().front().getNumArguments() != 4 ||
      loads.size() != 2 || add->getNumOperands() != 2 ||
      add->getNumResults() != 1 || add->getNumRegions() != 0 ||
      add->getNumSuccessors() != 0 ||
      add.getFastmath() != arith::FastMathFlags::none)
    return defer(reasons, "unsupported-irregular-shape");

  triton::LoadOp indexLoad;
  triton::LoadOp valueLoad;
  for (triton::LoadOp load : loads) {
    auto type = dyn_cast<RankedTensorType>(load.getType());
    if (type && type.hasStaticShape() && type.getRank() == 1 &&
        type.getElementType().isInteger(64))
      indexLoad = load;
    else if (type && type.hasStaticShape() && type.getRank() == 1 &&
             type.getElementType().isF32())
      valueLoad = load;
  }
  if (!indexLoad || !valueLoad || !hasPlainLoadSemantics(indexLoad) ||
      !hasPlainLoadSemantics(valueLoad))
    return defer(reasons, "unsupported-irregular-load");
  auto indexType = cast<RankedTensorType>(indexLoad.getType());
  auto valueType = cast<RankedTensorType>(valueLoad.getType());
  if (indexType.getNumElements() != valueType.getNumElements())
    return defer(reasons, "unsupported-irregular-shape");

  auto indirect = valueLoad.getPtr().getDefiningOp<triton::AddPtrOp>();
  auto valueBaseSplat =
      indirect ? indirect.getPtr().getDefiningOp<triton::SplatOp>()
               : triton::SplatOp();
  Block &entry = function.getBody().front();
  if (!indirect || !valueBaseSplat ||
      indirect.getOffset() != indexLoad.getResult() ||
      !indexLoad.getResult().hasOneUse() ||
      !indirect.getResult().hasOneUse() ||
      !valueLoad.getResult().hasOneUse() ||
      valueBaseSplat.getSrc() != entry.getArgument(0))
    return defer(reasons, "unsupported-irregular-pointer");

  Value scalarValue;
  Value gatheredValue = valueLoad.getResult();
  if (add.getLhs() == gatheredValue)
    scalarValue = add.getRhs();
  else if (add.getRhs() == gatheredValue)
    scalarValue = add.getLhs();
  else
    return defer(reasons, "unsupported-irregular-dataflow");
  auto scalarSplat = scalarValue.getDefiningOp<triton::SplatOp>();
  if (!scalarSplat || scalarSplat.getSrc() != entry.getArgument(3) ||
      !scalarSplat.getResult().hasOneUse() || !add.getResult().hasOneUse())
    return defer(reasons, "unsupported-irregular-dataflow");

  auto store = dyn_cast<triton::StoreOp>(*add.getResult().getUsers().begin());
  if (!hasPlainStoreSemantics(store) || store.getValue() != add.getResult() ||
      !isDirectlyInEntryBlock(store, function))
    return defer(reasons, "unsupported-irregular-store");
  BlockArgument indexBase = tracePointerBase(indexLoad.getPtr());
  BlockArgument destinationBase = tracePointerBase(store.getPtr());
  if (!indexBase || !destinationBase ||
      indexBase != entry.getArgument(1) ||
      destinationBase != entry.getArgument(2))
    return defer(reasons, "unsupported-irregular-pointer");

  const auto ranges = collectOperations<triton::MakeRangeOp>(module);
  const auto splats = collectOperations<triton::SplatOp>(module);
  const auto addPtrs = collectOperations<triton::AddPtrOp>(module);
  const auto adds = collectOperations<arith::AddFOp>(module);
  const auto stores = collectOperations<triton::StoreOp>(module);
  if (ranges.size() != 1 || splats.size() != 4 ||
      addPtrs.size() != 3 || loads.size() != 2 || adds.size() != 1 ||
      stores.size() != 1)
    return defer(reasons, "unsupported-irregular-structure");
  // The source pointer carries no allocation extent in this TTIR form.  The
  // real full-compiler boundary therefore materializes memref<?xf32> before
  // gather_load, and PlanMemory cannot derive a static UB size.  The 8-element
  // index/result tensors alone are not a safe lower-bound model of the
  // mandatory source buffer.
  for (Operation &operation : entry.without_terminator())
    matched.insert(&operation);
  return defer(reasons, "unsupported-irregular-source-extent");
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
  SmallVector<triton::ReduceOp> reductions;
  module.walk([&](triton::ReduceOp reduce) { reductions.push_back(reduce); });
  SmallVector<arith::AddFOp> adds;
  module.walk([&](arith::AddFOp add) { adds.push_back(add); });
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  SmallVector<triton::DotOp> dots;
  module.walk([&](triton::DotOp dot) { dots.push_back(dot); });
  const bool irregular =
      llvm::any_of(loads, [](triton::LoadOp load) {
        auto addPtr = load.getPtr().getDefiningOp<triton::AddPtrOp>();
        return addPtr &&
               static_cast<bool>(
                   addPtr.getOffset().getDefiningOp<triton::LoadOp>());
      });
  if (!dots.empty()) {
    if (dots.size() != 1 || loops.size() != 0 || reductions.size() != 0 ||
        adds.size() != 0 ||
        failed(materializeDynamicCVDotExp(module, dots.front(), graph, matched,
                                          unsupportedReasons)))
      return failure();
  } else if (!loops.empty()) {
    if (loops.size() != 1 || reductions.size() != 0 || adds.size() != 1 ||
        failed(materializeLoopCarriedAdd(loads, loops.front(), graph, matched,
                                         unsupportedReasons)))
      return failure();
  } else if (!reductions.empty()) {
    if (reductions.size() != 1 || loads.size() != 1 ||
        failed(materializeReductionSum(loads.front(), reductions.front(),
                                       graph, matched, unsupportedReasons)))
      return failure();
  } else if (!adds.empty()) {
    if (adds.size() != 1)
      return failure();
    LogicalResult result =
        irregular
            ? materializeIrregularIndirectAdd(module, loads, adds.front(),
                                              graph, matched,
                                              unsupportedReasons)
            : materializeBinaryAdd(loads, adds.front(), graph, matched,
                                   unsupportedReasons);
    if (failed(result))
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
