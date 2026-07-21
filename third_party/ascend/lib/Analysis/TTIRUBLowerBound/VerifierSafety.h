#ifndef TRITON_THIRD_PARTY_ASCEND_LIB_ANALYSIS_TTIRUBLOWERBOUND_VERIFIERSAFETY_H_
#define TRITON_THIRD_PARTY_ASCEND_LIB_ANALYSIS_TTIRUBLOWERBOUND_VERIFIERSAFETY_H_

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir::triton::ascend::ub::detail {

bool hasValidLoadOperandSegments(llvm::ArrayRef<int32_t> segments,
                                 unsigned numOperands);
bool hasValidStoreOperandSegments(llvm::ArrayRef<int32_t> segments,
                                  unsigned numOperands);

} // namespace mlir::triton::ascend::ub::detail

#endif
