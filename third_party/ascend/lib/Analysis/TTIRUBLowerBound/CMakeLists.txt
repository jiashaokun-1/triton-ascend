add_triton_library(TTIRUBLowerBound
  DirectTensorLoadMaterialization.cpp
  MandatoryUBResourceGraph.cpp
  TTIRUBLowerBound.cpp
  UBResourceContract.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TritonIR
)
