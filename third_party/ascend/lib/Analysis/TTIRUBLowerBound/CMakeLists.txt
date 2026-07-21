add_triton_library(TTIRUBLowerBound
  MandatoryUBResourceGraph.cpp
  UBResourceContract.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TritonIR
)
