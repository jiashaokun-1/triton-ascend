add_triton_library(TTIRUBLowerBound
  MandatoryUBResourceGraph.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TritonIR
)
