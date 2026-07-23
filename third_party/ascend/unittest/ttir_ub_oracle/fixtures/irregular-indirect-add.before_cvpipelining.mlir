// -----// IR Dump Before CVPipelining (cv-pipelining) //----- //
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 1 : i32}, {}, {}, {}, {}, {}, {}, {}], function_type = (memref<?xi8>, memref<?xi8>, memref<?xf32>, memref<?xi64>, memref<?xf32>, f32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "irregular_indirect_add"}> ({
  ^bb0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: f32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "memref.reinterpret_cast"(%arg3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 8>, static_strides = array<i64: 1>}> : (memref<?xi64>) -> memref<8xi64, strided<[1]>>
    %2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xi64>
    "memref.copy"(%1, %2) : (memref<8xi64, strided<[1]>>, memref<8xi64>) -> ()
    %3 = "bufferization.to_tensor"(%2) <{restrict, writable}> : (memref<8xi64>) -> tensor<8xi64>
    %4 = "tensor.empty"() : () -> tensor<8xf32>
    %5 = "hfusion.gather_load"(%arg2, %3, %0, %4) : (memref<?xf32>, tensor<8xi64>, i32, tensor<8xf32>) -> tensor<8xf32>
    %6 = "memref.reinterpret_cast"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 8>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<8xf32, strided<[1]>>
    %7 = "linalg.fill"(%arg5, %4) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      "linalg.yield"(%arg12) : (f32) -> ()
    }) : (f32, tensor<8xf32>) -> tensor<8xf32>
    %8 = "arith.addf"(%5, %7) <{fastmath = #arith.fastmath<none>}> : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    "bufferization.materialize_in_destination"(%8, %6) <{writable}> : (tensor<8xf32>, memref<8xf32, strided<[1]>>) -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "mix_simd_simt"} : () -> ()
}) : () -> ()
