// -----// IR Dump Before CVPipelining (cv-pipelining) //----- //
"func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, {hacc.arg_type = #hacc.arg_type<workspace>}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 1 : i32}, {}, {}, {}], function_type = (i64, memref<?xi8>, memref<?xi8>, memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32, i32) -> (), sym_name = "add"}> ({
^bb0(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: i32, %arg7: i32, %arg8: i32):
  "hivm.hir.set_mask_norm"() : () -> ()
  %0 = "arith.muli"(%arg6, %arg7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %1 = "arith.muli"(%0, %arg8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "annotation.mark"(%1) {logical_block_num} : (i32) -> ()
  %2 = "memref.reinterpret_cast"(%arg3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<65536xf32, strided<[1]>>
  %3 = "memref.reinterpret_cast"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<65536xf32, strided<[1]>>
  %4 = "memref.reinterpret_cast"(%arg5) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<65536xf32, strided<[1]>>
  %5 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<65536xf32>
  "hivm.hir.load"(%2, %5) <{init_out_buffer = false, may_implicit_transpose_with_last_axis = false, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}> : (memref<65536xf32, strided<[1]>>, memref<65536xf32>) -> ()
  %6 = "bufferization.to_tensor"(%5) <{restrict, writable}> : (memref<65536xf32>) -> tensor<65536xf32>
  %7 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<65536xf32>
  "hivm.hir.load"(%3, %7) <{init_out_buffer = false, may_implicit_transpose_with_last_axis = false, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}> : (memref<65536xf32, strided<[1]>>, memref<65536xf32>) -> ()
  %8 = "bufferization.to_tensor"(%7) <{restrict, writable}> : (memref<65536xf32>) -> tensor<65536xf32>
  %9 = "tensor.empty"() : () -> tensor<65536xf32>
  %10 = "hivm.hir.vadd"(%6, %8, %9) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (tensor<65536xf32>, tensor<65536xf32>, tensor<65536xf32>) -> tensor<65536xf32>
  "hivm.hir.store"(%10, %4) <{may_implicit_transpose_with_last_axis = false}> : (tensor<65536xf32>, memref<65536xf32, strided<[1]>>) -> ()
  "func.return"() : () -> ()
}) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "simd"} : () -> ()

