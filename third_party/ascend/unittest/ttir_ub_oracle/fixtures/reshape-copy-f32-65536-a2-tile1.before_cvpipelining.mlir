// -----// IR Dump Before CVPipelining (cv-pipelining) //----- //
"func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, {hacc.arg_type = #hacc.arg_type<workspace>}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 1 : i32}, {}, {}, {}], function_type = (i64, memref<?xi8>, memref<?xi8>, memref<?xf32>, memref<?xf32>, i32, i32, i32) -> (), sym_name = "reshape_copy_round_trip"}> ({
^bb0(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32):
  "hivm.hir.set_mask_norm"() : () -> ()
  %0 = "arith.muli"(%arg5, %arg6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %1 = "arith.muli"(%0, %arg7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "annotation.mark"(%1) {logical_block_num} : (i32) -> ()
  %2 = "memref.reinterpret_cast"(%arg3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<65536xf32, strided<[1]>>
  %3 = "memref.reinterpret_cast"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<65536xf32, strided<[1]>>
  %4 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<65536xf32>
  "hivm.hir.load"(%2, %4) <{init_out_buffer = false, may_implicit_transpose_with_last_axis = false, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}> : (memref<65536xf32, strided<[1]>>, memref<65536xf32>) -> ()
  %5 = "bufferization.to_tensor"(%4) <{restrict, writable}> : (memref<65536xf32>) -> tensor<65536xf32>
  "hivm.hir.store"(%5, %3) <{may_implicit_transpose_with_last_axis = false}> : (tensor<65536xf32>, memref<65536xf32, strided<[1]>>) -> ()
  "func.return"() : () -> ()
}) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "simd"} : () -> ()
