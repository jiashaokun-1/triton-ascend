// -----// IR Dump Before CVPipelining (cv-pipelining) //----- //
"func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, {hacc.arg_type = #hacc.arg_type<workspace>}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 1 : i32}, {}, {}, {}], function_type = (i64, memref<?xi8>, memref<?xi8>, memref<?xf32>, memref<?xf32>, i32, i32, i32) -> (), sym_name = "reduction_sum"}> ({
^bb0(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32):
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  "hivm.hir.set_mask_norm"() : () -> ()
  %1 = "arith.muli"(%arg5, %arg6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %2 = "arith.muli"(%1, %arg7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "annotation.mark"(%2) {logical_block_num} : (i32) -> ()
  %3 = "memref.reinterpret_cast"(%arg3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<65536xf32, strided<[1]>>
  %4 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<65536xf32>
  "hivm.hir.load"(%3, %4) <{init_out_buffer = false, may_implicit_transpose_with_last_axis = false, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}> : (memref<65536xf32, strided<[1]>>, memref<65536xf32>) -> ()
  %5 = "bufferization.to_tensor"(%4) <{restrict, writable}> : (memref<65536xf32>) -> tensor<65536xf32>
  %6 = "bufferization.alloc_tensor"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> tensor<f32>
  %7 = "tensor.empty"() : () -> tensor<1xf32>
  %8 = "hivm.hir.vreduce"(%5, %7) <{arith = #hivm.reduce_op<sum>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, reduce_dims = array<i64: 0>}> : (tensor<65536xf32>, tensor<1xf32>) -> tensor<1xf32>
  %9 = "tensor.extract"(%8, %0) : (tensor<1xf32>, index) -> f32
  %10 = "tensor.empty"() : () -> tensor<1xf32>
  %11 = "tensor.insert"(%9, %10, %0) : (f32, tensor<1xf32>, index) -> tensor<1xf32>
  %12 = "memref.reinterpret_cast"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<?xf32>) -> memref<1xf32, strided<[1]>>
  "hivm.hir.store"(%11, %12) <{may_implicit_transpose_with_last_axis = false}> : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
  "func.return"() : () -> ()
}) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "simd"} : () -> ()
