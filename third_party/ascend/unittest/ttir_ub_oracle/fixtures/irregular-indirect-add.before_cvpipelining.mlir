module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.0.0">, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
func.func @irregular_indirect_add(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg3, %c0 : memref<?xf32>
  %alloc = memref.alloc(%dim) : memref<?xf32>
  hivm.hir.load ins(%arg3 : memref<?xf32>) outs(%alloc : memref<?xf32>) {"inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  %alloc_0 = memref.alloc() : memref<8xi64>
  hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc_0 : memref<8xi64>) init_out_buffer = false may_implicit_transpose_with_last_axis = false eviction_policy = <EvictFirst> core_type = <VECTOR>
  %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<8xi64>
  %3 = tensor.empty() : tensor<8xf32>
  %4 = hivm.hir.gather_load ins(%alloc : memref<?xf32>, %2 : tensor<8xi64>, %c1_i32 : i32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
  %5 = hivm.hir.vadd ins(%4, %arg6 : tensor<8xf32>, f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
  hivm.hir.store ins(%5 : tensor<8xf32>) outs(%reinterpret_cast_1 : memref<8xf32, strided<[1]>>)
  return
}
}
