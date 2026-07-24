module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.0.0">, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
func.func @dynamic_cv_mix_dot_exp(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  scope.scope : () -> () {
    hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %2 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x16xf32>
    %3 = tensor.empty() : tensor<16x16xf32>
    %4 = hivm.hir.vexp ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32} : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    hivm.hir.store ins(%4 : tensor<16x16xf32>) outs(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>)
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  scope.scope : () -> () {
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc : memref<16x16xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false eviction_policy = <EvictFirst> core_type = <CUBE>
    %2 = bufferization.to_tensor %alloc restrict writable {gm_load_bufferable, ssbuffer.block_id = 0 : i32} : memref<16x16xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_1 = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false eviction_policy = <EvictFirst> core_type = <CUBE>
    %3 = bufferization.to_tensor %alloc_1 restrict writable {gm_load_bufferable, ssbuffer.block_id = 0 : i32} : memref<16x16xf32>
    %4 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<16x16xf32>
    %5 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%2, %3, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_2 = memref.alloc() {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} ins(%5 : tensor<16x16xf32>) outs(%alloc_2 : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}
