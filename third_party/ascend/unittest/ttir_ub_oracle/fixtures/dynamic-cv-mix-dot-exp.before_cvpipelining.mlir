// -----// IR Dump Before CVPipelining (cv-pipelining) //----- //
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 0 : i32}, {tt.tensor_kind = 1 : i32}, {}, {}, {}, {}, {}, {}], function_type = (memref<?xi8>, memref<?xi8>, memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32, i32, i32, i32, i32) -> (), sym_name = "dynamic_cv_mix_dot_exp"}> ({
  ^bb0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32):
    %0 = "arith.constant"() <{value = 0.000000e+00 : f32}> {ssbuffer.block_id = 1 : i32} : () -> f32
    "scope.scope"() ({
      "hivm.hir.sync_block_wait"() <{pipe = #hivm.pipe<PIPE_V>, static_flag_id = 1 : i64, tcore_type = #hivm.tcore_type<VECTOR>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : () -> ()
      %13 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : () -> memref<16x16xf32, #hivm.address_space<ub>>
      "annotation.mark"(%13) {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<16x16xf32, #hivm.address_space<ub>>) -> ()
      %14 = "memref.memory_space_cast"(%13) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<16x16xf32, #hivm.address_space<ub>>) -> memref<16x16xf32>
      %15 = "bufferization.to_tensor"(%14) <{restrict, writable}> {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<16x16xf32>) -> tensor<16x16xf32>
      %16 = "tensor.empty"() {ssbuffer.block_id = 1 : i32} : () -> tensor<16x16xf32>
      %17 = "linalg.fill"(%0, %16) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        "linalg.yield"(%arg16) : (f32) -> ()
      }) {ssbuffer.block_id = 1 : i32} : (f32, tensor<16x16xf32>) -> tensor<16x16xf32>
      %18 = "arith.addf"(%15, %17) <{fastmath = #arith.fastmath<none>}> {ssbuffer.block_id = 1 : i32} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
      %19 = "memref.reinterpret_cast"(%arg4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 16, 1>}> {ssbuffer.block_id = 1 : i32} : (memref<?xf32>) -> memref<16x16xf32, strided<[16, 1]>>
      %20 = "math.exp"(%18) <{fastmath = #arith.fastmath<none>}> {ssbuffer.block_id = 1 : i32} : (tensor<16x16xf32>) -> tensor<16x16xf32>
      "bufferization.materialize_in_destination"(%20, %19) <{writable}> {ssbuffer.block_id = 1 : i32} : (tensor<16x16xf32>, memref<16x16xf32, strided<[16, 1]>>) -> ()
      "scope.return"() : () -> ()
    }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : () -> ()
    "scope.scope"() ({
      %1 = "memref.reinterpret_cast"(%arg2) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 16, 1>}> {ssbuffer.block_id = 0 : i32} : (memref<?xf32>) -> memref<16x16xf32, strided<[16, 1]>>
      %2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.block_id = 0 : i32} : () -> memref<16x16xf32>
      "memref.copy"(%1, %2) {ssbuffer.block_id = 0 : i32} : (memref<16x16xf32, strided<[16, 1]>>, memref<16x16xf32>) -> ()
      %3 = "bufferization.to_tensor"(%2) <{restrict, writable}> {gm_load_bufferable, ssbuffer.block_id = 0 : i32} : (memref<16x16xf32>) -> tensor<16x16xf32>
      %4 = "memref.reinterpret_cast"(%arg3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 16>, static_strides = array<i64: 16, 1>}> {ssbuffer.block_id = 0 : i32} : (memref<?xf32>) -> memref<16x16xf32, strided<[16, 1]>>
      %5 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.block_id = 0 : i32} : () -> memref<16x16xf32>
      "memref.copy"(%4, %5) {ssbuffer.block_id = 0 : i32} : (memref<16x16xf32, strided<[16, 1]>>, memref<16x16xf32>) -> ()
      %6 = "bufferization.to_tensor"(%5) <{restrict, writable}> {gm_load_bufferable, ssbuffer.block_id = 0 : i32} : (memref<16x16xf32>) -> tensor<16x16xf32>
      %7 = "tensor.empty"() {ssbuffer.block_id = 0 : i32} : () -> tensor<16x16xf32>
      %8 = "linalg.fill"(%0, %7) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        "linalg.yield"(%arg14) : (f32) -> ()
      }) {ssbuffer.block_id = 0 : i32} : (f32, tensor<16x16xf32>) -> tensor<16x16xf32>
      %9 = "linalg.matmul"(%3, %6, %8) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg11: f32, %arg12: f32, %arg13: f32):
        %11 = "arith.mulf"(%arg11, %arg12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %12 = "arith.addf"(%arg13, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "linalg.yield"(%12) : (f32) -> ()
      }) {input_precision = "ieee", ssbuffer.block_id = 0 : i32} : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
      %10 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} : () -> memref<16x16xf32, #hivm.address_space<ub>>
      "annotation.mark"(%10) {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<16x16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.fixpipe"(%9, %10) <{dma_mode = #hivm.dma_mode<nz2nd>}> {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} : (tensor<16x16xf32>, memref<16x16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.sync_block_set"() <{operandSegmentSizes = array<i32: 0, 0>, pipe = #hivm.pipe<PIPE_V>, static_flag_id = 1 : i64, tcore_type = #hivm.tcore_type<CUBE>, tpipe = #hivm.pipe<PIPE_FIX>}> {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 0 : i32} : () -> ()
      "scope.return"() : () -> ()
    }) {hivm.tcore_type = #hivm.tcore_type<CUBE>} : () -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} : () -> ()
}) : () -> ()
