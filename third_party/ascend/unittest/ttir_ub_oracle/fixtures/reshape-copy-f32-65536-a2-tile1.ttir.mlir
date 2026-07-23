module {
  tt.func public @reshape_copy_round_trip(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %view = tt.reshape %value : tensor<65536xf32> -> tensor<256x256xf32>
    %flat = tt.reshape %view : tensor<256x256xf32> -> tensor<65536xf32>
    tt.store %dst_ptrs, %flat : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
