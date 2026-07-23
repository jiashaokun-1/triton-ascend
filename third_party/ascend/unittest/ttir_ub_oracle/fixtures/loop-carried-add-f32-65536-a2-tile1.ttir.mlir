module {
  tt.func public @loop_carried_add(%init_src: !tt.ptr<f32>, %step_src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %init_splat = tt.splat %init_src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %init_ptrs = tt.addptr %init_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %step_splat = tt.splat %step_src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %step_ptrs = tt.addptr %step_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %init = tt.load %init_ptrs : tensor<65536x!tt.ptr<f32>>
    %result = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<65536xf32> {
      %step = tt.load %step_ptrs : tensor<65536x!tt.ptr<f32>>
      %next = arith.addf %acc, %step : tensor<65536xf32>
      scf.yield %next : tensor<65536xf32>
    }
    tt.store %dst_ptrs, %result : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
