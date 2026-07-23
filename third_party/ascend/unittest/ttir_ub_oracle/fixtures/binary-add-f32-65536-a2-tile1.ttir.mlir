
module {
  tt.func public @add(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %lhs_value = tt.load %lhs_ptrs : tensor<65536x!tt.ptr<f32>>
    %rhs_value = tt.load %rhs_ptrs : tensor<65536x!tt.ptr<f32>>
    %sum = arith.addf %lhs_value, %rhs_value : tensor<65536xf32>
    tt.store %dst_ptrs, %sum : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
