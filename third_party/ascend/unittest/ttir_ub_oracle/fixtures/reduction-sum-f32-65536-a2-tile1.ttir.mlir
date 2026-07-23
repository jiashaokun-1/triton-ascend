module {
  tt.func public @reduction_sum(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %sum = "tt.reduce" (%value) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<65536xf32>) -> f32
    tt.store %dst, %sum : !tt.ptr<f32>
    tt.return
  }
}
