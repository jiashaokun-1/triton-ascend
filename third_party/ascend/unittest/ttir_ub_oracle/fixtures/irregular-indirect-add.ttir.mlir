"builtin.module"() ({
  "tt.func"() <{function_type = (!tt.ptr<f32>, !tt.ptr<i64>, !tt.ptr<f32>, f32) -> (), sym_name = "irregular_indirect_add", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>, %arg3: f32):
    %0 = "tt.make_range"() <{end = 8 : i32, start = 0 : i32}> : () -> tensor<8xi32>
    %1 = "tt.splat"(%arg1) : (!tt.ptr<i64>) -> tensor<8x!tt.ptr<i64>>
    %2 = "tt.addptr"(%1, %0) : (tensor<8x!tt.ptr<i64>>, tensor<8xi32>) -> tensor<8x!tt.ptr<i64>>
    %3 = "tt.load"(%2) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<8x!tt.ptr<i64>>) -> tensor<8xi64>
    %4 = "tt.splat"(%arg0) : (!tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>>
    %5 = "tt.addptr"(%4, %3) : (tensor<8x!tt.ptr<f32>>, tensor<8xi64>) -> tensor<8x!tt.ptr<f32>>
    %6 = "tt.load"(%5) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<8x!tt.ptr<f32>>) -> tensor<8xf32>
    %7 = "tt.splat"(%arg2) : (!tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>>
    %8 = "tt.addptr"(%7, %0) : (tensor<8x!tt.ptr<f32>>, tensor<8xi32>) -> tensor<8x!tt.ptr<f32>>
    %9 = "tt.splat"(%arg3) : (f32) -> tensor<8xf32>
    %10 = "arith.addf"(%6, %9) <{fastmath = #arith.fastmath<none>}> : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    "tt.store"(%8, %10) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) -> ()
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) : () -> ()
