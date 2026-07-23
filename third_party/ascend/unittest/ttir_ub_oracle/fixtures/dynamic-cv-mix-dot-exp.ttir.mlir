"builtin.module"() ({
  "tt.func"() <{function_type = (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>) -> (), sym_name = "dynamic_cv_mix_dot_exp", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>):
    %0 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<16x16xf32>}> : () -> tensor<16x16xf32>
    %1 = "arith.constant"() <{value = dense<16> : tensor<16x1xi32>}> : () -> tensor<16x1xi32>
    %2 = "tt.make_range"() <{end = 16 : i32, start = 0 : i32}> : () -> tensor<16xi32>
    %3 = "tt.expand_dims"(%2) <{axis = 1 : i32}> : (tensor<16xi32>) -> tensor<16x1xi32>
    %4 = "tt.expand_dims"(%2) <{axis = 0 : i32}> : (tensor<16xi32>) -> tensor<1x16xi32>
    %5 = "arith.muli"(%3, %1) <{overflowFlags = #arith.overflow<none>}> : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x1xi32>
    %6 = "tt.splat"(%arg0) : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>>
    %7 = "tt.addptr"(%6, %5) : (tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>) -> tensor<16x1x!tt.ptr<f32>>
    %8 = "tt.broadcast"(%7) : (tensor<16x1x!tt.ptr<f32>>) -> tensor<16x16x!tt.ptr<f32>>
    %9 = "tt.broadcast"(%4) : (tensor<1x16xi32>) -> tensor<16x16xi32>
    %10 = "tt.addptr"(%8, %9) : (tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>) -> tensor<16x16x!tt.ptr<f32>>
    %11 = "tt.load"(%10) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<16x16x!tt.ptr<f32>>) -> tensor<16x16xf32>
    %12 = "tt.splat"(%arg1) : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>>
    %13 = "tt.addptr"(%12, %5) : (tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>) -> tensor<16x1x!tt.ptr<f32>>
    %14 = "tt.broadcast"(%13) : (tensor<16x1x!tt.ptr<f32>>) -> tensor<16x16x!tt.ptr<f32>>
    %15 = "tt.addptr"(%14, %9) : (tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>) -> tensor<16x16x!tt.ptr<f32>>
    %16 = "tt.load"(%15) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<16x16x!tt.ptr<f32>>) -> tensor<16x16xf32>
    %17 = "tt.dot"(%11, %16, %0) <{inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %18 = "tt.splat"(%arg2) : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>>
    %19 = "tt.addptr"(%18, %5) : (tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>) -> tensor<16x1x!tt.ptr<f32>>
    %20 = "tt.broadcast"(%19) : (tensor<16x1x!tt.ptr<f32>>) -> tensor<16x16x!tt.ptr<f32>>
    %21 = "tt.addptr"(%20, %9) : (tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>) -> tensor<16x16x!tt.ptr<f32>>
    %22 = "math.exp"(%17) <{fastmath = #arith.fastmath<none>}> : (tensor<16x16xf32>) -> tensor<16x16xf32>
    "tt.store"(%21, %22) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<16x16x!tt.ptr<f32>>, tensor<16x16xf32>) -> ()
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) : () -> ()
