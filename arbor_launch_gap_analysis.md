# Arbor launch-bound analysis

GPU: NVIDIA GeForce RTX 4090

## Phase 1: gap histogram

- total_wall_ms: 8177.159
- gpu_busy_ms: 6965.934
- gpu_idle_ms: 1211.225
- gpu_idle_ratio: 14.8%

| gap > threshold_us | count |
|---|---:|
| 10 | 6848 |
| 50 | 1295 |
| 100 | 227 |
| 500 | 26 |
| 1000 | 18 |

## Phase 1: top gaps

| gap_us | before | after | suspected cause |
|---|---|---|---|
| 227421.46 | void <unnamed>::softmax_warp_backward<c10::BFloat16, c10::BFloat16, float, (int)9, (bool)1, (bool)0>(T2 *, const T1 *, const T1 *, int, int, int, const bool *) | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<c10::BFloat16, (int)2, (int)1, (int)1>, at::native::Copy<c10::BFloat16, c10::BFloat16>>(T1, T2, T3...) | L. unknown |
| 224886.67 | void <unnamed>::softmax_warp_backward<c10::BFloat16, c10::BFloat16, float, (int)9, (bool)1, (bool)0>(T2 *, const T1 *, const T1 *, int, int, int, const bool *) | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<c10::BFloat16, (int)2, (int)1, (int)1>, at::native::Copy<c10::BFloat16, c10::BFloat16>>(T1, T2, T3...) | L. unknown |
| 89171.25 | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x64_64x3_nt_align2>(T1::Params) | L. unknown |
| 88085.07 | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x64_64x3_nt_align2>(T1::Params) | L. unknown |
| 52452.36 | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<c10::BFloat16, (int)2, (int)1, (int)1>, at::native::Copy<c10::BFloat16, c10::BFloat16>>(T1, T2, T3...) | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | L. unknown |
| 50141.59 | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<c10::BFloat16, (int)2, (int)1, (int)1>, at::native::Copy<c10::BFloat16, c10::BFloat16>>(T1, T2, T3...) | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | L. unknown |
| 42472.44 | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | triton_poi_fused_embedding_0 | L. unknown |
| 41698.69 | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | triton_poi_fused_embedding_0 | L. unknown |
| 7858.14 | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<long, (int)2, (int)1, (int)1>, at::native::Copy<long, long>>(T1, T2, T3...) | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | L. unknown |
| 7395.71 | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<long, (int)2, (int)1, (int)1>, at::native::Copy<long, long>>(T1, T2, T3...) | void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | L. unknown |
| 5331.18 | [CUDA memcpy] | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<long, (int)2, (int)1, (int)1>, at::native::Copy<long, long>>(T1, T2, T3...) | D. H2D / data loader wait |
| 4113.05 | [CUDA memcpy] | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<long, (int)2, (int)1, (int)1>, at::native::Copy<long, long>>(T1, T2, T3...) | D. H2D / data loader wait |
| 3989.92 | triton_poi_fused_full_33 | void <unnamed>::softmax_warp_forward<c10::BFloat16, c10::BFloat16, float, (int)9, (bool)1, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, bool) | L. unknown |
| 3964.30 | triton_poi_fused_full_33 | void <unnamed>::softmax_warp_forward<c10::BFloat16, c10::BFloat16, float, (int)9, (bool)1, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, bool) | L. unknown |
| 3330.88 | triton_poi_fused_embedding_dense_backward_43 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3) | L. unknown |
| 3152.89 | triton_poi_fused_embedding_dense_backward_43 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3) | L. unknown |
| 3045.98 | [CUDA memcpy] | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<long, (int)2, (int)1, (int)1>, at::native::Copy<long, long>>(T1, T2, T3...) | D. H2D / data loader wait |
| 2172.93 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3) | void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(int)2>, at::native::<unnamed>::UnaryOpFunctor<long, (int)2, (int)1, (int)1>, at::native::Copy<long, long>>(T1, T2, T3...) | L. unknown |
| 984.59 | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) | void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3) | L. unknown |
| 975.29 | void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) | void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<char *, (unsigned long)1>>(int, T2, T3) | L. unknown |

### cause buckets

- L. unknown: 17
- D. H2D / data loader wait: 3

## Phase 2: launch count ranking

| kernel/op | calls/step | median_us | total_ms/step | graph | fusion candidate |
|---|---:|---:|---:|---|---|
| [CUDA memset] | 641.0 | 0.362 | 0.335 | unknown | - |
| _a8_dequant_fp8_cast_transpose_kernel | 2944.0 | 3.256 | 31.166 | unknown | A8 quant+packed GEMM (high effort) |
| _a8_quantize_rows_kernel | 2944.0 | 3.388 | 29.637 | unknown | A8 quant+packed GEMM (high effort) |
| _a8_quantize_rows_scaled_kernel | 2944.0 | 5.460 | 64.197 | unknown | A8 quant+packed GEMM (high effort) |
| _fp8_cast_transpose_kernel | 2944.0 | 3.421 | 82.388 | unknown | - |
| _packed_bitlinear_kernel | 5888.0 | 225.779 | 1762.641 | unknown | fused A8 quant+packed GEMM (high effort) |
| sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warpsize2x2x1_tensor16x8x32_algo2_execute_kernel__5x_cublas | 640.0 | 152.723 | 105.331 | unknown | fused A8 quant+packed GEMM (high effort) |
| sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_algo2_execute_kernel__5x_cublas | 192.0 | 318.703 | 54.505 | unknown | fused A8 quant+packed GEMM (high effort) |
| sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warpsize2x2x1_tensor16x8x32_algo2_execute_kernel__5x_cublas | 1920.0 | 47.169 | 111.028 | unknown | fused A8 quant+packed GEMM (high effort) |
| sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage6_warpsize2x2x1_tensor16x8x32_algo2_execute_kernel__5x_cublas | 96.0 | 182.096 | 19.635 | unknown | fused A8 quant+packed GEMM (high effort) |
| triton_per_fused__fused_rms_norm__fused_rms_norm_backward_mul_view_21 | 640.0 | 1.973 | 1.374 | unknown | RMSNorm+A8 quant |
| triton_per_fused__fused_rms_norm__fused_rms_norm_backward_mul_view_4 | 96.0 | 327.401 | 33.763 | unknown | RMSNorm+A8 quant |
| triton_per_fused__fused_rms_norm__fused_rms_norm_backward_view_17 | 1952.0 | 1.974 | 3.959 | unknown | RMSNorm+A8 quant |
| triton_per_fused__to_copy_abs_add_amax_cat_clamp_min_div_mul_threshold_backward_view_24 | 640.0 | 1.085 | 0.692 | unknown | - |
| triton_per_fused__to_copy_abs_add_amax_cat_clamp_min_div_mul_threshold_backward_view_8 | 864.0 | 1.086 | 1.159 | unknown | - |
| triton_per_fused__to_copy_abs_amax_clamp_min_div_view_18 | 1280.0 | 1.119 | 1.539 | unknown | - |
| triton_per_fused__to_copy_amax_clamp_min_div_maximum_mul_neg_30 | 2560.0 | 1.283 | 3.279 | unknown | - |
| triton_poi_fused__unsafe_view_add_cat_clone_mul_neg_select_slice_backward_transpose_view_11 | 96.0 | 217.458 | 21.798 | unknown | - |
| triton_poi_fused__unsafe_view_add_cat_clone_mul_neg_select_slice_backward_transpose_view_30 | 640.0 | 17.960 | 12.852 | unknown | - |
| triton_poi_fused_add_cat_mul_threshold_backward_22 | 640.0 | 32.465 | 22.567 | unknown | - |
| triton_poi_fused_add_cat_mul_threshold_backward_6 | 96.0 | 392.351 | 41.474 | unknown | - |
| triton_poi_fused_add_mul_slice_split_with_sizes_stack_sub_transpose_view_14 | 640.0 | 3.782 | 2.417 | unknown | - |
| triton_poi_fused_add_mul_slice_split_with_sizes_stack_sub_transpose_view_15 | 640.0 | 1.973 | 1.271 | unknown | - |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward__unsafe_view_clone_flex_attention_backward_transpose_view_28 | 640.0 | 5.099 | 3.482 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward__unsafe_view_clone_transpose_view_26 | 640.0 | 10.295 | 6.615 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward__unsafe_view_clone_transpose_view_27 | 640.0 | 4.902 | 3.164 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward_abs_add_amax_view_25 | 1248.0 | 21.792 | 23.879 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward_mul_view_19 | 640.0 | 51.017 | 37.483 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward_mul_view_20 | 640.0 | 27.301 | 17.762 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__fused_rms_norm_backward_view_16 | 1280.0 | 4.901 | 7.081 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm__unsafe_view_clone_transpose_view_17 | 640.0 | 4.671 | 3.022 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm_add_view_20 | 1216.0 | 6.513 | 9.704 | unknown | RMSNorm+A8 quant |
| triton_red_fused__fused_rms_norm_mul_relu_split_with_sizes_view_5 | 96.0 | 331.925 | 34.650 | unknown | RMSNorm+A8 quant |
| triton_red_fused_abs_add_amax_cat_mul_threshold_backward_view_7 | 96.0 | 215.055 | 23.847 | unknown | - |
| triton_red_fused_abs_amax_view_12 | 96.0 | 112.100 | 12.807 | unknown | - |
| triton_red_fused_amax_amin_31 | 1920.0 | 4.638 | 10.409 | unknown | - |
| triton_tem_fused__fused_rms_norm__fused_rms_norm_backward__unsafe_view_clone_flex_attention_backward_transpose_view_29 | 640.0 | 201.998 | 136.350 | unknown | RMSNorm+A8 quant |
| triton_tem_fused_add_flex_attention_mul_slice_split_with_sizes_stack_sub_transpose_view_16 | 640.0 | 34.143 | 23.443 | unknown | - |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3) | 813.0 | 5.675 | 21.161 | unknown | - |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(int, T2, T3) | 8580.0 | 9.441 | 247.693 | unknown | - |
| void pytorch_flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<(int)64, (int)64, (int)128, (int)8, (int)2, (int)4, (int)4, (bool)1, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)64, (int)64, (int)128, (int)8, cutlass::bfloat16_t>>>(pytorch_flash::Flash_bwd_params, int) | 96.0 | 355.641 | 39.991 | unknown | - |
| void pytorch_flash::flash_bwd_dot_do_o_kernel<(bool)1, Flash_bwd_kernel_traits<(int)64, (int)64, (int)128, (int)8, (int)2, (int)4, (int)4, (bool)1, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)64, (int)64, (int)128, (int)8, cutlass::bfloat16_t>>>(pytorch_flash::Flash_bwd_params) | 96.0 | 370.787 | 37.906 | unknown | fused A8 quant+packed GEMM (high effort) |
| void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(int)64, (int)64, (int)128, (int)8, (int)2, (int)4, (int)4, (bool)1, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)64, (int)64, (int)128, (int)8, cutlass::bfloat16_t>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0>(pytorch_flash::Flash_bwd_params) | 32.0 | 782.990 | 27.571 | unknown | - |
| void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(int)64, (int)64, (int)128, (int)8, (int)2, (int)4, (int)4, (bool)1, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)64, (int)64, (int)128, (int)8, cutlass::bfloat16_t>>, (bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0>(pytorch_flash::Flash_bwd_params) | 64.0 | 795.736 | 53.084 | unknown | - |
| void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)64, (int)128, (int)128, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)64, (int)128, (int)128, (int)4, cutlass::bfloat16_t>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)0>(pytorch_flash::Flash_fwd_params) | 32.0 | 377.368 | 12.451 | unknown | - |
| void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)64, (int)128, (int)128, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)64, (int)128, (int)128, (int)4, cutlass::bfloat16_t>>, (bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)0>(pytorch_flash::Flash_fwd_params) | 64.0 | 392.480 | 26.424 | unknown | - |

