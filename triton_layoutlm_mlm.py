# Triton kernels for LayoutLM_MLM
# Model: microsoft/layoutlm-base-uncased

triton_per_fused_add_embedding_native_layer_norm_sub_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr11 + (r0_1), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr12 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 10000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 10000")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 64*tmp4), xmask, other=0.0)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 128, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 128")
    tmp13 = tl.load(in_ptr3 + (r0_1 + 64*tmp11), xmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK, R0_BLOCK], 1024, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 1024")
    tmp21 = tl.load(in_ptr5 + (r0_1 + 64*tmp19), xmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp24 = tmp23 + tmp16
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert(((0 <= tmp26) & (tmp26 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp26 < 1024")
    tmp28 = tl.load(in_ptr6 + (r0_1 + 64*tmp26), xmask, other=0.0)
    tmp29 = tmp22 + tmp28
    tmp31 = tmp30 + tmp16
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 1024")
    tmp35 = tl.load(in_ptr5 + (r0_1 + 64*tmp33), xmask, other=0.0)
    tmp36 = tmp29 + tmp35
    tmp38 = tmp37 + tmp16
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tl.device_assert(((0 <= tmp40) & (tmp40 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp40 < 1024")
    tmp42 = tl.load(in_ptr6 + (r0_1 + 64*tmp40), xmask, other=0.0)
    tmp43 = tmp36 + tmp42
    tmp44 = tmp37 - tmp23
    tmp45 = tmp44 + tmp16
    tmp46 = tmp44 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp44)
    tl.device_assert(((0 <= tmp47) & (tmp47 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp47 < 1024")
    tmp49 = tl.load(in_ptr7 + (r0_1 + 64*tmp47), xmask, other=0.0)
    tmp50 = tmp43 + tmp49
    tmp51 = tmp30 - tmp15
    tmp52 = tmp51 + tmp16
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tl.device_assert(((0 <= tmp54) & (tmp54 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp54 < 1024")
    tmp56 = tl.load(in_ptr8 + (r0_1 + 64*tmp54), xmask, other=0.0)
    tmp57 = tmp50 + tmp56
    tmp59 = tl.full([XBLOCK, R0_BLOCK], 2, tl.int32)
    tmp60 = tmp58 + tmp59
    tmp61 = tmp58 < 0
    tmp62 = tl.where(tmp61, tmp60, tmp58)
    tl.device_assert(((0 <= tmp62) & (tmp62 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp62 < 2")
    tmp64 = tl.load(in_ptr10 + (r0_1 + 64*tmp62), xmask, other=0.0)
    tmp65 = tmp57 + tmp64
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
    tmp68 = tl.where(xmask, tmp66, 0)
    tmp69 = tl.broadcast_to(tmp66, [XBLOCK, R0_BLOCK])
    tmp71 = tl.where(xmask, tmp69, 0)
    tmp72 = tl.sum(tmp71, 1)[:, None]
    tmp73 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp74 = tmp73.to(tl.float32)
    tmp75 = (tmp72 / tmp74)
    tmp76 = tmp66 - tmp75
    tmp77 = tmp76 * tmp76
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, R0_BLOCK])
    tmp80 = tl.where(xmask, tmp78, 0)
    tmp81 = tl.sum(tmp80, 1)[:, None]
    tmp82 = tmp65 - tmp75
    tmp83 = 64.0
    tmp84 = (tmp81 / tmp83)
    tmp85 = 1e-12
    tmp86 = tmp84 + tmp85
    tmp87 = libdevice.rsqrt(tmp86)
    tmp88 = tmp82 * tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp90 + tmp91
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp92, xmask)
''', device_str='cuda')

triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1 = async_compile.triton('triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 24832}}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tmp7 = -3.4028234663852886e+38
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, float("-inf"))
    tmp15 = triton_helpers.max2(tmp14, 1)[:, None]
    tmp16 = tmp10 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp9 - tmp15
    tmp23 = tl_math.exp(tmp22)
    tmp24 = (tmp23 / tmp21)
    tl.store(in_out_ptr0 + (r0_1 + 32*x0), tmp24, xmask)
''', device_str='cuda')

triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 2)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')

triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_per_fused_add_native_layer_norm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 33536}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp31, xmask)
''', device_str='cuda')

triton_poi_fused_gelu_4 = async_compile.triton('triton_poi_fused_gelu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')

triton_poi_fused_addmm_tanh_5 = async_compile.triton('triton_poi_fused_addmm_tanh_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_tanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_tanh_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')

@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr11 + (r0_1), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr12 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 10000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 10000")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 64*tmp4), xmask, other=0.0)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 128, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 128")
    tmp13 = tl.load(in_ptr3 + (r0_1 + 64*tmp11), xmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK, R0_BLOCK], 1024, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 1024")
    tmp21 = tl.load(in_ptr5 + (r0_1 + 64*tmp19), xmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp24 = tmp23 + tmp16
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert(((0 <= tmp26) & (tmp26 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp26 < 1024")
    tmp28 = tl.load(in_ptr6 + (r0_1 + 64*tmp26), xmask, other=0.0)
    tmp29 = tmp22 + tmp28
    tmp31 = tmp30 + tmp16
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp33 < 1024")
    tmp35 = tl.load(in_ptr5 + (r0_1 + 64*tmp33), xmask, other=0.0)
    tmp36 = tmp29 + tmp35
    tmp38 = tmp37 + tmp16
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tl.device_assert(((0 <= tmp40) & (tmp40 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp40 < 1024")
    tmp42 = tl.load(in_ptr6 + (r0_1 + 64*tmp40), xmask, other=0.0)
    tmp43 = tmp36 + tmp42
    tmp44 = tmp37 - tmp23
    tmp45 = tmp44 + tmp16
    tmp46 = tmp44 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp44)
    tl.device_assert(((0 <= tmp47) & (tmp47 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp47 < 1024")
    tmp49 = tl.load(in_ptr7 + (r0_1 + 64*tmp47), xmask, other=0.0)
    tmp50 = tmp43 + tmp49
    tmp51 = tmp30 - tmp15
    tmp52 = tmp51 + tmp16
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tl.device_assert(((0 <= tmp54) & (tmp54 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp54 < 1024")
    tmp56 = tl.load(in_ptr8 + (r0_1 + 64*tmp54), xmask, other=0.0)
    tmp57 = tmp50 + tmp56
    tmp59 = tl.full([XBLOCK, R0_BLOCK], 2, tl.int32)
    tmp60 = tmp58 + tmp59
    tmp61 = tmp58 < 0
    tmp62 = tl.where(tmp61, tmp60, tmp58)
    tl.device_assert(((0 <= tmp62) & (tmp62 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp62 < 2")
    tmp64 = tl.load(in_ptr10 + (r0_1 + 64*tmp62), xmask, other=0.0)
    tmp65 = tmp57 + tmp64
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, R0_BLOCK])
    tmp68 = tl.where(xmask, tmp66, 0)
    tmp69 = tl.broadcast_to(tmp66, [XBLOCK, R0_BLOCK])
    tmp71 = tl.where(xmask, tmp69, 0)
    tmp72 = tl.sum(tmp71, 1)[:, None]
    tmp73 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp74 = tmp73.to(tl.float32)
    tmp75 = (tmp72 / tmp74)
    tmp76 = tmp66 - tmp75
    tmp77 = tmp76 * tmp76
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, R0_BLOCK])
    tmp80 = tl.where(xmask, tmp78, 0)
    tmp81 = tl.sum(tmp80, 1)[:, None]
    tmp82 = tmp65 - tmp75
    tmp83 = 64.0
    tmp84 = (tmp81 / tmp83)
    tmp85 = 1e-12
    tmp86 = tmp84 + tmp85
    tmp87 = libdevice.rsqrt(tmp86)
    tmp88 = tmp82 * tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp90 + tmp91
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp92, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/46/c46ro5jl6j2363dbrn77t75tysshbirxvhouutdyotgsgcwnnggc.py
# Topologically Sorted Source Nodes: [attn_weights, extended_attention_mask_1, sub, extended_attention_mask_2, attn_weights_1, , exp, softmax], Original ATen: [aten.mul, aten._to_copy, aten.rsub, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_1
#   attn_weights => mul_3
#   attn_weights_1 => add_10
#   exp => exp_default_1
#   extended_attention_mask_1 => convert_element_type
#   extended_attention_mask_2 => mul
#   softmax => div
#   sub => sub, sub_tensor_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 0.1767766952966369), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.4028234663852886e+38), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_10, -1), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %getitem_12), kwargs = {})
#   %exp_default_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_13), kwargs = {})
triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1 = async_compile.triton('triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 24832}}
)


@triton.jit
def triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tmp7 = -3.4028234663852886e+38
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, float("-inf"))
    tmp15 = triton_helpers.max2(tmp14, 1)[:, None]
    tmp16 = tmp10 - tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp9 - tmp15
    tmp23 = tl_math.exp(tmp22)
    tmp24 = (tmp23 / tmp21)
    tl.store(in_out_ptr0 + (r0_1 + 32*x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/g4/cg4thbu5ovgsjk7iq5wiwwbufsyh3msco5an6g2e3tarxolmvluy.py
# Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_1 => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 2)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xy/cxyymdsmgdi7v7hynbxq3gxbyxuaqpr2jgva7heqyeenonoyp2ll.py
# Topologically Sorted Source Nodes: [add_9, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_9 => add_11
#   hidden_states_2 => add_12, add_13, mul_4, mul_5, rsqrt_1, sub_5, var_mean_1
# Graph fragment:
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %add_9), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_3), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg22_1), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg23_1), kwargs = {})
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_per_fused_add_native_layer_norm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 33536}}
)


@triton.jit
def triton_per_fused_add_native_layer_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 64.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/kp/ckpgqyg6lgu4v76wuffqkjuapupbpgzx3kicnmcrrgjqetnwqaqu.py
# Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_4 => add_14, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_14), kwargs = {})
triton_poi_fused_gelu_4 = async_compile.triton('triton_poi_fused_gelu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49664}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_gelu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/a6/ca6pgtvasujc3lnz4r2mpiwz6iuaio5r6v6l5426yqlbx5bim5kn.py
# Topologically Sorted Source Nodes: [add, pooled_output_1], Original ATen: [aten.addmm, aten.tanh]
# Source node to ATen node mapping:
#   add => add_tensor
#   pooled_output_1 => tanh
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg47_1), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_addmm_tanh_5 = async_compile.triton('triton_poi_fused_addmm_tanh_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_tanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_addmm_tanh_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 32), (32, 1))
    assert_size_stride(arg1_1, (1, 32), (32, 1))
    assert_size_stride(arg2_1, (1, 32), (32, 1))
    assert_size_stride(arg3_1, (1, 32, 4), (128, 4, 1))
    assert_size_stride(arg4_1, (10000, 64), (64, 1))
    assert_size_stride(arg5_1, (1, 128), (128, 1))
    assert_size_stride(arg6_1, (128, 64), (64, 1))
    assert_size_stride(arg7_1, (1024, 64), (64, 1))
    assert_size_stride(arg8_1, (1024, 64), (64, 1))
    assert_size_stride(arg9_1, (1024, 64), (64, 1))
    assert_size_stride(arg10_1, (1024, 64), (64, 1))
    assert_size_stride(arg11_1, (2, 64), (64, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, 64), (64, 1))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, 64), (64, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, 64), (64, 1))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, 64), (64, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (128, 64), (64, 1))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (64, 128), (128, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, 64), (64, 1))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (64, 64), (64, 1))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, 64), (64, 1))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, 64), (64, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (128, 64), (64, 1))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (64, 128), (128, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, 64), (64, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32, 64), (2048, 64, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, add, left_position_embeddings, add_1, upper_position_embeddings, add_2, right_position_embeddings, add_3, lower_position_embeddings, add_4, sub_1, h_position_embeddings, add_5, sub_2, w_position_embeddings, add_6, token_type_embeddings, embeddings, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.sub, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_sub_0.run(buf5, arg0_1, arg4_1, arg5_1, arg6_1, arg3_1, arg7_1, arg8_1, arg9_1, arg10_1, arg2_1, arg11_1, arg12_1, arg13_1, 32, 64, stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf6 = empty_strided_cuda((32, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf5, (32, 64), (64, 1), 0), reinterpret_tensor(arg14_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf6)
        del arg14_1
        del arg15_1
        buf7 = empty_strided_cuda((32, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, reinterpret_tensor(buf5, (32, 64), (64, 1), 0), reinterpret_tensor(arg16_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf7)
        del arg16_1
        del arg17_1
        buf8 = empty_strided_cuda((2, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (2, 32, 32), (32, 64, 1), 0), reinterpret_tensor(buf7, (2, 32, 32), (32, 1, 64), 0), out=buf8)
        buf12 = reinterpret_tensor(buf8, (1, 2, 32, 32), (2048, 1024, 32, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_weights, extended_attention_mask_1, sub, extended_attention_mask_2, attn_weights_1, , exp, softmax], Original ATen: [aten.mul, aten._to_copy, aten.rsub, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1.run(buf12, arg1_1, 64, 32, stream=stream0)
        buf11 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, reinterpret_tensor(buf5, (32, 64), (64, 1), 0), reinterpret_tensor(arg18_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf11)
        del arg18_1
        del arg19_1
        buf13 = reinterpret_tensor(buf6, (2, 32, 32), (1024, 32, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (2, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf11, (2, 32, 32), (32, 64, 1), 0), out=buf13)
        buf14 = reinterpret_tensor(buf12, (1, 32, 2, 32), (2048, 64, 32, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf13, buf14, 2048, stream=stream0)
        buf15 = reinterpret_tensor(buf13, (32, 64), (64, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf14, (32, 64), (64, 1), 0), reinterpret_tensor(arg20_1, (64, 64), (1, 64), 0), out=buf15)
        del arg20_1
        buf19 = reinterpret_tensor(buf15, (1, 32, 64), (2048, 64, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_2], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_3.run(buf19, arg21_1, buf5, arg22_1, arg23_1, 32, 64, stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        buf20 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf19, (32, 64), (64, 1), 0), reinterpret_tensor(arg24_1, (64, 128), (1, 64), 0), out=buf20)
        del arg24_1
        buf21 = reinterpret_tensor(buf20, (1, 32, 128), (4096, 128, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_4.run(buf21, arg25_1, 4096, stream=stream0)
        del arg25_1
        buf22 = reinterpret_tensor(buf5, (32, 64), (64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf21, (32, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 64), (1, 128), 0), out=buf22)
        del arg26_1
        buf26 = reinterpret_tensor(buf22, (1, 32, 64), (2048, 64, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [add_10, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_3.run(buf26, arg27_1, buf19, arg28_1, arg29_1, 32, 64, stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        buf27 = reinterpret_tensor(buf19, (32, 64), (64, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf26, (32, 64), (64, 1), 0), reinterpret_tensor(arg30_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf27)
        del arg30_1
        del arg31_1
        buf28 = reinterpret_tensor(buf14, (32, 64), (64, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg33_1, reinterpret_tensor(buf26, (32, 64), (64, 1), 0), reinterpret_tensor(arg32_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf28)
        del arg32_1
        del arg33_1
        buf29 = reinterpret_tensor(buf11, (2, 32, 32), (1024, 32, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (2, 32, 32), (32, 64, 1), 0), reinterpret_tensor(buf28, (2, 32, 32), (32, 1, 64), 0), out=buf29)
        buf33 = reinterpret_tensor(buf29, (1, 2, 32, 32), (2048, 1024, 32, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [extended_attention_mask_1, sub, extended_attention_mask_2, attn_weights_4, attn_weights_5, , exp, softmax_1], Original ATen: [aten._to_copy, aten.rsub, aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_rsub_sub_1.run(buf33, arg1_1, 64, 32, stream=stream0)
        del arg1_1
        buf32 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg35_1, reinterpret_tensor(buf26, (32, 64), (64, 1), 0), reinterpret_tensor(arg34_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf32)
        del arg34_1
        del arg35_1
        buf34 = reinterpret_tensor(buf27, (2, 32, 32), (1024, 32, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (2, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf32, (2, 32, 32), (32, 64, 1), 0), out=buf34)
        del buf32
        buf35 = reinterpret_tensor(buf33, (1, 32, 2, 32), (2048, 64, 32, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_output_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf34, buf35, 2048, stream=stream0)
        buf36 = reinterpret_tensor(buf34, (32, 64), (64, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf35, (32, 64), (64, 1), 0), reinterpret_tensor(arg36_1, (64, 64), (1, 64), 0), out=buf36)
        del arg36_1
        del buf35
        buf40 = reinterpret_tensor(buf36, (1, 32, 64), (2048, 64, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_3.run(buf40, arg37_1, buf26, arg38_1, arg39_1, 32, 64, stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        buf41 = reinterpret_tensor(buf21, (32, 128), (128, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf40, (32, 64), (64, 1), 0), reinterpret_tensor(arg40_1, (64, 128), (1, 64), 0), out=buf41)
        del arg40_1
        buf42 = reinterpret_tensor(buf41, (1, 32, 128), (4096, 128, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_4.run(buf42, arg41_1, 4096, stream=stream0)
        del arg41_1
        buf43 = reinterpret_tensor(buf26, (32, 64), (64, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf42, (32, 128), (128, 1), 0), reinterpret_tensor(arg42_1, (128, 64), (1, 128), 0), out=buf43)
        del arg42_1
        del buf42
        buf47 = reinterpret_tensor(buf43, (1, 32, 64), (2048, 64, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_3.run(buf47, arg43_1, buf40, arg44_1, arg45_1, 32, 64, stream=stream0)
        del arg43_1
        del arg44_1
        del arg45_1
        del buf40
        buf48 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1, 64), (64, 1), 0), reinterpret_tensor(arg46_1, (64, 64), (1, 64), 0), out=buf48)
        del arg46_1
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [add, pooled_output_1], Original ATen: [aten.addmm, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_tanh_5.run(buf49, arg47_1, 64, stream=stream0)
        del arg47_1
    return (buf47, buf49, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((1, 32, 4), (128, 4, 1), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((10000, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
