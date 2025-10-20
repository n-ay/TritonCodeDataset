# Triton kernels for XLNet
# Model: xlnet-base-cased

triton_poi_fused_embedding_0 = async_compile.triton('triton_poi_fused_embedding_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 10000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 10000")
    tmp6 = tl.load(in_ptr1 + (x0 + 64*tmp4), xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')

triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 82944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')

triton_per_fused__softmax_add_amax_index_select_mul_sub_3 = async_compile.triton('triton_per_fused__softmax_add_amax_index_select_mul_sub_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_amax_index_select_mul_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 32768}}
)
@triton.jit
def triton_per_fused__softmax_add_amax_index_select_mul_sub_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 32*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32 + r0_2 + 63*x0 + 2048*x1 + 2048*((r0_2 + 63*x0) // 2016)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = 0.125
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = (tmp14 / tmp18)
    tl.store(in_out_ptr0 + (r0_2 + 32*x3), tmp19, xmask)
''', device_str='cuda')

triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 2048*x0), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')

triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 32768, 'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 64)
    x2 = xindex // 64
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 128*x1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + 128*y0), tmp0, xmask & ymask)
''', device_str='cuda')

triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 33280}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = (tmp18 / tmp20)
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp29, xmask)
''', device_str='cuda')

triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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

triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 33536}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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

@triton.jit
def triton_poi_fused_embedding_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 10000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 10000")
    tmp6 = tl.load(in_ptr1 + (x0 + 64*tmp4), xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/fk/cfkqgv7kozfokm2nv3psq4pr33geut26abb2vsagjorv65astlqc.py
# Topologically Sorted Source Nodes: [add, add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %arg6_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %arg7_1), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 82944}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


cpp_fused_cos_mul_sin_2 = async_compile.cpp_pybinding(['float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(32L)))
                    {
                        auto tmp0 = 2L*x1;
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = at::vec::Vectorized<float>::arange(tmp1, 2);
                        auto tmp3 = static_cast<float>(0.015625);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 * tmp4;
                        auto tmp6 = static_cast<float>(10000.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp7.pow(tmp5);
                        auto tmp9 = tmp8.reciprocal();
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = 32L + ((-1L)*x0);
                        auto tmp14 = c10::convert<float>(tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp15 * tmp12;
                        auto tmp17 = tmp16.sin();
                        tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(32L)))
                    {
                        auto tmp0 = 2L*x1;
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = at::vec::Vectorized<float>::arange(tmp1, 2);
                        auto tmp3 = static_cast<float>(0.015625);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 * tmp4;
                        auto tmp6 = static_cast<float>(10000.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp7.pow(tmp5);
                        auto tmp9 = tmp8.reciprocal();
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = 32L + ((-1L)*x0);
                        auto tmp14 = c10::convert<float>(tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp15 * tmp12;
                        auto tmp17 = tmp16.cos();
                        tmp17.store(out_ptr1 + static_cast<int64_t>(x1 + 64L*x0));
                    }
                }
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_root/47/c47pqgbr7jpfxe7evoucdw2eid5swp3bqd54lor4ty5qpoqmnb7s.py
# Topologically Sorted Source Nodes: [x_3, add_2, add_3, , mul_1, attn_prob], Original ATen: [aten.index_select, aten.add, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#    => amax_default_1, mul_tensor_2, sub_tensor_1
#   add_2 => add_2
#   add_3 => add_3
#   attn_prob => div_1, exp, sum_1
#   mul_1 => mul_tensor_3
#   x_3 => index
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_25, [None, None, None, %iota_2]), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %index), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0), kwargs = {})
#   %mul_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_2, [3], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_1), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_1, 0.125), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_3,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_add_amax_index_select_mul_sub_3 = async_compile.triton('triton_per_fused__softmax_add_amax_index_select_mul_sub_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_amax_index_select_mul_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 32768}}
)


@triton.jit
def triton_per_fused__softmax_add_amax_index_select_mul_sub_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 32*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32 + r0_2 + 63*x0 + 2048*x1 + 2048*((r0_2 + 63*x0) // 2016)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = 0.125
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = (tmp14 / tmp18)
    tl.store(in_out_ptr0 + (r0_2 + 32*x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jm/cjmuscpbd6qqjm3jfyiiuxryz6h2krjhozqczuxc5gpf7wz4565a.py
# Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_out => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_40,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 2048*x0), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/zp/czpvw75jlxphqxfktvzr3xiks6tuqrj6tjlsix42y3ee4puy5heh.py
# Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_out => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 32768, 'x': 65536}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 64)
    x2 = xindex // 64
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 128*x1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + 128*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/uh/cuhzzzsoshehrtgabkgnwh5ujfr4hcvbqt5mmu2gefe2venwc2av.py
# Topologically Sorted Source Nodes: [attn_out_2, output], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_out_2 => add_4
#   output => add_5, add_6, mul_3, mul_4, rsqrt, sub_1, var_mean
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_33, %embedding), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg9_1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg10_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 33280}}
)


@triton.jit
def triton_per_fused_add_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp26 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = (tmp18 / tmp20)
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ot/cotizdrpfenrayc7ly5qyesphhw7hxiphctdvz5jh3mdz5fjzkjj.py
# Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   output_2 => add_7, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_7), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49664}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_gelu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_root/e4/ce4ufugididf3llniry2a5hplsnf7llmfbsnup4t5ghuvw4iav5y.py
# Topologically Sorted Source Nodes: [add_5, output_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_5 => add_8
#   output_6 => add_10, add_9, mul_8, mul_9, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_37, %add_6), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_3), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg15_1), kwargs = {})
#   %add_10 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg16_1), kwargs = {})
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 33536}}
)


@triton.jit
def triton_per_fused_add_native_layer_norm_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 32), (32, 1))
    assert_size_stride(arg1_1, (10000, 64), (64, 1))
    assert_size_stride(arg2_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg3_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg4_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg5_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg6_1, (2, 64), (64, 1))
    assert_size_stride(arg7_1, (2, 64), (64, 1))
    assert_size_stride(arg8_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (128, 64), (64, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (64, 128), (128, 1))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg18_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg19_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg20_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg21_1, (2, 64), (64, 1))
    assert_size_stride(arg22_1, (2, 64), (64, 1))
    assert_size_stride(arg23_1, (64, 2, 64), (128, 64, 1))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (128, 64), (64, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (64, 128), (128, 1))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [word_emb_k], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_0.run(arg0_1, arg1_1, buf0, 2048, stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (32, 64), (64, 1), 0), reinterpret_tensor(arg2_1, (64, 128), (128, 1), 0), out=buf1)
        del arg2_1
        buf2 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (32, 64), (64, 1), 0), reinterpret_tensor(arg3_1, (64, 128), (128, 1), 0), out=buf2)
        del arg3_1
        buf3 = empty_strided_cuda((32, 1, 2, 64), (128, 1, 64, 1), torch.float32)
        buf10 = empty_strided_cuda((32, 1, 2, 64), (128, 1, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, add_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf1, arg6_1, arg7_1, buf3, buf10, 4096, stream=stream0)
        del arg6_1
        del arg7_1
        buf4 = empty_strided_cuda((2, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ac], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (2, 32, 64), (64, 128, 1), 0), reinterpret_tensor(buf2, (2, 64, 32), (64, 1, 128), 0), out=buf4)
    buf7 = empty_strided_cpu((64, 64), (64, 1), torch.float32)
    buf5 = reinterpret_tensor(buf7, (64, 32), (64, 1), 0)  # alias
    buf6 = reinterpret_tensor(buf7, (64, 32), (64, 1), 32)  # alias
    cpp_fused_cos_mul_sin_2(buf5, buf6)
    del buf5
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = reinterpret_tensor(buf3, (64, 1, 64), (64, 64, 1), 0); del buf3  # reuse
        buf8.copy_(reinterpret_tensor(buf7, (64, 1, 64), (64, 0, 1), 0), False)
        del buf7
        buf9 = empty_strided_cuda((64, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf8, (64, 64), (64, 1), 0), reinterpret_tensor(arg5_1, (64, 128), (128, 1), 0), out=buf9)
        del arg5_1
        buf11 = reinterpret_tensor(buf2, (2, 32, 64), (2048, 64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [bd], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (2, 32, 64), (64, 128, 1), 0), reinterpret_tensor(buf9, (2, 64, 64), (64, 1, 128), 0), out=buf11)
        buf15 = reinterpret_tensor(buf4, (1, 2, 32, 32), (2048, 1024, 32, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_3, add_2, add_3, , mul_1, attn_prob], Original ATen: [aten.index_select, aten.add, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_amax_index_select_mul_sub_3.run(buf15, buf11, 64, 32, stream=stream0)
        buf14 = reinterpret_tensor(buf11, (32, 128), (128, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (32, 64), (64, 1), 0), reinterpret_tensor(arg4_1, (64, 128), (128, 1), 0), out=buf14)
        del arg4_1
        buf16 = reinterpret_tensor(buf10, (2, 32, 64), (2048, 64, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [attn_vec], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (2, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf14, (2, 32, 64), (64, 128, 1), 0), out=buf16)
        buf17 = reinterpret_tensor(buf14, (32, 64, 2, 1, 1), (128, 2, 1, 1, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf16, buf17, 4096, stream=stream0)
        buf18 = reinterpret_tensor(buf9, (64, 2, 1, 64, 1), (128, 64, 64, 1, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(arg8_1, buf18, 64, 128, stream=stream0)
        del arg8_1
        buf19 = reinterpret_tensor(buf15, (32, 64), (64, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf17, (32, 128), (128, 1), 0), reinterpret_tensor(buf18, (128, 64), (64, 1), 0), out=buf19)
        buf23 = reinterpret_tensor(buf19, (32, 1, 64), (64, 64, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [attn_out_2, output], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_6.run(buf23, buf0, arg9_1, arg10_1, 32, 64, stream=stream0)
        del arg10_1
        del arg9_1
        buf24 = reinterpret_tensor(buf17, (32, 128), (128, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf23, (32, 64), (64, 1), 0), reinterpret_tensor(arg11_1, (64, 128), (1, 64), 0), out=buf24)
        del arg11_1
        buf25 = reinterpret_tensor(buf24, (32, 1, 128), (128, 128, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf25, arg12_1, 4096, stream=stream0)
        del arg12_1
        buf26 = empty_strided_cuda((32, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf25, (32, 128), (128, 1), 0), reinterpret_tensor(arg13_1, (128, 64), (1, 128), 0), out=buf26)
        del arg13_1
        buf30 = reinterpret_tensor(buf26, (32, 1, 64), (64, 64, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [add_5, output_6], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_8.run(buf30, arg14_1, buf23, arg15_1, arg16_1, 32, 64, stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        buf31 = reinterpret_tensor(buf25, (32, 128), (128, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf30, (32, 64), (64, 1), 0), reinterpret_tensor(arg17_1, (64, 128), (128, 1), 0), out=buf31)
        del arg17_1
        buf32 = reinterpret_tensor(buf16, (32, 128), (128, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf30, (32, 64), (64, 1), 0), reinterpret_tensor(arg18_1, (64, 128), (128, 1), 0), out=buf32)
        del arg18_1
        buf33 = reinterpret_tensor(buf1, (32, 1, 2, 64), (128, 1, 64, 1), 0); del buf1  # reuse
        buf36 = empty_strided_cuda((32, 1, 2, 64), (128, 1, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_6, add_7], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf31, arg21_1, arg22_1, buf33, buf36, 4096, stream=stream0)
        del arg21_1
        del arg22_1
        del buf31
        buf34 = reinterpret_tensor(buf23, (2, 32, 32), (1024, 32, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [ac_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (2, 32, 64), (64, 128, 1), 0), reinterpret_tensor(buf32, (2, 64, 32), (64, 1, 128), 0), out=buf34)
        del buf32
        del buf33
        buf35 = reinterpret_tensor(buf18, (64, 128), (128, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf8, (64, 64), (64, 1), 0), reinterpret_tensor(arg20_1, (64, 128), (128, 1), 0), out=buf35)
        del arg20_1
        buf37 = reinterpret_tensor(buf8, (2, 32, 64), (2048, 64, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [bd_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (2, 32, 64), (64, 128, 1), 0), reinterpret_tensor(buf35, (2, 64, 64), (64, 1, 128), 0), out=buf37)
        buf41 = reinterpret_tensor(buf34, (1, 2, 32, 32), (2048, 1024, 32, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_7, add_8, add_9, , mul_1, attn_prob_2], Original ATen: [aten.index_select, aten.add, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_amax_index_select_mul_sub_3.run(buf41, buf37, 64, 32, stream=stream0)
        buf40 = reinterpret_tensor(buf37, (32, 128), (128, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf30, (32, 64), (64, 1), 0), reinterpret_tensor(arg19_1, (64, 128), (128, 1), 0), out=buf40)
        del arg19_1
        buf42 = reinterpret_tensor(buf36, (2, 32, 64), (2048, 64, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (2, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf40, (2, 32, 64), (64, 128, 1), 0), out=buf42)
        buf43 = reinterpret_tensor(buf40, (32, 64, 2, 1, 1), (128, 2, 1, 1, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf42, buf43, 4096, stream=stream0)
        del buf42
        buf44 = reinterpret_tensor(buf35, (64, 2, 1, 64, 1), (128, 64, 64, 1, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(arg23_1, buf44, 64, 128, stream=stream0)
        del arg23_1
        buf45 = reinterpret_tensor(buf41, (32, 64), (64, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf43, (32, 128), (128, 1), 0), reinterpret_tensor(buf44, (128, 64), (64, 1), 0), out=buf45)
        del buf44
        buf49 = reinterpret_tensor(buf45, (32, 1, 64), (64, 64, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [attn_out_5, output_7], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_6.run(buf49, buf30, arg24_1, arg25_1, 32, 64, stream=stream0)
        del arg24_1
        del arg25_1
        buf50 = reinterpret_tensor(buf43, (32, 128), (128, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (32, 64), (64, 1), 0), reinterpret_tensor(arg26_1, (64, 128), (1, 64), 0), out=buf50)
        del arg26_1
        buf51 = reinterpret_tensor(buf50, (32, 1, 128), (128, 128, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [output_9], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf51, arg27_1, 4096, stream=stream0)
        del arg27_1
        buf52 = empty_strided_cuda((32, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf51, (32, 128), (128, 1), 0), reinterpret_tensor(arg28_1, (128, 64), (1, 128), 0), out=buf52)
        del arg28_1
        del buf51
        buf56 = reinterpret_tensor(buf52, (32, 1, 64), (64, 64, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [add_11, output_13], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_8.run(buf56, arg29_1, buf49, arg30_1, arg31_1, 32, 64, stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        del buf49
    return (reinterpret_tensor(buf56, (1, 32, 64), (64, 64, 1), 0), buf0, buf30, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((10000, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((2, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((2, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
