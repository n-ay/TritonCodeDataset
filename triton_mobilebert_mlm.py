# Triton kernels for MobileBert_MLM
# Model: google/mobilebert-uncased

triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 31, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (1 + x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 10000, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tl.broadcast_to(tmp13, [XBLOCK])) & (tl.broadcast_to(tmp13, [XBLOCK]) < 10000)) | ~(tmp8), "index out of bounds: 0 <= tl.broadcast_to(tmp13, [XBLOCK]) < 10000")
    tmp15 = tl.load(in_ptr1 + (128*tmp13 + (x0)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (x1), tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full([XBLOCK], 10000, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK]) < 10000)) | ~(tmp21), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK]) < 10000")
    tmp28 = tl.load(in_ptr1 + (128*tmp26 + ((-128) + x0)), tmp21, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp0 >= tmp19
    tmp30 = tl.full([1], 384, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = (-1) + x1
    tmp33 = tl.full([1], 0, tl.int64)
    tmp34 = tmp32 >= tmp33
    tmp35 = tmp34 & tmp29
    tmp36 = tl.load(in_ptr0 + ((-1) + x1), tmp35, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.full([XBLOCK], 10000, tl.int32)
    tmp38 = tmp36 + tmp37
    tmp39 = tmp36 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp36)
    tl.device_assert(((0 <= tl.broadcast_to(tmp40, [XBLOCK])) & (tl.broadcast_to(tmp40, [XBLOCK]) < 10000)) | ~(tmp35), "index out of bounds: 0 <= tl.broadcast_to(tmp40, [XBLOCK]) < 10000")
    tmp42 = tl.load(in_ptr1 + (128*tmp40 + ((-256) + x0)), tmp35, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp29, tmp42, tmp43)
    tmp45 = tl.where(tmp21, tmp28, tmp44)
    tmp46 = tl.where(tmp4, tmp17, tmp45)
    tl.store(out_ptr0 + (x2), tmp46, None)
''', device_str='cuda')

triton_poi_fused_add_embedding_mul_1 = async_compile.triton('triton_poi_fused_add_embedding_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK], 512, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert((0 <= tmp7) & (tmp7 < 512), "index out of bounds: 0 <= tmp7 < 512")
    tmp9 = tl.load(in_ptr2 + (x0 + 128*tmp7), None)
    tmp10 = tmp2 + tmp9
    tmp12 = tl.full([XBLOCK], 2, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tl.device_assert((0 <= tmp15) & (tmp15 < 2), "index out of bounds: 0 <= tmp15 < 2")
    tmp17 = tl.load(in_ptr4 + (x0 + 128*tmp15), None)
    tmp18 = tmp10 + tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')

triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')

triton_poi_fused__scaled_dot_product_efficient_attention_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = -3.4028234663852886e+38
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(out_ptr1 + (x2), tmp5, xmask)
''', device_str='cuda')

triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 68608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')

triton_poi_fused_relu_5 = async_compile.triton('triton_poi_fused_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 99328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')

triton_poi_fused_add_mul_6 = async_compile.triton('triton_poi_fused_add_mul_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')

@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 384)
    x1 = xindex // 384
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 31, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (1 + x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 10000, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tl.broadcast_to(tmp13, [XBLOCK])) & (tl.broadcast_to(tmp13, [XBLOCK]) < 10000)) | ~(tmp8), "index out of bounds: 0 <= tl.broadcast_to(tmp13, [XBLOCK]) < 10000")
    tmp15 = tl.load(in_ptr1 + (128*tmp13 + (x0)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (x1), tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full([XBLOCK], 10000, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK]) < 10000)) | ~(tmp21), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK]) < 10000")
    tmp28 = tl.load(in_ptr1 + (128*tmp26 + ((-128) + x0)), tmp21, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp0 >= tmp19
    tmp30 = tl.full([1], 384, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = (-1) + x1
    tmp33 = tl.full([1], 0, tl.int64)
    tmp34 = tmp32 >= tmp33
    tmp35 = tmp34 & tmp29
    tmp36 = tl.load(in_ptr0 + ((-1) + x1), tmp35, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.full([XBLOCK], 10000, tl.int32)
    tmp38 = tmp36 + tmp37
    tmp39 = tmp36 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp36)
    tl.device_assert(((0 <= tl.broadcast_to(tmp40, [XBLOCK])) & (tl.broadcast_to(tmp40, [XBLOCK]) < 10000)) | ~(tmp35), "index out of bounds: 0 <= tl.broadcast_to(tmp40, [XBLOCK]) < 10000")
    tmp42 = tl.load(in_ptr1 + (128*tmp40 + ((-256) + x0)), tmp35, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp29, tmp42, tmp43)
    tmp45 = tl.where(tmp21, tmp28, tmp44)
    tmp46 = tl.where(tmp4, tmp17, tmp45)
    tl.store(out_ptr0 + (x2), tmp46, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ey/ceyajastmfdb4mdiy6wt76kg4czzdsebs2d2yxyjdw234tsgnx4h.py
# Topologically Sorted Source Nodes: [position_embeddings, add, token_type_embeddings, embeddings, mul_1, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   embeddings => add_1
#   embeddings_1 => add_2
#   mul_1 => mul_1
#   position_embeddings => embedding_1
#   token_type_embeddings => embedding_2
# Graph fragment:
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg7_1, %slice_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %embedding_1), kwargs = {})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg8_1, %arg2_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %arg9_1), kwargs = {})
#   %add_2 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg10_1), kwargs = {})
triton_poi_fused_add_embedding_mul_1 = async_compile.triton('triton_poi_fused_add_embedding_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_add_embedding_mul_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK], 512, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert((0 <= tmp7) & (tmp7 < 512), "index out of bounds: 0 <= tmp7 < 512")
    tmp9 = tl.load(in_ptr2 + (x0 + 128*tmp7), None)
    tmp10 = tmp2 + tmp9
    tmp12 = tl.full([XBLOCK], 2, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tl.device_assert((0 <= tmp15) & (tmp15 < 2), "index out of bounds: 0 <= tmp15 < 2")
    tmp17 = tl.load(in_ptr4 + (x0 + 128*tmp15), None)
    tmp18 = tmp10 + tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(in_out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tu/ctum7jwabhgfmab62iivzu7y6iph2sj7wpun4khfdzzuwddzaekk.py
# Topologically Sorted Source Nodes: [mul_3, layer_input_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   layer_input_3 => add_4
#   mul_3 => mul_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %arg17_1), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg18_1), kwargs = {})
triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50688}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_add_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/2l/c2lo2hurgr2bzwrjaz4ottzvctlgyyoek2pm4dd665mv3o6erlk5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default, _scaled_dot_product_efficient_attention_default_1
# Graph fragment:
#   %_scaled_dot_product_efficient_attention_default_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_4, %permute_6, %permute_8, %expand_default_1, False), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_24, %permute_26, %permute_28, %expand_default, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16640}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = -3.4028234663852886e+38
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(out_ptr1 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ts/ctsodjphcb4pnx2hhgxpmiv6sj5gtfcmm3ch3vmfkhb7z5ocjspq.py
# Topologically Sorted Source Nodes: [mul_2, layer_input_1, add_6, mul_4, layer_outputs_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_6
#   layer_input_1 => add_3
#   layer_outputs_1 => add_7
#   mul_2 => mul_2
#   mul_4 => mul_4
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %arg13_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg14_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_23, %add_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %arg27_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg28_1), kwargs = {})
triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 68608}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_add_mul_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/le/cleoeaqbqtapigvzpujhnw6df6wl3a3xfrp457ba76prrhf24k3g.py
# Topologically Sorted Source Nodes: [hidden_states_1], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   hidden_states_1 => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_25,), kwargs = {})
triton_poi_fused_relu_5 = async_compile.triton('triton_poi_fused_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 99328}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dx/cdxkuy6vm43zi5hxoo3uonw57toyxvutfun7eezoegpxwgyss7ge.py
# Topologically Sorted Source Nodes: [add_8, mul_5, layer_outputs_3], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_8 => add_8
#   layer_outputs_3 => add_9
#   mul_5 => mul_5
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_27, %add_7), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %arg33_1), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg34_1), kwargs = {})
triton_poi_fused_add_mul_6 = async_compile.triton('triton_poi_fused_add_mul_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=40, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D8652C598408CE48EF7F8CD5982693EC810713EFE0D363B5F98854BFCB8CCE70', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67072}},
    min_elem_per_thread=0
)


@triton.jit
def triton_poi_fused_add_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 32), (32, 1))
    assert_size_stride(arg1_1, (1, 32), (32, 1))
    assert_size_stride(arg2_1, (1, 32), (32, 1))
    assert_size_stride(arg3_1, (10000, 128), (128, 1))
    assert_size_stride(arg4_1, (1, 512), (512, 1))
    assert_size_stride(arg5_1, (128, 384), (384, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (512, 128), (128, 1))
    assert_size_stride(arg8_1, (2, 128), (128, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, 128), (128, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, 128), (128, 1))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, 128), (128, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, 128), (128, 1))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, 128), (128, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, 128), (128, 1))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (256, 128), (128, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (128, 256), (256, 1))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (256, 128), (128, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (128, 256), (256, 1))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (256, 128), (128, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (128, 256), (256, 1))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (256, 128), (128, 1))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (128, 256), (256, 1))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, 128), (128, 1))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, 128), (128, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, 128), (128, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, 128), (128, 1))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, 128), (128, 1))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, 128), (128, 1))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, 128), (128, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (256, 128), (128, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (128, 256), (256, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (256, 128), (128, 1))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (128, 256), (256, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (256, 128), (128, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (128, 256), (256, 1))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (256, 128), (128, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (128, 256), (256, 1))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, 128), (128, 1))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32, 384), (12288, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg0_1, arg3_1, buf0, 12288, stream=stream0)
        del arg0_1
        del arg3_1
        buf1 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (32, 384), (384, 1), 0), reinterpret_tensor(arg5_1, (384, 128), (1, 384), 0), out=buf1)
        del arg5_1
        del buf0
        buf2 = reinterpret_tensor(buf1, (1, 32, 128), (4096, 128, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [position_embeddings, add, token_type_embeddings, embeddings, mul_1, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_embedding_mul_1.run(buf2, arg6_1, arg4_1, arg7_1, arg2_1, arg8_1, arg9_1, arg10_1, 4096, stream=stream0)
        del arg10_1
        del arg2_1
        del arg4_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf3 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf2, (32, 128), (128, 1), 0), reinterpret_tensor(arg15_1, (128, 128), (1, 128), 0), out=buf3)
        del arg15_1
        buf4 = reinterpret_tensor(buf3, (1, 32, 128), (4096, 128, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [mul_3, layer_input_3], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_2.run(buf4, arg16_1, arg17_1, arg18_1, 4096, stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf5 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg20_1, reinterpret_tensor(buf4, (32, 128), (128, 1), 0), reinterpret_tensor(arg19_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg19_1
        del arg20_1
        buf6 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg22_1, reinterpret_tensor(buf4, (32, 128), (128, 1), 0), reinterpret_tensor(arg21_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del arg21_1
        del arg22_1
        buf7 = reinterpret_tensor(buf4, (32, 128), (128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf2, (32, 128), (128, 1), 0), reinterpret_tensor(arg23_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf7)
        del arg23_1
        del arg24_1
        buf8 = empty_strided_cuda((1, 1, 32, 32), (1024, 0, 32, 1), torch.float32)
        buf40 = empty_strided_cuda((1, 1, 32, 32), (1024, 0, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_3.run(arg1_1, buf8, buf40, 1024, stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (1, 2, 32, 64), (4096, 64, 128, 1), 0), reinterpret_tensor(buf6, (1, 2, 32, 64), (4096, 64, 128, 1), 0), reinterpret_tensor(buf7, (1, 2, 32, 64), (4096, 64, 128, 1), 0), reinterpret_tensor(buf8, (1, 2, 32, 32), (1024, 0, 32, 1), 0), False)
        del buf5
        del buf8
        buf10 = buf9[0]
        assert_size_stride(buf10, (1, 2, 32, 64), (4096, 64, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf9
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf10, (32, 128), (128, 1), 0), reinterpret_tensor(arg25_1, (128, 128), (1, 128), 0), out=buf14)
        del arg25_1
        buf15 = reinterpret_tensor(buf10, (32, 128), (128, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf2, (32, 128), (128, 1), 0), reinterpret_tensor(arg11_1, (128, 128), (1, 128), 0), out=buf15)
        del arg11_1
        buf16 = reinterpret_tensor(buf14, (1, 32, 128), (4096, 128, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [mul_2, layer_input_1, add_6, mul_4, layer_outputs_1], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf16, arg26_1, buf15, arg12_1, arg13_1, arg14_1, arg27_1, arg28_1, 4096, stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg26_1
        del arg27_1
        del arg28_1
        buf17 = empty_strided_cuda((32, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf16, (32, 128), (128, 1), 0), reinterpret_tensor(arg29_1, (128, 256), (1, 128), 0), out=buf17)
        del arg29_1
        buf18 = reinterpret_tensor(buf17, (1, 32, 256), (8192, 256, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf18, arg30_1, 8192, stream=stream0)
        del arg30_1
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf18, (32, 256), (256, 1), 0), reinterpret_tensor(arg31_1, (256, 128), (1, 256), 0), out=buf19)
        del arg31_1
        buf20 = reinterpret_tensor(buf19, (1, 32, 128), (4096, 128, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [add_8, mul_5, layer_outputs_3], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf20, arg32_1, buf16, arg33_1, arg34_1, 4096, stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf21 = reinterpret_tensor(buf18, (32, 256), (256, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf20, (32, 128), (128, 1), 0), reinterpret_tensor(arg35_1, (128, 256), (1, 128), 0), out=buf21)
        del arg35_1
        buf22 = reinterpret_tensor(buf21, (1, 32, 256), (8192, 256, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf22, arg36_1, 8192, stream=stream0)
        del arg36_1
        buf23 = reinterpret_tensor(buf16, (32, 128), (128, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf22, (32, 256), (256, 1), 0), reinterpret_tensor(arg37_1, (256, 128), (1, 256), 0), out=buf23)
        del arg37_1
        buf24 = reinterpret_tensor(buf23, (1, 32, 128), (4096, 128, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [add_10, mul_6, layer_outputs_5], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf24, arg38_1, buf20, arg39_1, arg40_1, 4096, stream=stream0)
        del arg38_1
        del arg39_1
        del arg40_1
        buf25 = reinterpret_tensor(buf22, (32, 256), (256, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf24, (32, 128), (128, 1), 0), reinterpret_tensor(arg41_1, (128, 256), (1, 128), 0), out=buf25)
        del arg41_1
        buf26 = reinterpret_tensor(buf25, (1, 32, 256), (8192, 256, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf26, arg42_1, 8192, stream=stream0)
        del arg42_1
        buf27 = reinterpret_tensor(buf20, (32, 128), (128, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf26, (32, 256), (256, 1), 0), reinterpret_tensor(arg43_1, (256, 128), (1, 256), 0), out=buf27)
        del arg43_1
        buf28 = reinterpret_tensor(buf27, (1, 32, 128), (4096, 128, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [add_12, mul_7, layer_outputs_7], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf28, arg44_1, buf24, arg45_1, arg46_1, 4096, stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf29 = reinterpret_tensor(buf26, (32, 256), (256, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf28, (32, 128), (128, 1), 0), reinterpret_tensor(arg47_1, (128, 256), (1, 128), 0), out=buf29)
        del arg47_1
        buf30 = reinterpret_tensor(buf29, (1, 32, 256), (8192, 256, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf30, arg48_1, 8192, stream=stream0)
        del arg48_1
        buf31 = reinterpret_tensor(buf24, (32, 128), (128, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf30, (32, 256), (256, 1), 0), reinterpret_tensor(arg49_1, (256, 128), (1, 256), 0), out=buf31)
        del arg49_1
        buf32 = reinterpret_tensor(buf31, (1, 32, 128), (4096, 128, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [add_14, mul_8, layer_output_1], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf32, arg50_1, buf28, arg51_1, arg52_1, 4096, stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf33 = reinterpret_tensor(buf28, (32, 128), (128, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf32, (32, 128), (128, 1), 0), reinterpret_tensor(arg53_1, (128, 128), (1, 128), 0), out=buf33)
        del arg53_1
        buf34 = reinterpret_tensor(buf33, (1, 32, 128), (4096, 128, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [add_16, mul_9, layer_outputs_10], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf34, arg54_1, buf2, arg55_1, arg56_1, 4096, stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        buf35 = reinterpret_tensor(buf2, (32, 128), (128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf34, (32, 128), (128, 1), 0), reinterpret_tensor(arg61_1, (128, 128), (1, 128), 0), out=buf35)
        del arg61_1
        buf36 = reinterpret_tensor(buf35, (1, 32, 128), (4096, 128, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [mul_11, layer_input_7], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_2.run(buf36, arg62_1, arg63_1, arg64_1, 4096, stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        buf37 = reinterpret_tensor(buf32, (32, 128), (128, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg66_1, reinterpret_tensor(buf36, (32, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf37)
        del arg65_1
        del arg66_1
        buf38 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg68_1, reinterpret_tensor(buf36, (32, 128), (128, 1), 0), reinterpret_tensor(arg67_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf38)
        del arg67_1
        del arg68_1
        buf39 = reinterpret_tensor(buf36, (32, 128), (128, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg70_1, reinterpret_tensor(buf34, (32, 128), (128, 1), 0), reinterpret_tensor(arg69_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf39)
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf41 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf37, (1, 2, 32, 64), (4096, 64, 128, 1), 0), reinterpret_tensor(buf38, (1, 2, 32, 64), (4096, 64, 128, 1), 0), reinterpret_tensor(buf39, (1, 2, 32, 64), (4096, 64, 128, 1), 0), reinterpret_tensor(buf40, (1, 2, 32, 32), (1024, 0, 32, 1), 0), False)
        del buf37
        del buf38
        del buf40
        buf42 = buf41[0]
        assert_size_stride(buf42, (1, 2, 32, 64), (4096, 64, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf42, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf41
        buf46 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf42, (32, 128), (128, 1), 0), reinterpret_tensor(arg71_1, (128, 128), (1, 128), 0), out=buf46)
        del arg71_1
        buf47 = reinterpret_tensor(buf42, (32, 128), (128, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf34, (32, 128), (128, 1), 0), reinterpret_tensor(arg57_1, (128, 128), (1, 128), 0), out=buf47)
        del arg57_1
        buf48 = reinterpret_tensor(buf46, (1, 32, 128), (4096, 128, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [mul_10, layer_input_5, add_21, mul_12, layer_outputs_12], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf48, arg72_1, buf47, arg58_1, arg59_1, arg60_1, arg73_1, arg74_1, 4096, stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        del arg72_1
        del arg73_1
        del arg74_1
        buf49 = reinterpret_tensor(buf30, (32, 256), (256, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf48, (32, 128), (128, 1), 0), reinterpret_tensor(arg75_1, (128, 256), (1, 128), 0), out=buf49)
        del arg75_1
        buf50 = reinterpret_tensor(buf49, (1, 32, 256), (8192, 256, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf50, arg76_1, 8192, stream=stream0)
        del arg76_1
        buf51 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf50, (32, 256), (256, 1), 0), reinterpret_tensor(arg77_1, (256, 128), (1, 256), 0), out=buf51)
        del arg77_1
        buf52 = reinterpret_tensor(buf51, (1, 32, 128), (4096, 128, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [add_23, mul_13, layer_outputs_14], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf52, arg78_1, buf48, arg79_1, arg80_1, 4096, stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        buf53 = reinterpret_tensor(buf50, (32, 256), (256, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf52, (32, 128), (128, 1), 0), reinterpret_tensor(arg81_1, (128, 256), (1, 128), 0), out=buf53)
        del arg81_1
        buf54 = reinterpret_tensor(buf53, (1, 32, 256), (8192, 256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf54, arg82_1, 8192, stream=stream0)
        del arg82_1
        buf55 = reinterpret_tensor(buf48, (32, 128), (128, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf54, (32, 256), (256, 1), 0), reinterpret_tensor(arg83_1, (256, 128), (1, 256), 0), out=buf55)
        del arg83_1
        buf56 = reinterpret_tensor(buf55, (1, 32, 128), (4096, 128, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [add_25, mul_14, layer_outputs_16], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf56, arg84_1, buf52, arg85_1, arg86_1, 4096, stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        buf57 = reinterpret_tensor(buf54, (32, 256), (256, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf56, (32, 128), (128, 1), 0), reinterpret_tensor(arg87_1, (128, 256), (1, 128), 0), out=buf57)
        del arg87_1
        buf58 = reinterpret_tensor(buf57, (1, 32, 256), (8192, 256, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf58, arg88_1, 8192, stream=stream0)
        del arg88_1
        buf59 = reinterpret_tensor(buf52, (32, 128), (128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf58, (32, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 128), (1, 256), 0), out=buf59)
        del arg89_1
        buf60 = reinterpret_tensor(buf59, (1, 32, 128), (4096, 128, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [add_27, mul_15, layer_outputs_18], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf60, arg90_1, buf56, arg91_1, arg92_1, 4096, stream=stream0)
        del arg90_1
        del arg91_1
        del arg92_1
        buf61 = reinterpret_tensor(buf58, (32, 256), (256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf60, (32, 128), (128, 1), 0), reinterpret_tensor(arg93_1, (128, 256), (1, 128), 0), out=buf61)
        del arg93_1
        buf62 = reinterpret_tensor(buf61, (1, 32, 256), (8192, 256, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf62, arg94_1, 8192, stream=stream0)
        del arg94_1
        buf63 = reinterpret_tensor(buf56, (32, 128), (128, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf62, (32, 256), (256, 1), 0), reinterpret_tensor(arg95_1, (256, 128), (1, 256), 0), out=buf63)
        del arg95_1
        del buf62
        buf64 = reinterpret_tensor(buf63, (1, 32, 128), (4096, 128, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [add_29, mul_16, layer_output_3], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf64, arg96_1, buf60, arg97_1, arg98_1, 4096, stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        buf65 = reinterpret_tensor(buf60, (32, 128), (128, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf64, (32, 128), (128, 1), 0), reinterpret_tensor(arg99_1, (128, 128), (1, 128), 0), out=buf65)
        del arg99_1
        del buf64
        buf66 = reinterpret_tensor(buf65, (1, 32, 128), (4096, 128, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [add_31, mul_17, layer_outputs_21], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf66, arg100_1, buf34, arg101_1, arg102_1, 4096, stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del buf34
    return (buf66, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((10000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
