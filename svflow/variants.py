"""
SV-Flow 变体配置定义 — 用于消融实验 (Appendix A)

五种变体:
  1. core       — 纯 SVGD 互斥 + 时间退火 (无 TP, 无 OP)  ← 推荐配置
  2. full       — SVGD + 切平面投影(TP) + 正交动能保护(OP)
  3. wo_op      — SVGD + TP only (移除正交保护)
  4. wo_tp      — SVGD + OP only (移除切平面投影)
  5. isotropic  — 1/r² 各向同性距离惩罚 (Metadiffusion 风格基线)

用法:
    from svflow.variants import get_config
    cfg = get_config('core')
    sampler = SVFlowSampler(model, n_trajectories=10, **cfg)
"""


def get_config(variant_name: str) -> dict:
    """
    返回指定变体的 Sampler 配置字典。

    Args:
        variant_name: 变体标识符
            'core'      — SVGD kernel only (推荐)
            'full'      — SVGD + TP + OP
            'wo_op'     — SVGD + TP, 无 OP
            'wo_tp'     — SVGD + OP, 无 TP
            'isotropic' — 1/r² 各向同性排斥 (无 SVGD 核)

    Returns:
        dict: 传递给 SVFlowSampler.__init__() 的关键词参数
    """
    # 基础配置 (所有变体共享)
    base = {
        'lambda_max': 1.0,
        't_on': 0.5,
        'd_min': 2.0,
    }

    variants = {
        'core': {
            **base,
            'use_svgd_kernel': True,
            'use_tangent_projection': False,
            'use_orthogonal_preservation': False,
        },
        'full': {
            **base,
            'use_svgd_kernel': True,
            'use_tangent_projection': True,
            'use_orthogonal_preservation': True,
        },
        'wo_op': {
            **base,
            'use_svgd_kernel': True,
            'use_tangent_projection': True,
            'use_orthogonal_preservation': False,
        },
        'wo_tp': {
            **base,
            'use_svgd_kernel': True,
            'use_tangent_projection': False,
            'use_orthogonal_preservation': True,
        },
        'isotropic': {
            **base,
            'use_svgd_kernel': False,
            'use_tangent_projection': False,
            'use_orthogonal_preservation': False,
        },
    }

    if variant_name not in variants:
        raise ValueError(
            f"Unknown variant: '{variant_name}'. "
            f"Available: {sorted(variants.keys())}"
        )
    return variants[variant_name]


def describe_variant(variant_name: str) -> str:
    """返回变体的人类可读描述。"""
    descriptions = {
        'core':       'SV-Flow Core (SVGD kernel + time annealing, no TP/OP)',
        'full':       'SV-Flow FULL (SVGD + TP + OP)',
        'wo_op':      'SV-Flow w/o OP (SVGD + TP, orthogonal protection removed)',
        'wo_tp':      'SV-Flow w/o TP (SVGD + OP, tangent projection removed)',
        'isotropic':  'Isotropic Repulsion (1/r² distance penalty, no SVGD kernel)',
    }
    return descriptions.get(variant_name, f'Unknown variant: {variant_name}')
