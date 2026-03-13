"""
config_utils/__init__.py

公開 API。各子模組用途：

  context.py          → build_sim_context, build_mask_context
  constants.py        → 物理常數與閾值
  mask_io.py          → load_solid_mask
  geometry.py         → fill_geometry, calc_l_char, calc_max_blockage
  feasibility.py      → check_feasibility
  blockage_adjuster.py→ fill_blockage_adj
  nu_sampler.py       → fill_nu_sample
  steps_calc.py       → fill_physics_and_steps
  config_assembler.py → build_config
  preview.py          → print_re_preview, print_summary
"""

from .constants import CS2, CS, MA_LIMIT, TAU_MIN, U_STEP_FACTOR, U_GAP_MAX, MIN_OPEN, RE_MAX
from .mask_io import load_solid_mask
from .geometry import fill_geometry, calc_l_char, calc_max_blockage
from .feasibility import check_feasibility
from .blockage_adjuster import fill_blockage_adj
from .nu_sampler import fill_nu_sample
from .steps_calc import fill_physics_and_steps
from .config_assembler import build_config
from .preview import print_re_preview, print_summary
from .context import build_sim_context, build_mask_context

__all__ = [
    "CS2", "CS", "MA_LIMIT", "TAU_MIN", "U_STEP_FACTOR", "U_GAP_MAX", "MIN_OPEN", "RE_MAX",
    "load_solid_mask",
    "fill_geometry", "calc_l_char", "calc_max_blockage",
    "check_feasibility",
    "fill_blockage_adj",
    "fill_nu_sample",
    "fill_physics_and_steps",
    "build_config",
    "print_re_preview", "print_summary",
    "build_sim_context", "build_mask_context",
]
