"""Gurobi MIP models for tiny CPMP benchmarks.

The 4D slot-occupancy model is the recommended exact benchmark.  The 3D
integer-position model remains available as an archived comparison model.
"""

from .mip_3d_integer_position import solve_cpmp_3d_integer_position
from .mip_4d_slot_occupancy import solve_cpmp_4d_slot_occupancy
from .hybrid_solvers import FinalStackAssignmentMipSolver, PatternColumnGenerationSolver, RollingHorizonMipSolver

__all__ = [
    "solve_cpmp_3d_integer_position",
    "solve_cpmp_4d_slot_occupancy",
    "RollingHorizonMipSolver",
    "FinalStackAssignmentMipSolver",
    "PatternColumnGenerationSolver",
]
