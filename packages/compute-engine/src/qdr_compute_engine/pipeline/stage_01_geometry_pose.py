"""Stage 1 -- geometry and pose.

Accepts a pre-built geometry on the context, or parses one from the config's ``xyz`` or
``pdb`` field, then validates and canonicalises it.
"""

from qdr_compute_engine.geometry.pose_builder import (
    build_pose,
    parse_pdb,
    parse_xyz,
    require_closed_shell,
)
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_core.errors import ValidationError

STAGE_NAME = "01_geometry_pose"


def run(context: StageContext) -> StageContext:
    geometry = context.geometry

    if geometry is None:
        config = context.config
        if "xyz" in config:
            geometry = parse_xyz(str(config["xyz"]))
        elif "pdb" in config:
            geometry = parse_pdb(str(config["pdb"]), charge=int(config.get("charge", 0)))
        else:
            raise ValidationError(
                "no geometry available: provide a Geometry on the context or an "
                "'xyz'/'pdb' field in the config",
                config_keys=sorted(config),
            )

    charge = int(context.config.get("charge", geometry.charge))
    multiplicity = int(context.config.get("spin_multiplicity", geometry.spin_multiplicity))
    geometry = type(geometry)(
        atoms=geometry.atoms,
        charge=charge,
        spin_multiplicity=multiplicity,
        units=geometry.units,
    )

    prepared = build_pose(geometry)

    # Fail here rather than five stages later with a wrong number: everything downstream
    # assumes a closed-shell RHF reference, and nothing downstream checks.
    require_closed_shell(prepared)

    return context.evolve(geometry=prepared)
