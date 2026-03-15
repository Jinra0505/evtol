"""Compatibility entry for consumer-facing TCE reporting extensions.

This module re-exports the existing runner implementation so legacy scripts can
keep using `project.src.runner` while TCE-focused workflows can import
`project.src.runner_new`.
"""

from .runner import (  # noqa: F401
    apply_named_scenario_entry,
    build_tce_scenarios_entry,
    main,
    run_equilibrium,
)
