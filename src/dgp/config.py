"""Data-generating process parameters and structural rules.

This module is the single source of truth for every DGP parameter. The
generator in :mod:`src.dgp.generate` reads all of its numbers from here; no
parameter is defined in the generator itself (``phases/phase1.md``: "do not bury
parameters in generate.py").

Model
-----
Customers are simulated in discrete monthly periods (KKBox-style subscription
churn) carrying an IBM-Telco-style covariate schema. Churn in period ``t`` is a
Bernoulli draw with a **logistic** discrete-time hazard::

    logit h_{i,t} = alpha(t) + x_{i,t} . beta + tau(stage_{i,t}, action)

where

* ``alpha(t)`` is the baseline churn log-odds, high at low tenure and decaying
  toward a floor -- the early-tenure hazard spike;
* ``x_{i,t} . beta`` is the linear predictor from static (Telco) and
  time-varying covariates;
* ``tau(stage, action)`` is a stage-conditional treatment effect (a shift in the
  churn logit) applied only under an intervention. In the realized (control)
  world the action is ``no_action`` and tau is zero everywhere.

The logit link is deliberate: Phase 2 must fit a *discrete-time logistic* hazard
(pooled logistic regression with a tenure spline) to match this link. A
continuous-time Cox proportional-hazards fit would be on a different link and
would not recover these coefficients -- that would be misspecification, not a
finding.

Counterfactual semantics and the estimand
------------------------------------------
A candidate action applied to a customer shifts the churn logit by
``tau(stage_{i,t}, action)`` in every period ``t``. tau is non-zero only in the
stage(s) an action targets, and because stages occur at characteristic tenures
the effect lands at the intended lifecycle moment. Covariate trajectories and
the per-period uniform draws that resolve the churn event are held fixed across
the realized and all counterfactual worlds (common random numbers), so a
zero-effect action reproduces the realized timeline exactly.

Because the hazards are authored here, the ground-truth treatment effect is
computed **analytically**, not from a single Monte-Carlo draw. For customer i
and action a let ``S_{i,a}(t) = prod_{s<=t} (1 - h_{i,a,s})``. The two estimands
later phases target are::

    survive_horizon_prob(i, a) = S_{i,a}(T)
    ITE_save(i, a)             = S_{i,a}(T) - S_{i,no_action}(T)
    expected_months(i, a)      = sum_{t=1..T} S_{i,a}(t-1)      (S_0 = 1)

``expected_months`` counts billed periods (a customer alive at the start of
period t is billed in t), so a churn drawn in period c yields exactly c billed
periods. Phase 3 validates an uplift model against ``ITE_save`` with a Qini
curve; Phase 4 turns the survival curve into residual lifetime value.

Re-churn is emergent, not a constant
------------------------------------
Treatment lowers the hazard but does **not** repair covariates. A win-back save
at attrition leaves usage collapsed, so the saved customer stays in the
attrition/at-risk region, faces a high covariate hazard, and re-churns quickly
-- so its ``expected_months`` gain is small. An onboarding save acts early, the
customer then progresses to ``established`` with a low hazard, and its
``expected_months`` gain is large. The re-churn penalty Phase 4 needs is
therefore produced by the DGP itself; it must not be re-introduced as a
hand-picked multiplier. ``generate.py``'s entry point prints this asymmetry and
``tests/test_dgp.py`` asserts it.

Artifacts and the train / ground-truth separation (enforced by convention)
--------------------------------------------------------------------------
The generator writes four parquet files to ``data/``:

* ``realized_panel.parquet``    -- the control (no-intervention) world, the only
  panel Phase 2 fits its hazard model on. Contains observables and the churn
  event; it does **not** contain the true hazard.
* ``experiment_panel.parquet``  -- a randomized-assignment RCT (each customer is
  assigned one action uniformly at random and its outcome realized under common
  random numbers). This is the *only* honest training substrate for Phase 3
  uplift: it is the only artifact with treatment variation. Randomized
  assignment makes ignorability hold by construction.
* ``counterfactual_lookup.parquet`` -- GROUND TRUTH. Per (customer, action): the
  analytic estimands above plus the single common-random-numbers draw used by
  the tests. Used to validate models and score policies. **Never a model input.**
* ``survival_curves.parquet``   -- GROUND TRUTH. Per (customer, action, period):
  the true hazard and survival probability, for Phase 2 calibration and Phase 4
  discounted residual LTV. **Never a model input.**

Scope assumptions stated plainly (no hidden assumptions)
--------------------------------------------------------
* One action per customer; no multi-touch sequencing and no contact fatigue.
  Channel-cost asymmetry and budget live in Phase 4/5. Multi-touch would require
  extending this DGP.
* Lifecycle stages are *definitional* ground truth (a rule, not an estimated
  partition). The thresholds below are round numbers on purpose; the claim under
  test is that conditioning on the stage helps, not that the cutoffs are optimal.
* Residual value is built from ``monthly_charges`` (revenue) scaled by
  :attr:`DGPConfig.contribution_margin`; Phase 4 owns discounting via
  :attr:`DGPConfig.monthly_discount_rate`.

All effect sizes are on the churn log-odds scale; negative tau lowers the hazard
(a save), positive beta raises it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- lifecycle stages and candidate actions ---------------------------------

STAGES: tuple[str, ...] = (
    "onboarding",
    "established",
    "at_risk",
    "attrition_in_progress",
)
"""Lifecycle stages, in their canonical code order (0..3)."""

ACTIONS: tuple[str, ...] = (
    "no_action",
    "onboarding_activation",
    "proactive_engagement",
    "winback_discount",
)
"""Candidate interventions. ``no_action`` (index 0) is the null action (tau == 0)."""

STAGE_CODE: dict[str, int] = {name: i for i, name in enumerate(STAGES)}
ACTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTIONS)}

# columns repeated from the per-customer static table into the long panels
STATIC_COLUMNS: tuple[str, ...] = (
    "contract",
    "payment_method",
    "plan_tier",
    "monthly_charges",
    "senior_citizen",
    "has_tech_support",
    "has_online_security",
    "has_streaming",
)


def _default_static_spec() -> dict[str, dict]:
    """Static (Telco-template) covariate sampling distributions and effects.

    Categorical covariates use reference-cell coding: ``beta`` is the churn
    log-odds shift of each level relative to an implicit reference level whose
    beta is 0, so ``alpha(t)`` is the baseline for the reference profile.
    """
    return {
        "contract": {
            "levels": ("month_to_month", "one_year", "two_year"),
            "probs": (0.55, 0.25, 0.20),
            "beta": {"month_to_month": 0.0, "one_year": -0.80, "two_year": -1.50},
        },
        "payment_method": {
            "levels": ("electronic_check", "mailed_check", "bank_transfer", "credit_card"),
            "probs": (0.34, 0.23, 0.22, 0.21),
            "beta": {
                "electronic_check": 0.45,
                "mailed_check": 0.10,
                "bank_transfer": -0.20,
                "credit_card": -0.25,
            },
        },
        "plan_tier": {
            "levels": ("basic", "standard", "premium"),
            "probs": (0.40, 0.35, 0.25),
            "beta": {"basic": 0.10, "standard": 0.0, "premium": -0.15},
            # monthly charges (USD) drawn from a tier-specific normal.
            "charge_mean": {"basic": 35.0, "standard": 65.0, "premium": 95.0},
            "charge_sd": {"basic": 8.0, "standard": 10.0, "premium": 12.0},
        },
        "senior_citizen": {"prob": 0.16, "beta": 0.35},
        "service_flags": {
            # name: (P(flag == 1), beta when flag == 1)
            "has_tech_support": (0.50, -0.30),
            "has_online_security": (0.50, -0.25),
            "has_streaming": (0.60, 0.05),
        },
        # monthly charges enter the hazard through a standardized z-score.
        "charges_ref_mean": 65.0,
        "charges_ref_sd": 25.0,
        "charges_beta": 0.15,
    }


def _default_treatment_matrix() -> dict[str, dict[str, float]]:
    """Stage-conditional treatment effects ``tau[action][stage]`` (logit shift).

    Read each row as: applying this action to a customer while they occupy this
    stage shifts their churn log-odds by tau. Zero means the action does nothing
    in that stage; negative means a save.

    The diagonal story: onboarding_activation is potent during onboarding,
    proactive_engagement works best on at-risk customers before usage collapses,
    and winback_discount only bites once attrition is in progress -- and even
    then far more weakly than onboarding activation does early. The headline
    assumption is visible as -1.60 (onboarding) vs -0.50 (win-back).
    """
    return {
        "no_action": {s: 0.0 for s in STAGES},
        "onboarding_activation": {
            "onboarding": -1.60,
            "established": 0.0,
            "at_risk": 0.0,
            "attrition_in_progress": 0.0,
        },
        "proactive_engagement": {
            "onboarding": -0.20,
            "established": -0.40,
            "at_risk": -0.90,
            "attrition_in_progress": -0.10,
        },
        "winback_discount": {
            "onboarding": 0.0,
            "established": 0.0,
            "at_risk": -0.25,
            "attrition_in_progress": -0.50,
        },
    }


@dataclass(frozen=True)
class DGPConfig:
    """All parameters of the data-generating process.

    Defaults define the canonical synthetic population. Tests construct smaller
    populations by overriding ``n_customers`` / ``max_periods``; nothing else
    needs to change.
    """

    # --- population and reproducibility ---
    n_customers: int = 5000
    max_periods: int = 24
    seed: int = 20260525

    # --- baseline hazard alpha(t) = floor + (peak - floor) * exp(-decay*(t-1)) ---
    # Tuned so the reference customer's hazard falls from ~0.18 at tenure 1 to
    # ~0.03 by tenure 12: the early-tenure spike Phase 2 must recover.
    alpha_peak: float = -1.50
    alpha_floor: float = -3.70
    alpha_decay: float = 0.25

    # --- static covariate schema (Telco template) ---
    static: dict = field(default_factory=_default_static_spec)

    # --- time-varying covariate dynamics and effects ---
    # Activation milestone progress: 1 - exp(-rate * tenure), rate ~ U(lo, hi).
    activation_rate_range: tuple[float, float] = (0.25, 0.90)
    activation_beta: float = -0.90  # fully activated lowers the hazard

    # Usage index: per-customer baseline + drift + AR(1) noise. A minority of
    # customers are "decliners" whose usage drifts down, eventually triggering
    # the at-risk / attrition stages.
    usage_base_sd: float = 0.50
    decliner_fraction: float = 0.25
    usage_drift_decliner: tuple[float, float] = (-0.12, 0.03)  # (mean, sd)
    usage_drift_stable: tuple[float, float] = (0.0, 0.02)
    usage_ar_phi: float = 0.60
    usage_ar_sd: float = 0.30
    usage_beta: float = -0.70  # low usage raises the hazard

    # Support contacts: Poisson with rate rising as usage falls.
    support_rate_intercept: float = -0.30
    support_rate_usage: float = -0.50
    support_max: int = 5
    support_beta: float = 0.25

    # --- lifecycle-stage rule thresholds (tenure + activation + usage/support) ---
    onboarding_max_tenure: int = 4
    activation_complete: float = 0.95
    attrition_usage_z: float = -1.00
    atrisk_usage_z: float = -0.40
    atrisk_support: int = 2

    # --- economics handed to Phase 4 (stated here so they are not hidden) ---
    monthly_discount_rate: float = 0.01  # ~12.7% annual; discounts residual value
    contribution_margin: float = 0.65  # fraction of monthly_charges that is margin

    # --- stage-conditional treatment effects keyed by (action, stage) ---
    treatment: dict = field(default_factory=_default_treatment_matrix)

    def treatment_matrix(self) -> np.ndarray:
        """Return tau as a ``(n_actions, n_stages)`` array in canonical order."""
        mat = np.zeros((len(ACTIONS), len(STAGES)), dtype=float)
        for action, row in self.treatment.items():
            a = ACTION_INDEX[action]
            for stage, value in row.items():
                mat[a, STAGE_CODE[stage]] = value
        return mat

    @property
    def discount_factor(self) -> float:
        """Per-period discount factor ``1 / (1 + monthly_discount_rate)``."""
        return 1.0 / (1.0 + self.monthly_discount_rate)


DEFAULT_CONFIG = DGPConfig()
"""The canonical population used by ``generate.py``'s entry point and the artifact."""


def baseline_logit_hazard(
    tenure: np.ndarray | float, cfg: DGPConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Baseline churn log-odds ``alpha(t)`` for 1-indexed ``tenure``.

    Decays from ``alpha_peak`` at tenure 1 toward ``alpha_floor`` as tenure
    grows, encoding the early-tenure hazard spike.
    """
    tenure_arr = np.asarray(tenure, dtype=float)
    return cfg.alpha_floor + (cfg.alpha_peak - cfg.alpha_floor) * np.exp(
        -cfg.alpha_decay * (tenure_arr - 1.0)
    )


def assign_stage_codes(
    tenure: np.ndarray,
    activation: np.ndarray,
    usage: np.ndarray,
    support: np.ndarray,
    cfg: DGPConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Map (tenure, activation, usage, support) to lifecycle-stage codes.

    Vectorized; all inputs broadcast to a common shape. Codes follow
    :data:`STAGES` order (onboarding=0, established=1, at_risk=2,
    attrition_in_progress=3). Precedence: a customer still inside their
    onboarding window and not yet activated is ``onboarding``; otherwise a
    collapsed usage index is ``attrition_in_progress``; otherwise mildly
    depressed usage or elevated support contacts is ``at_risk``; otherwise
    ``established``.
    """
    tenure_b, activation_b, usage_b, support_b = np.broadcast_arrays(
        tenure, activation, usage, support
    )
    codes = np.full(tenure_b.shape, STAGE_CODE["established"], dtype=np.int8)

    onboarding = (tenure_b <= cfg.onboarding_max_tenure) & (
        activation_b < cfg.activation_complete
    )
    attrition = (~onboarding) & (usage_b < cfg.attrition_usage_z)
    at_risk = (
        (~onboarding)
        & (~attrition)
        & ((usage_b < cfg.atrisk_usage_z) | (support_b >= cfg.atrisk_support))
    )

    # assign in precedence order (later writes win where masks overlap; they do not)
    codes[at_risk] = STAGE_CODE["at_risk"]
    codes[attrition] = STAGE_CODE["attrition_in_progress"]
    codes[onboarding] = STAGE_CODE["onboarding"]
    return codes
