"""Behavioral tests for the data-generating process.

These assert properties the rest of the project depends on, not merely that the
code runs: seed reproducibility, the early-tenure hazard spike, the
common-random-numbers guarantee (a zero-effect action reproduces the realized
timeline and any helpful action can only delay churn), agreement between the
analytic ground truth and the sampled draw, and the headline stage-conditional
asymmetry that Phase 5 must recover.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import expit
from scipy.stats import spearmanr

from src.dgp.config import ACTIONS, STAGES, DEFAULT_CONFIG, DGPConfig
from src.dgp.generate import (
    _build_panel,
    empirical_hazard_by_tenure,
    generate_dataset,
)

HELPFUL_ACTIONS = [a for a in ACTIONS if a != "no_action"]
ATTRITION_CODE = STAGES.index("attrition_in_progress")


@pytest.fixture(scope="module")
def artifacts():
    """The canonical population, generated once and shared across tests."""
    return generate_dataset(DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_seed_reproducibility():
    cfg = DGPConfig(n_customers=500, max_periods=12)
    a = generate_dataset(cfg, seed=7)
    b = generate_dataset(cfg, seed=7)
    assert_frame_equal(a.realized_panel, b.realized_panel)
    assert_frame_equal(a.experiment_panel, b.experiment_panel)
    assert_frame_equal(a.counterfactual_lookup, b.counterfactual_lookup)
    assert_frame_equal(a.survival_curves, b.survival_curves)

    different = generate_dataset(cfg, seed=8)
    assert not a.realized_panel.equals(different.realized_panel)


# --------------------------------------------------------------------------- #
# calibration: the early-tenure hazard spike
# --------------------------------------------------------------------------- #
def test_hazard_decreases_with_tenure(artifacts):
    curve = empirical_hazard_by_tenure(artifacts.realized_panel)
    hazard = curve.sort_values("tenure")["hazard"].to_numpy()
    tenure = curve.sort_values("tenure")["tenure"].to_numpy()

    assert hazard[0] == hazard.max(), "tenure-1 hazard must be the global maximum"
    assert np.all(np.diff(hazard[:6]) < 0), "hazard must strictly fall over early tenure"
    assert hazard[:3].mean() > 2.0 * hazard[-3:].mean(), "early spike must dwarf the tail"
    rho, _ = spearmanr(tenure, hazard)
    assert rho < -0.3, f"hazard should trend down with tenure (rho={rho:.3f})"


# --------------------------------------------------------------------------- #
# common random numbers
# --------------------------------------------------------------------------- #
def test_zero_effect_action_reproduces_realized_timeline(artifacts):
    """A zero-effect action's hazard is the untreated hazard, and resolving churn
    from the shared uniforms reproduces the realized timeline exactly."""
    base = artifacts.outcomes["no_action"]

    # no_action adds zero tau, so its hazard equals expit(base_logit)
    np.testing.assert_allclose(base.hazard, expit(artifacts.base_logit))

    # independently re-resolve churn from base_logit and the shared uniforms
    crossed = artifacts.churn_uniforms < expit(artifacts.base_logit)
    churned = crossed.any(axis=1)
    churn_period = np.where(churned, crossed.argmax(axis=1) + 1, 0)
    assert np.array_equal(churn_period, base.churn_period)

    # and the panel built from the no_action outcome is the realized panel
    rebuilt = _build_panel(
        artifacts.static,
        artifacts.activation,
        artifacts.usage,
        artifacts.support,
        artifacts.stage_codes,
        base.churned,
        base.churn_period,
        base.months_active,
    )
    assert_frame_equal(rebuilt, artifacts.realized_panel)


def test_helpful_action_can_only_delay_churn(artifacts):
    """With common random numbers, a hazard-lowering action never moves churn
    earlier (sampled) and never lowers expected retained months (analytic)."""
    base = artifacts.outcomes["no_action"]
    horizon = artifacts.config.max_periods
    base_cp = np.where(base.churned, base.churn_period, horizon + 1)
    for action in HELPFUL_ACTIONS:
        oc = artifacts.outcomes[action]
        cp = np.where(oc.churned, oc.churn_period, horizon + 1)
        assert np.all(cp >= base_cp), f"{action} moved churn earlier under CRN"
        assert np.all(oc.expected_months >= base.expected_months - 1e-9)


def test_treatment_is_not_a_noop(artifacts):
    """Sanity guard: onboarding_activation actually changes some timelines."""
    base = artifacts.outcomes["no_action"]
    onb = artifacts.outcomes["onboarding_activation"]
    assert np.any(onb.churn_period != base.churn_period)
    assert onb.delta_expected_months.mean() > 0.5


# --------------------------------------------------------------------------- #
# analytic ground truth agrees with the sampled draw (also catches off-by-one)
# --------------------------------------------------------------------------- #
def test_analytic_matches_sampled(artifacts):
    for action in ACTIONS:
        oc = artifacts.outcomes[action]
        diff = abs(oc.months_active.mean() - oc.expected_months.mean())
        assert diff < 0.4, f"{action}: sampled vs analytic months differ by {diff:.3f}"


# --------------------------------------------------------------------------- #
# the headline stage-conditional asymmetry (DGP ground truth)
# --------------------------------------------------------------------------- #
def test_onboarding_effect_dominates_winback(artifacts):
    enters_attrition = (artifacts.stage_codes == ATTRITION_CODE).any(axis=1)
    onb = artifacts.outcomes["onboarding_activation"]
    wb = artifacts.outcomes["winback_discount"]

    onb_months = onb.delta_expected_months.mean()
    wb_months = wb.delta_expected_months[enters_attrition].mean()
    assert onb_months > 2.0 * wb_months, (
        f"onboarding save ({onb_months:.2f} mo) should dwarf win-back "
        f"({wb_months:.2f} mo among attrition entrants)"
    )
    assert onb.ite_save.mean() > wb.ite_save[enters_attrition].mean()


# --------------------------------------------------------------------------- #
# stage partition, ranges, NaN, and the experiment's treatment variation
# --------------------------------------------------------------------------- #
def test_stage_partition_non_degenerate(artifacts):
    fractions = artifacts.realized_panel["stage"].value_counts(normalize=True)
    for stage in STAGES:
        assert stage in fractions.index, f"stage {stage} never occurs"
        assert fractions[stage] > 0.03, f"stage {stage} is degenerate ({fractions[stage]:.3f})"


def test_value_ranges_and_no_nan(artifacts):
    horizon = artifacts.config.max_periods
    lk = artifacts.counterfactual_lookup
    assert lk["survive_horizon_prob"].between(0.0, 1.0).all()
    assert lk["expected_months"].between(1.0, horizon).all()
    assert lk["expected_discounted_months"].between(0.0, horizon).all()

    sc = artifacts.survival_curves
    assert sc["survival_prob"].between(0.0, 1.0).all()
    assert sc["hazard"].between(0.0, 1.0, inclusive="neither").all()

    assert not artifacts.realized_panel.isna().any().any()
    assert not artifacts.experiment_panel.isna().any().any()

    # survival is non-increasing within each (customer, action)
    for action in ACTIONS:
        surv = artifacts.outcomes[action].survival
        assert np.all(np.diff(surv, axis=1) <= 1e-12)


def test_experiment_panel_has_treatment_variation(artifacts):
    """Honest uplift training needs every action represented, including control."""
    assigned = set(artifacts.experiment_panel["assigned_action"].unique())
    assert assigned == set(ACTIONS)


def test_counterfactual_lookup_parquet_roundtrip(artifacts, tmp_path):
    """The nullable Int64 churn-period column must survive a parquet round-trip."""
    import pandas as pd

    path = tmp_path / "lookup.parquet"
    artifacts.counterfactual_lookup.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)
    assert_frame_equal(reloaded, artifacts.counterfactual_lookup)
    assert reloaded["sampled_churn_period"].dtype == "Int64"
