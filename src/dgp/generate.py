"""Synthetic population generator.

Produces the realized (control) timeline, a randomized-assignment experiment for
honest uplift training, and the analytic counterfactual ground truth for every
candidate action -- all under common random numbers. See :mod:`src.dgp.config`
for the model, the estimands, and the train / ground-truth separation contract.

Common random numbers are enforced *structurally*: the per-period covariate
trajectories and the uniform draws that resolve the churn event are drawn once,
up front, from dedicated seeded streams. The per-action simulation
(:func:`_simulate_action`) is a pure function of those fixed arrays and makes no
random draw of its own, so the only thing that differs between the realized
world and any counterfactual is the deterministic treatment shift ``tau``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import (
    ACTIONS,
    DEFAULT_CONFIG,
    STAGES,
    STATIC_COLUMNS,
    DGPConfig,
    assign_stage_codes,
    baseline_logit_hazard,
)

# columns of each long discrete-time panel, in order
_PANEL_COLUMNS: tuple[str, ...] = (
    ("customer_id", "period", "tenure", "stage")
    + STATIC_COLUMNS
    + ("activation_progress", "usage_index", "support_contacts", "event", "at_risk")
)


@dataclass
class ActionOutcome:
    """Per-customer outcomes for one action: the sampled draw and the analytic truth."""

    hazard: np.ndarray  # (n, T) true per-period hazard
    survival: np.ndarray  # (n, T) S(t) = P(survive through period t)
    # sampled under common random numbers
    churned: np.ndarray  # (n,) bool
    churn_period: np.ndarray  # (n,) int, 1..T if churned else 0 (censored sentinel)
    months_active: np.ndarray  # (n,) int billed periods (= churn_period, or T if censored)
    # analytic ground truth
    expected_months: np.ndarray  # (n,) E[billed periods]
    expected_discounted_months: np.ndarray  # (n,) discounted E[billed periods]
    survive_horizon_prob: np.ndarray  # (n,) S(T)
    ite_save: np.ndarray  # (n,) S_a(T) - S_no_action(T)
    delta_expected_months: np.ndarray  # (n,) expected_months_a - expected_months_no_action


@dataclass
class DGPArtifacts:
    """Everything the generator produces, in memory (file writing is separate)."""

    config: DGPConfig
    static: pd.DataFrame
    realized_panel: pd.DataFrame
    experiment_panel: pd.DataFrame
    counterfactual_lookup: pd.DataFrame
    survival_curves: pd.DataFrame
    # arrays kept for tests and inspection
    outcomes: dict[str, ActionOutcome]
    churn_uniforms: np.ndarray  # (n, T) the common random numbers
    stage_codes: np.ndarray  # (n, T)
    base_logit: np.ndarray  # (n, T) logit with no treatment
    assigned_action_index: np.ndarray  # (n,) experiment assignment
    # time-varying covariate trajectories (identical across all actions)
    activation: np.ndarray  # (n, T)
    usage: np.ndarray  # (n, T)
    support: np.ndarray  # (n, T)


# --------------------------------------------------------------------------- #
# covariate draws
# --------------------------------------------------------------------------- #
def _sample_static(rng: np.random.Generator, n: int, cfg: DGPConfig) -> pd.DataFrame:
    """Draw per-customer static (Telco-template) covariates."""
    s = cfg.static
    contract = rng.choice(s["contract"]["levels"], size=n, p=s["contract"]["probs"])
    payment = rng.choice(
        s["payment_method"]["levels"], size=n, p=s["payment_method"]["probs"]
    )
    plan = rng.choice(s["plan_tier"]["levels"], size=n, p=s["plan_tier"]["probs"])
    charge_mean = np.array([s["plan_tier"]["charge_mean"][p] for p in plan])
    charge_sd = np.array([s["plan_tier"]["charge_sd"][p] for p in plan])
    charges = np.clip(rng.normal(charge_mean, charge_sd), 5.0, None)
    senior = (rng.random(n) < s["senior_citizen"]["prob"]).astype(int)

    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(n),
        "contract": contract,
        "payment_method": payment,
        "plan_tier": plan,
        "monthly_charges": np.round(charges, 2),
        "senior_citizen": senior,
    }
    for name, (prob, _beta) in s["service_flags"].items():
        data[name] = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(data)


def _static_logit(static_df: pd.DataFrame, cfg: DGPConfig) -> np.ndarray:
    """Static contribution to the churn logit, one value per customer."""
    s = cfg.static
    lp = np.zeros(len(static_df), dtype=float)
    lp += static_df["contract"].map(s["contract"]["beta"]).to_numpy()
    lp += static_df["payment_method"].map(s["payment_method"]["beta"]).to_numpy()
    lp += static_df["plan_tier"].map(s["plan_tier"]["beta"]).to_numpy()
    lp += s["senior_citizen"]["beta"] * static_df["senior_citizen"].to_numpy()
    for name, (_prob, beta) in s["service_flags"].items():
        lp += beta * static_df[name].to_numpy()
    z = (static_df["monthly_charges"].to_numpy() - s["charges_ref_mean"]) / s[
        "charges_ref_sd"
    ]
    lp += s["charges_beta"] * z
    return lp


def _simulate_time_varying(
    rng: np.random.Generator, n: int, cfg: DGPConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw activation progress, usage index, and support contacts: each (n, T)."""
    t_max = cfg.max_periods
    tenure = np.arange(1, t_max + 1)

    # activation milestone progress: monotone, saturating at 1.
    rate = rng.uniform(*cfg.activation_rate_range, size=n)
    activation = 1.0 - np.exp(-rate[:, None] * tenure[None, :])

    # usage: per-customer level + (decliner) drift + AR(1) noise.
    base = rng.normal(0.0, cfg.usage_base_sd, size=n)
    is_decliner = rng.random(n) < cfg.decliner_fraction
    drift = np.where(
        is_decliner,
        rng.normal(*cfg.usage_drift_decliner, size=n),
        rng.normal(*cfg.usage_drift_stable, size=n),
    )
    innov = rng.normal(0.0, cfg.usage_ar_sd, size=(n, t_max))
    ar = np.empty((n, t_max))
    ar[:, 0] = innov[:, 0]
    for t in range(1, t_max):
        ar[:, t] = cfg.usage_ar_phi * ar[:, t - 1] + innov[:, t]
    usage = base[:, None] + drift[:, None] * (tenure[None, :] - 1) + ar

    # support contacts: Poisson rate rises as usage falls.
    rate_support = np.exp(cfg.support_rate_intercept + cfg.support_rate_usage * usage)
    support = np.minimum(rng.poisson(rate_support), cfg.support_max)
    return activation, usage, support


# --------------------------------------------------------------------------- #
# per-action simulation (pure: no rng) -- this is where CRN is guaranteed
# --------------------------------------------------------------------------- #
def _simulate_action(
    base_logit: np.ndarray,
    stage_codes: np.ndarray,
    tau_row: np.ndarray,
    churn_uniforms: np.ndarray,
    discount_pow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one action's hazards, the sampled timeline, and the analytic truth.

    Pure function of pre-drawn arrays -- no random number is generated here, so
    the realized world and every counterfactual share the same ``churn_uniforms``
    and differ only by the deterministic ``tau`` shift. Returns
    ``(hazard, survival, churned, churn_period, months_active, expected_months,
    expected_discounted_months, survive_horizon_prob)``.
    """
    hazard = expit(base_logit + tau_row[stage_codes])  # (n, T)

    # sampled timeline under common random numbers
    crossed = churn_uniforms < hazard
    churned = crossed.any(axis=1)
    first_idx = np.argmax(crossed, axis=1)  # 0 where no crossing; masked by `churned`
    churn_period = np.where(churned, first_idx + 1, 0)
    t_max = hazard.shape[1]
    months_active = np.where(churned, churn_period, t_max)

    # analytic ground truth from the known hazards
    survival = np.cumprod(1.0 - hazard, axis=1)  # S(t), t = 1..T
    s_prev = np.empty_like(survival)  # S(t-1), with S(0) = 1
    s_prev[:, 0] = 1.0
    s_prev[:, 1:] = survival[:, :-1]
    expected_months = s_prev.sum(axis=1)
    expected_discounted_months = (s_prev * discount_pow[None, :]).sum(axis=1)
    survive_horizon_prob = survival[:, -1]

    return (
        hazard,
        survival,
        churned,
        churn_period,
        months_active,
        expected_months,
        expected_discounted_months,
        survive_horizon_prob,
    )


# --------------------------------------------------------------------------- #
# panel assembly
# --------------------------------------------------------------------------- #
def _build_panel(
    static_df: pd.DataFrame,
    activation: np.ndarray,
    usage: np.ndarray,
    support: np.ndarray,
    stage_codes: np.ndarray,
    churned: np.ndarray,
    churn_period: np.ndarray,
    months_active: np.ndarray,
    assigned_action: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assemble a long discrete-time panel: one row per (customer, at-risk period).

    A customer contributes rows for periods ``1..months_active`` (the periods in
    which they were billed). ``event`` is 1 in the churn period and 0 otherwise.
    """
    n, t_max = activation.shape
    period_idx = np.arange(1, t_max + 1)
    keep = period_idx[None, :] <= months_active[:, None]  # (n, T)

    event = np.zeros((n, t_max), dtype=int)
    churn_rows = np.where(churned)[0]
    event[churn_rows, churn_period[churned] - 1] = 1

    mask = keep.reshape(-1)
    stage_names = np.asarray(STAGES)[stage_codes.reshape(-1)[mask]]

    data: dict[str, np.ndarray] = {
        "customer_id": np.repeat(static_df["customer_id"].to_numpy(), t_max)[mask],
        "period": np.tile(period_idx, n)[mask],
        "tenure": np.tile(period_idx, n)[mask],
        "stage": stage_names,
    }
    for col in STATIC_COLUMNS:
        data[col] = np.repeat(static_df[col].to_numpy(), t_max)[mask]
    data["activation_progress"] = activation.reshape(-1)[mask]
    data["usage_index"] = usage.reshape(-1)[mask]
    data["support_contacts"] = support.reshape(-1)[mask]
    data["event"] = event.reshape(-1)[mask]
    data["at_risk"] = np.ones(int(mask.sum()), dtype=int)

    panel = pd.DataFrame(data)[list(_PANEL_COLUMNS)]
    if assigned_action is not None:
        panel.insert(1, "assigned_action", np.repeat(assigned_action, t_max)[mask])
    return panel


# --------------------------------------------------------------------------- #
# top-level generator
# --------------------------------------------------------------------------- #
def generate_dataset(cfg: DGPConfig = DEFAULT_CONFIG, seed: int | None = None) -> DGPArtifacts:
    """Generate the full synthetic dataset in memory.

    Uses three independent seeded streams (covariates, churn uniforms, RCT
    assignment) spawned from a single seed, so changing how many draws one
    stream makes never perturbs another.
    """
    if seed is None:
        seed = cfg.seed
    n, t_max = cfg.n_customers, cfg.max_periods
    cov_ss, churn_ss, assign_ss = np.random.SeedSequence(seed).spawn(3)
    cov_rng = np.random.default_rng(cov_ss)
    churn_rng = np.random.default_rng(churn_ss)
    assign_rng = np.random.default_rng(assign_ss)

    # --- draw everything stochastic up front (common random numbers) ---
    static_df = _sample_static(cov_rng, n, cfg)
    activation, usage, support = _simulate_time_varying(cov_rng, n, cfg)
    churn_uniforms = churn_rng.random((n, t_max))

    # --- deterministic structure ---
    tenure_row = np.arange(1, t_max + 1)
    alpha = baseline_logit_hazard(tenure_row, cfg)
    stage_codes = assign_stage_codes(tenure_row[None, :], activation, usage, support, cfg)
    base_logit = (
        alpha[None, :]
        + _static_logit(static_df, cfg)[:, None]
        + cfg.activation_beta * activation
        + cfg.usage_beta * usage
        + cfg.support_beta * support
    )
    tau_matrix = cfg.treatment_matrix()
    discount_pow = cfg.discount_factor ** (tenure_row - 1)

    # --- simulate each action against the same fixed draws ---
    outcomes: dict[str, ActionOutcome] = {}
    for a, action in enumerate(ACTIONS):
        (
            hazard,
            survival,
            churned,
            churn_period,
            months_active,
            expected_months,
            expected_discounted_months,
            survive_horizon_prob,
        ) = _simulate_action(
            base_logit, stage_codes, tau_matrix[a], churn_uniforms, discount_pow
        )
        outcomes[action] = ActionOutcome(
            hazard=hazard,
            survival=survival,
            churned=churned,
            churn_period=churn_period,
            months_active=months_active,
            expected_months=expected_months,
            expected_discounted_months=expected_discounted_months,
            survive_horizon_prob=survive_horizon_prob,
            ite_save=np.zeros(n),  # filled below relative to no_action
            delta_expected_months=np.zeros(n),
        )

    base = outcomes["no_action"]
    for action in ACTIONS:
        oc = outcomes[action]
        oc.ite_save = oc.survive_horizon_prob - base.survive_horizon_prob
        oc.delta_expected_months = oc.expected_months - base.expected_months

    # --- realized (control) panel ---
    realized_panel = _build_panel(
        static_df, activation, usage, support, stage_codes,
        base.churned, base.churn_period, base.months_active,
    )

    # --- randomized-assignment experiment panel (honest uplift training) ---
    assigned_idx = assign_rng.integers(0, len(ACTIONS), size=n)
    exp_churned = np.empty(n, dtype=bool)
    exp_period = np.empty(n, dtype=int)
    exp_months = np.empty(n, dtype=int)
    for a, action in enumerate(ACTIONS):
        sel = assigned_idx == a
        oc = outcomes[action]
        exp_churned[sel] = oc.churned[sel]
        exp_period[sel] = oc.churn_period[sel]
        exp_months[sel] = oc.months_active[sel]
    assigned_action = np.asarray(ACTIONS)[assigned_idx]
    experiment_panel = _build_panel(
        static_df, activation, usage, support, stage_codes,
        exp_churned, exp_period, exp_months, assigned_action=assigned_action,
    )

    # --- counterfactual lookup (ground truth) ---
    lookup_frames = []
    for action in ACTIONS:
        oc = outcomes[action]
        sampled_cp = pd.Series(oc.churn_period).astype("Int64").mask(~oc.churned, pd.NA)
        lookup_frames.append(
            pd.DataFrame(
                {
                    "customer_id": static_df["customer_id"].to_numpy(),
                    "action": action,
                    "expected_months": oc.expected_months,
                    "expected_discounted_months": oc.expected_discounted_months,
                    "survive_horizon_prob": oc.survive_horizon_prob,
                    "ite_save": oc.ite_save,
                    "delta_expected_months": oc.delta_expected_months,
                    "sampled_churn_period": sampled_cp,
                    "sampled_churned": oc.churned,
                    "sampled_months_active": oc.months_active,
                }
            )
        )
    counterfactual_lookup = pd.concat(lookup_frames, ignore_index=True)

    # --- survival curves (ground truth) ---
    curve_frames = []
    for action in ACTIONS:
        oc = outcomes[action]
        curve_frames.append(
            pd.DataFrame(
                {
                    "customer_id": np.repeat(static_df["customer_id"].to_numpy(), t_max),
                    "action": action,
                    "period": np.tile(tenure_row, n),
                    "hazard": oc.hazard.reshape(-1),
                    "survival_prob": oc.survival.reshape(-1),
                }
            )
        )
    survival_curves = pd.concat(curve_frames, ignore_index=True)

    return DGPArtifacts(
        config=cfg,
        static=static_df,
        realized_panel=realized_panel,
        experiment_panel=experiment_panel,
        counterfactual_lookup=counterfactual_lookup,
        survival_curves=survival_curves,
        outcomes=outcomes,
        churn_uniforms=churn_uniforms,
        stage_codes=stage_codes,
        base_logit=base_logit,
        assigned_action_index=assigned_idx,
        activation=activation,
        usage=usage,
        support=support,
    )


def empirical_hazard_by_tenure(panel: pd.DataFrame) -> pd.DataFrame:
    """Empirical discrete-time hazard per tenure: events / at-risk.

    Computed on whichever panel is passed; for the calibration check pass the
    realized (control) panel so the curve reflects the true baseline.
    """
    grouped = (
        panel.groupby("tenure")
        .agg(at_risk=("at_risk", "sum"), events=("event", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["at_risk"] > 0].copy()
    grouped["hazard"] = grouped["events"] / grouped["at_risk"]
    return grouped


# --------------------------------------------------------------------------- #
# file writing and CLI entry point
# --------------------------------------------------------------------------- #
def write_artifacts(artifacts: DGPArtifacts, out_dir: str | Path = "data") -> dict[str, Path]:
    """Write the four parquet artifacts, the calibration curve, and its plot."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "realized_panel": out / "realized_panel.parquet",
        "experiment_panel": out / "experiment_panel.parquet",
        "counterfactual_lookup": out / "counterfactual_lookup.parquet",
        "survival_curves": out / "survival_curves.parquet",
        "hazard_curve_csv": out / "hazard_by_tenure.csv",
        "hazard_curve_png": out / "hazard_by_tenure.png",
    }
    artifacts.realized_panel.to_parquet(paths["realized_panel"], index=False)
    artifacts.experiment_panel.to_parquet(paths["experiment_panel"], index=False)
    artifacts.counterfactual_lookup.to_parquet(paths["counterfactual_lookup"], index=False)
    artifacts.survival_curves.to_parquet(paths["survival_curves"], index=False)

    curve = empirical_hazard_by_tenure(artifacts.realized_panel)
    curve.to_csv(paths["hazard_curve_csv"], index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["tenure"], curve["hazard"], marker="o", color="#b5179e")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Empirical churn hazard")
    ax.set_title("Early-tenure churn hazard spike (realized / control population)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["hazard_curve_png"], dpi=120)
    plt.close(fig)
    return paths


def _print_summary(artifacts: DGPArtifacts) -> None:
    """Print the calibration curve, stage mix, and the headline DGP asymmetry."""
    curve = empirical_hazard_by_tenure(artifacts.realized_panel)
    print("\n=== Hazard by tenure (realized / control) ===")
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    realized_churn = artifacts.outcomes["no_action"].churned.mean()
    print(f"\nRealized churn within horizon: {realized_churn:.1%}")

    print("\n=== Stage mix (realized panel rows) ===")
    print(artifacts.realized_panel["stage"].value_counts(normalize=True).to_string())

    enters_attrition = (
        artifacts.stage_codes == STAGES.index("attrition_in_progress")
    ).any(axis=1)
    onb = artifacts.outcomes["onboarding_activation"]
    wb = artifacts.outcomes["winback_discount"]
    print("\n=== Headline DGP asymmetry (ground truth) ===")
    print(
        f"onboarding_activation: mean ITE_save = {onb.ite_save.mean():.4f}, "
        f"mean delta_expected_months (all customers) = {onb.delta_expected_months.mean():.3f}"
    )
    print(
        f"winback_discount:      mean ITE_save = {wb.ite_save[enters_attrition].mean():.4f}, "
        f"mean delta_expected_months (attrition entrants) = "
        f"{wb.delta_expected_months[enters_attrition].mean():.3f}"
    )


def main() -> None:
    """Generate the canonical population, write artifacts, print the calibration."""
    artifacts = generate_dataset(DEFAULT_CONFIG)
    paths = write_artifacts(artifacts, "data")
    _print_summary(artifacts)
    print("\nWrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
