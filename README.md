# Lifecycle-Aware Retention Orchestrator

A multi-agent retention system that allocates churn-intervention spend by
customer lifecycle stage instead of by a single flat churn-propensity threshold.

## Premise

Intervention cost rises and success probability falls as a customer moves down
the lifecycle. A dollar spent on onboarding activation is worth more than a
dollar spent on attrition-stage win-back, because early intervention acts on a
population with full lifetime value still ahead of it and reduces the churn
hazard before any decision hardens. A single propensity score collapses this
distinction. This project conditions the intervention policy on lifecycle stage
and shows the difference.

## What it does

Each customer is routed by lifecycle stage to a stage-specific intervention
policy. A hazard model estimates churn risk and its drivers, an uplift model
estimates per-action incremental save probability, and an economic layer scores
each candidate action by expected value: probability of save times residual
lifetime value minus intervention cost, with win-back saves discounted for
re-churn risk.

The result is validated against a known data-generating process, so the
recovered policy is checked against ground truth rather than asserted.

## Headline result

On one synthetic population, the stage-conditional policy is compared against a
flat-threshold baseline on retained lifetime value per dollar spent. The gain is
decomposed by lifecycle stage to show it comes from reallocating spend toward
early-stage activation.

## Architecture

A LangGraph supervisor assigns lifecycle stage and routes to subagents:
diagnostic (hazard), intervention design (uplift), economic arbitration
(expected value), and a critic node enforcing budget and contact-frequency
constraints.

## Stack

Python, lifelines, scikit-learn, econml, LangGraph, LangChain, LangSmith,
Streamlit.

## Setup

```bash
bash setup.sh
```

## Progress

**Phase 1 — data-generating process (complete).** The project rests on a known
ground truth, so it begins by authoring the synthetic population rather than by
fitting a model. Customers follow a Telco-style covariate schema and churn in
discrete monthly periods (KKBox-style) under a logistic hazard whose baseline is
high during onboarding and decays with tenure. Treatment effects are
stage-conditional and encoded as ground truth: an onboarding-activation save is
worth far more than a late-stage win-back, both because it acts when the churn
hazard is highest and because a win-back save leaves usage collapsed and
re-churns quickly.

Under common random numbers, the generator emits:

- a realized (no-intervention) panel — the only data the hazard model sees;
- a randomized-assignment experiment — the honest training substrate for uplift,
  and the only data containing treatment variation;
- per-customer analytic counterfactuals (expected retained months, survival
  probability, true incremental save) plus full survival curves — held out as
  ground truth for validation and policy scoring, never a model input.

A calibration check confirms the early-tenure hazard spike (hazard falls from
~0.12 to ~0.01 over the first months), and the headline asymmetry is present in
the ground truth: an onboarding save buys ~2.9 expected retained months across
the base, a win-back save ~0.7 even among customers already in attrition.

Remaining: hazard model (2), uplift model (3), economic layer (4), policy
comparison (5), agentic wrapper (6), dashboard and writeup (7).
