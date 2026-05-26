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

## Status

In development.
