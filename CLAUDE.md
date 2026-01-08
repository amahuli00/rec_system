# CLAUDE.md

## Project Goal

This project implements an industry-style end-to-end recommendation system while prioritizing learning and understanding.

Goals:
- Build a realistic recommendation system (candidate generation + ranking)
- Learn and refresh ML fundamentals and ML infrastructure concepts
- Favor clarity, correctness, and understanding over optimization

This is my first ML-heavy project in a while. Explanations, explicit assumptions, and step-by-step reasoning are encouraged.

---

## High-Level Architecture

User / Context
→ Candidate Generation (Two-Tower Model)
→ ANN Retrieval (FAISS-like)
→ Ranking (GBDT with feature engineering)
→ Final Ranked Results
→ Logging → Training Data Feedback Loop

Each stage should be modular, testable, and independently versioned.

---

## System Components

### Candidate Generation

Purpose:
- Reduce the full item corpus to a small set of relevant candidates

Approach:
- Two-Tower neural network (user tower + item tower)
- Trained on implicit feedback (MovieLens)

Substeps:
- Data preparation
- Model definition
- Training loop
- Offline evaluation (Recall@K)
- Embedding materialization
- ANN indexing

Constraints:
- Prioritize correctness and understanding
- Offline quality first, online serving later
- Simplicity over state-of-the-art performance

---

### Ranking

Purpose:
- Precisely order candidate items for a user

Approach:
- Gradient-boosted decision trees (XGBoost or LightGBM)
- Explicit feature engineering

Features:
- User features
- Item features
- User–item interaction features
- Optional context features

Substeps:
- Feature definition and schemas
- Offline feature materialization
- Training
- Offline evaluation (NDCG, MAP)
- Online-style inference service

---

### Feature Engineering & Data Pipelines

Principles:
- Explicit feature schemas
- Clear feature ownership
- Offline–online parity where possible
- Graceful handling of missing or stale features

Artifacts:
- Feature builders
- Feature schemas
- Versioned feature sets

---

### Serving & ML Infrastructure (Simulated)

Goals:
- Demonstrate ML infrastructure thinking rather than full production deployment

Concepts to model:
- Latency budgets (p95)
- Model versioning and rollback
- Shadow traffic simulation
- Canary deployments
- Feature drift detection (basic statistics)

---

## Development Workflow

### Phase 1: Exploration (Notebooks)

Purpose:
- Rapid iteration and learning

Rules:
- Use one notebook per major phase
- Notebooks may contain exploratory and imperfect code
- Notebooks are not the final artifact

Allowed:
- Data inspection
- Feature experimentation
- Metric exploration
- Debugging

Disallowed:
- Hidden business logic
- Notebooks as the only execution path

---

### Phase 2: Productization (Python Modules)

Purpose:
- Convert stabilized logic into production-style code

Rules:
- Extract logic into Python classes and modules
- Scripts and modules are the source of truth
- Notebooks may only call module APIs

Outcome:
- Clean, testable, reusable components
- Clear interfaces between system stages

---

## Expectations for Claude Code

When assisting on this project:
- Assume strong software engineering and infrastructure background
- Do not assume deep familiarity with modern ML tooling
- Prefer clarity over cleverness
- Explain why decisions are made, not just how
- Help refactor notebook code into clean, reusable modules
- Call out tradeoffs and common pitfalls

---

## Non-Goals

- State-of-the-art model performance
- Large-scale distributed training
- Managed cloud services
- Frontend or UI components

---

## Definition of Success

This project is successful if:
- The system runs end-to-end locally
- Offline metrics are computed correctly
- Each component can be explained clearly
- The architecture mirrors real-world systems
- Design tradeoffs can be articulated confidently

---

Guiding principle:
Build it simple. Make it correct. Understand every part. Then improve.


## How to Ask Claude for Changes

Use the following patterns when requesting help. These are preferred and expected.

### Preferred Request Style

- Specify the **component** (ranking, candidate generation, features, serving)
- Specify the **phase** (exploration vs productization)
- Specify whether the goal is **learning**, **correctness**, or **cleanup**

Example:
"Help me explore ranking feature ideas in a notebook. This is Phase 1 and learning-focused."

---

### Common Tasks and How to Request Them

**Exploration**
"Help me explore features for the ranking model in a notebook. Prioritize clarity and explanation."

**Training**
"Implement a simple, correct training loop for the ranking model. Avoid premature optimization."

**Refactoring**
"Refactor this notebook code into a reusable Python class suitable for production."

**Evaluation**
"Help me add offline evaluation metrics and explain what they mean."

**Infra Concepts**
"Help me simulate model versioning and rollback in a simple, local setup."

---

### Expectations When Responding

Claude should:
- Explain assumptions explicitly
- Describe tradeoffs
- Prefer simple baselines first
- Avoid unnecessary abstraction
- Flag common ML and ML-infra pitfalls

Claude should not:
- Over-optimize prematurely
- Skip explanations
- Introduce complex frameworks without justification
- Assume prior ML expertise

---

### Default Assumptions (Unless Stated Otherwise)

- Local development only
- Single-machine execution
- Small to medium datasets (MovieLens)
- Python-first implementation
- Emphasis on understandability over performance
