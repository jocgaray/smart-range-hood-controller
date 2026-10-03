# Smart Range Hood Controller: ML Onset Detection & Physics-Based MPC Optimization (Sanitized Architecture Showcase)

> **Project Scope & Intellectual Property Notice**

> This repository contains selected, IP-safe excerpts from a production-oriented smart range hood control project. Proprietary physics models, controller tuning, feature engineering, trained artifacts, raw telemetry, and detailed LLM prompts have been omitted. The published code is intended to demonstrate architecture, interfaces, and engineering practices rather than provide a complete reproduction of the production system.

## 🏛 Architecture & Engineering Highlights

This repository serves as a showcase for production-ready software engineering practices, clean system boundaries, and robust MLOps orchestration. Core numeric parameters, private calibration data, and proprietary model weights have been intentionally sanitized or stubbed.

* **Structured LLM Orchestration & Deterministic Safety:** Integrates LangChain with Pydantic v2 schemas (`ParsedScenarioParams`) to enforce strict output typing from natural-language prompts. Raw GenAI outputs undergo deterministic parameter reconciliation before reaching downstream engines to guarantee mathematical runtime consistency.
* **Modular Clean Architecture:** Strict separation of concerns between presentation (`frontend/`), REST API orchestration (`api/`), execution engines (`src/`), and training/data pipelines (`pipelines/`). Logic is decoupled via interfaces and dependency inversion so individual services can be modified or stubbed without side effects.
* **Production MLOps Pipeline:** Utilizes Data Version Control (DVC) for reproducible data lineage and pipeline execution (`dvc.yml`), paired with Optuna for structured hyperparameter optimization (`pipelines/train_optuna.py`).
* **Containerization & Multi-Environment Setup:** Targeted, lightweight Dockerfiles (`Dockerfile.api`, `Dockerfile.frontend`) and tiered Docker Compose configurations (`docker-compose.yml` vs. `docker-compose.prod.yml`) ensure consistent execution across local development and production environments.
* **Defensive Engineering & Layered QA:** Includes automated CI/CD GitHub Actions workflows covering linting, type safety, and multi-tier Pytest coverage (`tests/unit/` for isolated domain logic and `tests/integration/` for end-to-end API and feature pipeline validation).