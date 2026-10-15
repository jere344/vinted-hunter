# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-14

First public release.

### Added

- Local FastAPI web app that takes a search description and a budget, searches vinted.fr, and ranks listings by value.
- Server-sent events progress stream, saved hunts, and a single-page UI.
- Pipeline stages: plan, search, triage, details, vision extraction, specs research, and ranking.
- Hard requirements checked per listing, with a tri-state status (pass, fail, unknown).
- Web research of product specs and typical used prices, cached by normalised model.
- Shipping awareness: reads Vinted's default shipping price and ranks on landed cost.
- Verdict tiers and a present comparator for requirement checks.
- Two Claude backends: `claude-cli` (default, uses a local Claude Code login) and `api` (Anthropic API).
- Category-agnostic hunting: product-specific fields are declared by the planner rather than hard-coded.
- Test suite with fake Vinted, LLM, and CLI doubles. It needs no network access.

### Changed

- Specs research reuses caches, deduplicates identification calls, and skips listings that failed.
- Red flags are limited to the physical item and the transaction. Missing specifications are reported as gaps.
- Exact model codes are kept in cache keys, and only exact-variant specs can settle a resolution check.

### Fixed

- Defect caps, verdict tiers, comparator, product type, and new-price handling corrected after real-world runs.
