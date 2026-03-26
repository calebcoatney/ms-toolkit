# Main Branch Cleanup Design

**Date:** 2026-03-26
**Last tag:** `v0.1.2`

## Goal

Clean up `main` to be production-ready by isolating experimental and school project work onto local-only branches. The public GitHub repo receives only the cleaned `main`.

## Context

Since `v0.1.2`, several experimental features were committed to `main` alongside a school project (`DSCI 575/`) that was never committed. KMeans preselection is the only preselector used in production; GMM+PCA work has never been used in an applied scenario. The hybrid search feature and ClassifierWrapper are unvalidated.

## What Is Production-Ready

- KMeans preselector (existing, unchanged since `v0.1.2`)
- Core search methods (`search_vector`, `search_w2v`)
- `.gitignore` improvements committed since `v0.1.2`

## What Is Experimental

| Item | Reason |
|---|---|
| `search_hybrid()` + helper methods in `api.py` | Unvalidated, untested in applied scenarios |
| GMM+PCA preselector additions in `preselector.py` | GMM never used in production |
| Random PCA sampling in `preselector.py` | Tied to GMM/PCA work |
| `ms_toolkit/classifiers.py` (untracked) | New, unvalidated ClassifierWrapper |
| `ms_toolkit/__init__.py` changes (untracked) | Exports ClassifierWrapper — only needed with classifiers.py |

## Branch Structure

### `experimental/hybrid-search` (local only, never pushed)
- Created from current `main` HEAD
- Untracked `classifiers.py` and `__init__.py` changes committed here
- Preserves: hybrid search, GMM+PCA preselector work, ClassifierWrapper
- Future use: validate and integrate into a production release when ready

### `school/dsci-575` (local only, never pushed)
- Created from current `main` HEAD
- Untracked `DSCI 575/` directory committed here
- Preserves: carbon number classifier, PIONA classifier, MW regressor, hydrocarbon classifier, Word2Vec fine-tuning notebooks, utility scripts
- Future use: reference for classifier-based preselector development

### `main` (pushed to public GitHub)
- Reverts the three experimental commits (hybrid search, GMM+PCA fix, random PCA sampling)
- Retains the `.gitignore` commit from `2930dcf`
- Adds `.gitignore` entries for: `DSCI 575/`, `ms_toolkit/classifiers.py`, `models/*.npy`
- Result: equivalent to `v0.1.2` + improved `.gitignore`

## Commits to Revert on `main`

| Commit | Message | Action |
|---|---|---|
| `775dec6` | Added hybrid search function to API | Revert |
| `8be9a7e` | feat: Add hybrid search and fix GMM+PCA preselector | Revert |
| `c91af56` | fix: implemented random sampling for PCA for large datasets | Revert |
| `2930dcf` | Updated .gitignore | Keep |

## `.gitignore` Additions on `main`

```
DSCI 575/
ms_toolkit/classifiers.py
models/*.npy
```
