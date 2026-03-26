# Main Branch Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate experimental and school project work onto local-only branches, restoring `main` to a production-ready state equivalent to `v0.1.2` plus `.gitignore` improvements.

**Architecture:** Create two local branches from the current `main` HEAD (preserving all work), then restore `main`'s tracked files to their `v0.1.2` state and update `.gitignore`. The public GitHub repo receives only the cleaned `main`.

**Tech Stack:** Git

---

## File Map

| File | Action on `main` |
|---|---|
| `ms_toolkit/api.py` | Restored to `v0.1.2` state (removes `search_hybrid` and helpers) |
| `ms_toolkit/preselector.py` | Restored to `v0.1.2` state (removes GMM+PCA utilities and random PCA sampling) |
| `ms_toolkit/__init__.py` | Restored to `v0.1.2` state (removes `ClassifierWrapper` exports) |
| `.gitignore` | Updated — add `DSCI 575/`, `ms_toolkit/classifiers.py`, `models/*.npy` |
| `ms_toolkit/classifiers.py` | Committed to `experimental/hybrid-search` only; gitignored on `main` |
| `DSCI 575/` | Committed to `school/dsci-575` only; gitignored on `main` |
| `models/*.npy` | Gitignored on `main` (already untracked) |

---

## Task 1: Create `experimental/hybrid-search` branch and commit experimental work

**Files:**
- Create branch: `experimental/hybrid-search`
- Commit: `ms_toolkit/classifiers.py`, `ms_toolkit/__init__.py`, `ms_toolkit/api.py`, `ms_toolkit/preselector.py`

- [ ] **Step 1: Verify current working tree state**

```bash
git status --short
```

Expected output includes:
```
 M ms_toolkit/__init__.py
 M ms_toolkit/api.py
 M ms_toolkit/preselector.py
?? ms_toolkit/classifiers.py
?? "DSCI 575/"
?? models/massbank_25epochs.model.syn1neg.npy
?? models/massbank_25epochs.model.wv.vectors.npy
```

- [ ] **Step 2: Create and switch to experimental branch**

```bash
git checkout -b experimental/hybrid-search
```

Expected: `Switched to a new branch 'experimental/hybrid-search'`

This branch is created at current `main` HEAD, which already contains the committed hybrid search and GMM+PCA work (commits `775dec6`, `8be9a7e`, `c91af56`). The working tree modifications stay in place.

- [ ] **Step 3: Stage the experimental working tree changes**

```bash
git add ms_toolkit/classifiers.py ms_toolkit/__init__.py ms_toolkit/api.py ms_toolkit/preselector.py
```

- [ ] **Step 4: Verify staged files**

```bash
git diff --staged --stat
```

Expected: shows `classifiers.py` (new), `__init__.py`, `api.py`, `preselector.py` staged.

- [ ] **Step 5: Commit the experimental working tree changes**

```bash
git commit -m "feat: add ClassifierPreselector integration and ClassifierWrapper

- Add ClassifierWrapper unified interface for sklearn/PyTorch/XGBoost
- Export ClassifierPreselector and ClassifierWrapper from package
- Wire ClassifierPreselector into search_vector and search_w2v methods"
```

- [ ] **Step 6: Verify branch state**

```bash
git log --oneline -6
```

Expected: the new commit on top, followed by the existing experimental commits (`c91af56`, `8be9a7e`, `775dec6`), then `v0.1.2`-era commits.

---

## Task 2: Create `school/dsci-575` branch and commit school project files

**Files:**
- Create branch: `school/dsci-575`
- Commit: `DSCI 575/` directory

- [ ] **Step 1: Create school branch from current main (not from experimental)**

```bash
git checkout main
git checkout -b school/dsci-575
```

Expected: `Switched to a new branch 'school/dsci-575'`

Note: `DSCI 575/` and the `models/*.npy` files are untracked — they remain in the working directory across branch switches.

- [ ] **Step 2: Stage the school project directory**

```bash
git add "DSCI 575/"
```

- [ ] **Step 3: Verify what will be committed**

```bash
git diff --staged --stat
```

Expected: lists all files under `DSCI 575/` — notebooks (`.ipynb`), Python files (`oxygenates.py`, `piona.py`, `utils.py`), data file (`DataForPLSAug2022-renormalized.xlsx`), docs (`project-proposal.md`, `progress-report-outline.md`).

- [ ] **Step 4: Commit school project files**

```bash
git commit -m "school: add DSCI 575 project — molecular property prediction from spectral embeddings

Classifiers trained on Word2Vec embeddings for:
- Carbon number (C4-C16, SVC, 96.6% accuracy)
- Hydrocarbon classification (binary, neural net, 99% accuracy)
- Molecular weight regression (linear, R²=0.97)
- PIONA classification (5-class SVC, 89.9% accuracy)

Local-only branch. Not intended for public release."
```

- [ ] **Step 5: Verify branch has school files and is separate from experimental**

```bash
git log --oneline -3
git diff main --stat | head -20
```

Expected: the commit above, on top of `main` HEAD. Diff shows only `DSCI 575/` files.

---

## Task 3: Restore `main` to `v0.1.2` state for experimental files

**Files:**
- Modify: `ms_toolkit/api.py` (restore to `v0.1.2`)
- Modify: `ms_toolkit/preselector.py` (restore to `v0.1.2`)
- Modify: `ms_toolkit/__init__.py` (restore to `v0.1.2` — empty file)

- [ ] **Step 1: Switch back to main**

```bash
git checkout main
```

Expected: `Switched to branch 'main'`

The working tree modifications to `api.py`, `preselector.py`, `__init__.py` are gone (they were committed to `experimental/hybrid-search`). Untracked files (`DSCI 575/`, `models/*.npy`, `classifiers.py`) remain.

- [ ] **Step 2: Verify current state of main**

```bash
git log --oneline -5
```

Expected: shows the 5 commits since `v0.1.2` (merge + gitignore + c91af56 + 8be9a7e + 775dec6), plus the design doc commit.

- [ ] **Step 3: Restore the three experimental files to their `v0.1.2` state**

```bash
git restore --source=v0.1.2 ms_toolkit/api.py ms_toolkit/preselector.py ms_toolkit/__init__.py
```

This does not create a commit — it updates the working tree and stages the changes.

- [ ] **Step 4: Verify what was restored**

```bash
git diff --staged --stat
```

Expected: `api.py`, `preselector.py`, `__init__.py` shown as modified with line removals.

```bash
git diff --staged -- ms_toolkit/api.py | grep "^+.*def search_hybrid" | wc -l
```

Expected: `0` (the `search_hybrid` method is gone from the staged version)

```bash
git diff --staged -- ms_toolkit/__init__.py | grep "ClassifierWrapper" | wc -l
```

Expected: `0` (no ClassifierWrapper exports in staged version)

- [ ] **Step 5: Commit the restoration**

```bash
git commit -m "revert: restore api.py, preselector.py, __init__.py to v0.1.2 state

Moves experimental work (hybrid search, GMM+PCA preselector, ClassifierWrapper)
to local-only branch experimental/hybrid-search.
Main branch reflects production-ready v0.1.2 feature set."
```

---

## Task 4: Update `.gitignore` on `main`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check current `.gitignore` content**

```bash
cat .gitignore
```

Verify it does not already contain `DSCI 575/`, `ms_toolkit/classifiers.py`, or `models/*.npy`.

- [ ] **Step 2: Append the new entries**

Add to the end of `.gitignore`:

```
# Experimental / local-only
DSCI 575/
ms_toolkit/classifiers.py
models/*.npy
```

- [ ] **Step 3: Verify the untracked files are now ignored**

```bash
git status --short
```

Expected: `DSCI 575/`, `ms_toolkit/classifiers.py`, `models/massbank_25epochs.model.syn1neg.npy`, and `models/massbank_25epochs.model.wv.vectors.npy` should no longer appear as `??`. Only `.gitignore` itself should appear as modified.

- [ ] **Step 4: Commit the gitignore update**

```bash
git add .gitignore
git commit -m "chore: gitignore experimental files and school project directory"
```

---

## Task 5: Verify clean state and push `main`

- [ ] **Step 1: Confirm `main` is clean**

```bash
git status --short
```

Expected: only `models/massbank_25epochs.model` may appear as ` M` (this tracked binary file is out of scope for this cleanup — handle separately). No `??` untracked files should appear.

- [ ] **Step 2: Confirm experimental branches exist locally**

```bash
git branch
```

Expected output includes:
```
  experimental/hybrid-search
* main
  school/dsci-575
```

- [ ] **Step 3: Confirm experimental branches are NOT set to track remote**

```bash
git branch -vv
```

Expected: `experimental/hybrid-search` and `school/dsci-575` show no `[origin/...]` tracking info.

- [ ] **Step 4: Review the final `main` log**

```bash
git log --oneline v0.1.2..HEAD
```

Expected: shows only the design doc commit, the `.gitignore` restoration/update commits — no hybrid search or GMM/PCA commits.

- [ ] **Step 5: Push `main` to GitHub**

```bash
git push origin main
```

Expected: push succeeds. Verify on GitHub that `main` does not contain `search_hybrid`, `ClassifierWrapper`, or the `DSCI 575/` directory.

---

## Out of Scope (Separate Decision Needed)

`models/massbank_25epochs.model` is a tracked binary file showing as modified in the working tree. This was not part of the cleanup scope. Decide separately whether to:
- Commit the updated model
- Revert it to the committed version (`git restore models/massbank_25epochs.model`)
- Move it to `.gitignore` and use Git LFS or external storage
