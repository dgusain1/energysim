# Change fragments

Every user-visible change adds one small file here instead of editing `CHANGELOG.txt` (no merge conflicts).

File name: `<topic-or-pr>.<type>.md`, e.g. `ensemble-sweeps.added.md`.
Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `docs`.
Content: one or more Markdown bullet lines (a bare sentence becomes a bullet).

At release time the maintainer runs

    python scripts/build_changelog.py --version X.Y.Z          # merge into CHANGELOG.txt, delete fragments
    python scripts/build_changelog.py --version X.Y.Z --draft  # preview only
