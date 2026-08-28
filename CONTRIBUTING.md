# Contributing

Keep contributions focused on representation probing rather than model
training or planning experiments. A new environment should include:

1. an end-to-end notebook with a plain-language walkthrough;
2. grouped trajectory splits and at least one spatial-transfer check;
3. shuffled-label and state-shortcut controls;
4. hashes and provenance for every external artifact;
5. CPU-only tests for target construction and split integrity;
6. a compact result summary without large generated caches in Git.

Run `python -m pytest -q tests` before opening a change. Do not commit model
weights, trajectory archives, activation caches, credentials, or local paths.
