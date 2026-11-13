# Synthetic window-block vectors

`python -m clean.rr_f.tests.gen_vectors [--out DIR] [--seed N]` writes these vectors from seeded random numbers
(default seed 51; the same seed writes the same bytes), with the clean team's own NumPy reference computing the
expected outputs; `manifest.json` and `SHA256SUMS` describe them. Nothing in them comes from a model file, learned
weights or a game. The generated binaries are not committed.
