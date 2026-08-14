# Experiment artifacts

Each subdirectory is an immutable experiment bundle identified by its Slurm job
ID or another unique run ID. New experiments must use a new directory; do not
overwrite an earlier bundle with corrected or extended measurements.

A complete bundle should contain:

- raw per-condition result JSON and manifests;
- append-only full request/response ledgers;
- vLLM and Slurm logs;
- derived CSV tables, plots, provenance, and a Markdown summary;
- a bundle README describing the configuration and known limitations.

Prompts and responses are intentionally retained for research reproducibility.
Only synthetic or public benchmark data may be committed. Credentials and API
keys must never be recorded. Large future bundles should keep their curated
analysis and provenance here while placing oversized raw artifacts in a
versioned release or other durable research-data archive.

## Curated bundles

- `block-dataset-v1/`: Qwen-tokenized synthetic candidate-block labels with
  family-level splits and varied dependency positions.
- `quality-273641/`: first answer-sensitive CacheSelect speed-quality matrix,
  covering three prompt lengths, three repair radii, and three edit positions.
- `quality-274080/`: seven-family quality-boundary matrix that identifies
  pointer dependencies as a failure case for distance-only repair selection.
