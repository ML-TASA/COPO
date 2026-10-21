# Upstream provenance

The `verl/` directory is a modified vendored snapshot of the [verl project](https://github.com/verl-project/verl), distributed under the Apache License 2.0. The vendored package reports version `0.8.0.dev`.

Revision `03396b02bc3888bb58b63a1a3d4463bd6f0e7ff3` is retained as a comparison reference. The public vendored tree is not a byte-for-byte copy of that revision: it contains upstream files from a different snapshot together with COPO additions and intentional modifications. No exact single upstream base revision is claimed.

COPO-specific work includes the causal-reward and fixed-trajectory utilities, model integrations, and the required trainer, rollout, and compatibility changes. Upstream copyright headers, `LICENSE`, and `Notice.txt` are retained.

The upstream `recipe/` directory is an optional Git submodule from [verl-recipe](https://github.com/verl-project/verl-recipe). It is not required by the COPO collection, training, or test entry points and is not included in this standalone vendored source distribution.
