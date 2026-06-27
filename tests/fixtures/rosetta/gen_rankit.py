# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for rankit IR evaluation metrics.

Provenance for rankit_ir.json.

nDCG: rankit uses LINEAR gain with a log2(rank+1) discount (confirmed by its own
unit test), so the matching oracle is sklearn.metrics.ndcg_score (linear gain),
NOT pytrec_eval / trec_eval (which use 2^rel-1 gain). Both the binary and graded
rankit nDCG paths use linear gain, so sklearn covers both. The ranking is encoded
for sklearn by giving y_score strictly descending in document order, so the
sklearn ranking equals rankit's `ranked` order.

MAP / MRR / precision@k / recall@k are unambiguous IR formulas with no
scikit-learn function (sklearn's average_precision_score is PR-curve based, a
different quantity), so their reference is the canonical formula computed in numpy
(cross-implementation check).

Regenerate: uv run tests/fixtures/rosetta/gen_rankit.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import ndcg_score

n = 12
relevant = [0, 2, 5, 8]  # binary relevant document indices
grades = [3, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0]  # graded relevance per document
# Documents are evaluated in index order; a strictly-descending score makes the
# sklearn ranking equal to that document order.
y_score = list(range(n, 0, -1))

bin_true = [1 if i in relevant else 0 for i in range(n)]

# IR formulas (numpy) for binary relevance, ranks are 1-indexed.
relevant_ranks = [i + 1 for i in range(n) if i in relevant]
r = len(relevant_ranks)
ap = float(np.mean([(j + 1) / rank for j, rank in enumerate(relevant_ranks)]))
mrr = 1.0 / min(relevant_ranks)
p_at_5 = sum(1 for rk in relevant_ranks if rk <= 5) / 5.0
r_at_5 = sum(1 for rk in relevant_ranks if rk <= 5) / r

expected = {
    "ndcg_bin_5": float(ndcg_score([bin_true], [y_score], k=5)),
    "ndcg_bin_10": float(ndcg_score([bin_true], [y_score], k=10)),
    "ndcg_graded_5": float(ndcg_score([grades], [y_score], k=5)),
    "ndcg_graded_10": float(ndcg_score([grades], [y_score], k=10)),
    "map": ap,
    "mrr": float(mrr),
    "precision_at_5": p_at_5,
    "recall_at_5": r_at_5,
}

fixture = {
    "provenance": {
        "generator": "gen_rankit.py",
        "library": "scikit-learn (ndcg) + numpy (ir formulas)",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "note": "linear-gain nDCG vs sklearn.ndcg_score; map/mrr/p@k/r@k vs numpy.",
    },
    "n": n,
    "relevant": relevant,
    "grades": grades,
    "expected": expected,
}

out = Path(__file__).parent / "rankit_ir.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
for key, val in expected.items():
    print(f"{key:16s} {val:.10f}")
print(f"wrote {out}")
