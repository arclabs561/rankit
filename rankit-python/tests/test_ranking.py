"""Tests for ranklab Python bindings."""

import numpy as np
import pytest
import ranklab


class TestSoftRank:
    def test_soft_rank_basic(self):
        scores = [5.0, 1.0, 2.0, 4.0, 3.0]
        ranks = ranklab.soft_rank(scores, temperature=10.0)
        assert len(ranks) == 5
        # Ordering: 1.0 < 2.0 < 3.0 < 4.0 < 5.0
        assert ranks[1] < ranks[2]  # 1.0 < 2.0
        assert ranks[2] < ranks[4]  # 2.0 < 3.0
        assert ranks[4] < ranks[3]  # 3.0 < 4.0
        assert ranks[3] < ranks[0]  # 4.0 < 5.0

    def test_soft_rank_empty(self):
        ranks = ranklab.soft_rank([], temperature=1.0)
        assert len(ranks) == 0

    def test_soft_rank_single(self):
        ranks = ranklab.soft_rank([42.0], temperature=1.0)
        assert len(ranks) == 1
        assert float(ranks[0]) == 0.0

    def test_soft_rank_returns_numpy(self):
        ranks = ranklab.soft_rank([3.0, 1.0, 2.0])
        assert isinstance(ranks, np.ndarray)
        assert ranks.dtype == np.float64

    def test_soft_rank_numpy_input(self):
        scores = np.array([5.0, 1.0, 2.0, 4.0, 3.0])
        ranks = ranklab.soft_rank(scores, temperature=10.0)
        assert isinstance(ranks, np.ndarray)
        assert len(ranks) == 5
        assert ranks[1] < ranks[2]

    def test_soft_rank_numpy_f32_input(self):
        scores = np.array([5.0, 1.0, 2.0], dtype=np.float32)
        ranks = ranklab.soft_rank(scores, temperature=10.0)
        assert isinstance(ranks, np.ndarray)
        assert len(ranks) == 3


class TestSoftRankVariants:
    def test_neural_sort_numpy(self):
        scores = np.array([3.0, 1.0, 2.0])
        ranks = ranklab.soft_rank_neural_sort(scores, temperature=1.0)
        assert isinstance(ranks, np.ndarray)
        assert len(ranks) == 3

    def test_sigmoid_numpy(self):
        scores = np.array([3.0, 1.0, 2.0])
        ranks = ranklab.soft_rank_sigmoid(scores, temperature=1.0)
        assert isinstance(ranks, np.ndarray)
        assert len(ranks) == 3

    def test_smooth_i_numpy(self):
        scores = np.array([3.0, 1.0, 2.0])
        ranks = ranklab.soft_rank_smooth_i(scores, temperature=1.0)
        assert isinstance(ranks, np.ndarray)
        assert len(ranks) == 3


class TestLosses:
    def test_ranknet_loss_perfect(self):
        predictions = [0.9, 0.5, 0.1]
        relevance = [2.0, 1.0, 0.0]
        loss = ranklab.ranknet_loss(predictions, relevance)
        assert loss >= 0.0
        assert loss < 1.0

    def test_ranknet_loss_reversed(self):
        predictions = [0.1, 0.5, 0.9]
        relevance = [2.0, 1.0, 0.0]
        loss = ranklab.ranknet_loss(predictions, relevance)
        assert loss > 0.5

    def test_ranknet_loss_comparison(self):
        relevance = [2.0, 1.0, 0.0]
        good = ranklab.ranknet_loss([0.9, 0.5, 0.1], relevance)
        bad = ranklab.ranknet_loss([0.1, 0.5, 0.9], relevance)
        assert good < bad

    def test_ranknet_loss_numpy(self):
        predictions = np.array([0.9, 0.5, 0.1])
        relevance = np.array([2.0, 1.0, 0.0])
        loss = ranklab.ranknet_loss(predictions, relevance)
        assert isinstance(loss, float)
        assert loss >= 0.0
        assert loss < 1.0

    def test_approx_ndcg_perfect(self):
        predictions = [0.9, 0.5, 0.1]
        relevance = [2.0, 1.0, 0.0]
        score = ranklab.approx_ndcg(predictions, relevance, temperature=1.0)
        assert score > 0.5

    def test_approx_ndcg_numpy(self):
        predictions = np.array([0.9, 0.5, 0.1])
        relevance = np.array([2.0, 1.0, 0.0])
        score = ranklab.approx_ndcg(predictions, relevance, temperature=1.0)
        assert isinstance(score, float)
        assert score > 0.5

    def test_lambda_loss(self):
        predictions = [0.8, 0.3, 0.6]
        relevance = [2.0, 0.0, 1.0]
        loss = ranklab.lambda_loss(predictions, relevance)
        assert loss >= 0.0
        assert loss < 10.0

    def test_listnet_loss(self):
        predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
        relevance = [0.0, 1.0, 0.2, 0.8, 0.4]
        loss = ranklab.listnet_loss(predictions, relevance, temperature=1.0)
        assert loss >= 0.0

    def test_listmle_loss(self):
        predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
        relevance = [0.0, 1.0, 0.2, 0.8, 0.4]
        loss = ranklab.listmle_loss(predictions, relevance, temperature=1.0)
        assert loss >= 0.0

    def test_ranknet_loss_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.ranknet_loss([1, 2, 3], [1, 2])

    def test_approx_ndcg_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.approx_ndcg([1, 2, 3], [1, 2])

    def test_lambda_loss_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.lambda_loss([1, 2, 3], [1, 2])

    def test_listnet_loss_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.listnet_loss([1, 2, 3], [1, 2])

    def test_listmle_loss_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.listmle_loss([1, 2, 3], [1, 2])

    def test_losses_mixed_numpy_list(self):
        """Numpy predictions with list relevance (and vice versa)."""
        preds_np = np.array([0.9, 0.5, 0.1])
        rel_list = [2.0, 1.0, 0.0]
        loss1 = ranklab.ranknet_loss(preds_np, rel_list)
        loss2 = ranklab.ranknet_loss([0.9, 0.5, 0.1], np.array([2.0, 1.0, 0.0]))
        assert abs(loss1 - loss2) < 1e-12


class TestGradients:
    def test_lambdarank_gradients(self):
        scores = [0.5, 0.8, 0.3]
        relevance = [3.0, 1.0, 2.0]
        grads = ranklab.compute_lambdarank_gradients(scores, relevance)
        assert len(grads) == 3
        assert any(g != 0.0 for g in grads)

    def test_lambdarank_gradients_returns_numpy(self):
        grads = ranklab.compute_lambdarank_gradients([0.5, 0.8, 0.3], [3.0, 1.0, 2.0])
        assert isinstance(grads, np.ndarray)
        assert grads.dtype == np.float32

    def test_lambdarank_gradients_numpy_input(self):
        scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        relevance = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        grads = ranklab.compute_lambdarank_gradients(scores, relevance)
        assert isinstance(grads, np.ndarray)
        assert len(grads) == 3

    def test_lambdarank_gradients_numpy_f64_input(self):
        """f64 numpy arrays should be auto-converted to f32."""
        scores = np.array([0.5, 0.8, 0.3], dtype=np.float64)
        relevance = np.array([3.0, 1.0, 2.0], dtype=np.float64)
        grads = ranklab.compute_lambdarank_gradients(scores, relevance)
        assert isinstance(grads, np.ndarray)
        assert grads.dtype == np.float32

    def test_ranking_svm_gradients(self):
        scores = [0.5, 0.8, 0.3]
        relevance = [3.0, 1.0, 2.0]
        grads = ranklab.compute_ranking_svm_gradients(scores, relevance)
        assert len(grads) == 3

    def test_lambdarank_gradients_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.compute_lambdarank_gradients([1, 2, 3], [1, 2])

    def test_ranking_svm_gradients_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.compute_ranking_svm_gradients([1, 2, 3], [1, 2])

    def test_ranking_svm_gradients_numpy(self):
        scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        relevance = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        grads = ranklab.compute_ranking_svm_gradients(scores, relevance)
        assert isinstance(grads, np.ndarray)
        assert grads.dtype == np.float32


class TestEvalMetrics:
    def test_ndcg_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc1": 2, "doc2": 1, "doc3": 0}
        score = ranklab.ndcg(ranked, qrels, k=3)
        assert 0.0 < score <= 1.0

    def test_ndcg_perfect(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8)]
        qrels = {"doc1": 2, "doc2": 1}
        score = ranklab.ndcg(ranked, qrels, k=2)
        assert score > 0.9

    def test_map_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc1": 2, "doc2": 1, "doc3": 0}
        score = ranklab.map_score(ranked, qrels)
        assert score > 0.9

    def test_mrr_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc2": 1, "doc3": 0}
        score = ranklab.mrr(ranked, qrels)
        assert abs(score - 0.5) < 1e-9

    def test_precision_at_k_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc1": 1, "doc3": 1}
        score = ranklab.precision_at_k(ranked, qrels, k=2)
        assert abs(score - 0.5) < 1e-9

    def test_recall_at_k_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc1": 1, "doc3": 1}
        score = ranklab.recall_at_k(ranked, qrels, k=3)
        assert abs(score - 1.0) < 1e-9


class TestEdgeCases:
    def test_empty_inputs(self):
        result = ranklab.soft_rank([], temperature=1.0)
        assert len(result) == 0
        assert ranklab.ranknet_loss([], []) == 0.0

    def test_single_element(self):
        ranks = ranklab.soft_rank([1.0], temperature=1.0)
        assert len(ranks) == 1
        assert float(ranks[0]) == 0.0

    def test_differentiable_topk(self):
        scores = [5.0, 1.0, 2.0, 4.0, 3.0]
        values, ranks = ranklab.differentiable_topk(scores, k=3, temperature=1.0)
        assert len(values) == 5
        assert len(ranks) == 5

    def test_differentiable_topk_returns_numpy(self):
        values, indicators = ranklab.differentiable_topk([5.0, 1.0, 2.0], k=2)
        assert isinstance(values, np.ndarray)
        assert isinstance(indicators, np.ndarray)
        assert values.dtype == np.float64

    def test_differentiable_topk_numpy_input(self):
        scores = np.array([5.0, 1.0, 2.0, 4.0, 3.0])
        values, indicators = ranklab.differentiable_topk(scores, k=3, temperature=1.0)
        assert isinstance(values, np.ndarray)
        assert len(values) == 5

    def test_empty_numpy_input(self):
        scores = np.array([], dtype=np.float64)
        ranks = ranklab.soft_rank(scores, temperature=1.0)
        assert isinstance(ranks, np.ndarray)
        assert len(ranks) == 0


class TestAdditionalBinaryMetrics:
    """Tests for ERR, RBP, F-measure, Success@k, R-Precision, Average Precision."""

    def _make_data(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6), ("doc5", 0.5)]
        qrels = {"doc1": 1, "doc3": 1, "doc5": 1}
        return ranked, qrels

    def test_err_at_k(self):
        ranked, qrels = self._make_data()
        score = ranklab.err_at_k(ranked, qrels, k=5)
        assert 0.0 <= score <= 1.0

    def test_err_at_k_first_relevant(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8)]
        qrels = {"doc1": 1}
        score = ranklab.err_at_k(ranked, qrels, k=2)
        assert score > 0.0

    def test_rbp_at_k(self):
        ranked, qrels = self._make_data()
        score = ranklab.rbp_at_k(ranked, qrels, k=5)
        assert 0.0 <= score <= 1.0

    def test_rbp_at_k_custom_persistence(self):
        ranked, qrels = self._make_data()
        s1 = ranklab.rbp_at_k(ranked, qrels, k=5, persistence=0.5)
        s2 = ranklab.rbp_at_k(ranked, qrels, k=5, persistence=0.99)
        # Higher persistence => later docs matter more
        assert isinstance(s1, float)
        assert isinstance(s2, float)

    def test_f_measure_at_k(self):
        ranked, qrels = self._make_data()
        f1 = ranklab.f_measure_at_k(ranked, qrels, k=5)
        assert 0.0 <= f1 <= 1.0

    def test_f_measure_at_k_beta(self):
        ranked, qrels = self._make_data()
        f1 = ranklab.f_measure_at_k(ranked, qrels, k=5, beta=1.0)
        f2 = ranklab.f_measure_at_k(ranked, qrels, k=5, beta=2.0)
        # Both valid scores
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= f2 <= 1.0

    def test_success_at_k_found(self):
        ranked, qrels = self._make_data()
        # All 3 relevant docs in top-5 => hits_at_k == 1.0
        assert ranklab.success_at_k(ranked, qrels, k=5) == 1.0

    def test_success_at_k_partial(self):
        ranked, qrels = self._make_data()
        # k=1: 1 of 3 relevant found => 1/3
        score = ranklab.success_at_k(ranked, qrels, k=1)
        assert score > 0.0

    def test_success_at_k_not_found(self):
        ranked = [("doc2", 0.9), ("doc4", 0.8)]
        qrels = {"doc1": 1, "doc3": 1}
        assert ranklab.success_at_k(ranked, qrels, k=2) == 0.0

    def test_r_precision(self):
        ranked, qrels = self._make_data()
        score = ranklab.r_precision(ranked, qrels)
        # 3 relevant docs; in top-3 we have doc1 (rel), doc2 (not), doc3 (rel) => 2/3
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_average_precision(self):
        ranked, qrels = self._make_data()
        score = ranklab.average_precision(ranked, qrels)
        assert 0.0 < score <= 1.0


class TestStatistics:
    def test_paired_t_test_basic(self):
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.4, 0.55, 0.6, 0.75, 0.8]
        result = ranklab.paired_t_test(a, b)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "degrees_of_freedom" in result
        assert "mean_difference" in result
        assert "std_error" in result
        assert result["degrees_of_freedom"] == 4
        assert 0.0 <= result["p_value"] <= 1.0

    def test_paired_t_test_alpha(self):
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.4, 0.55, 0.6, 0.75, 0.8]
        r1 = ranklab.paired_t_test(a, b, alpha=0.01)
        r2 = ranklab.paired_t_test(a, b, alpha=0.99)
        # Stricter alpha => less likely significant
        assert isinstance(r1["significant"], bool)
        assert isinstance(r2["significant"], bool)

    def test_paired_t_test_numpy(self):
        a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        b = np.array([0.4, 0.55, 0.6, 0.75, 0.8])
        result = ranklab.paired_t_test(a, b)
        assert isinstance(result["t_statistic"], float)

    def test_paired_t_test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.paired_t_test([1, 2, 3], [1, 2])

    def test_confidence_interval_basic(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        lower, upper = ranklab.confidence_interval(scores)
        assert lower < upper
        mean = sum(scores) / len(scores)
        assert lower < mean < upper

    def test_confidence_interval_custom_confidence(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        lo_90, hi_90 = ranklab.confidence_interval(scores, confidence=0.90)
        lo_99, hi_99 = ranklab.confidence_interval(scores, confidence=0.99)
        # 99% CI should be wider than 90% CI
        assert (hi_99 - lo_99) > (hi_90 - lo_90)

    def test_confidence_interval_numpy(self):
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        lower, upper = ranklab.confidence_interval(scores)
        assert lower < upper

    def test_cohens_d_basic(self):
        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        d = ranklab.cohens_d(a, b)
        assert d > 0.0  # A > B

    def test_cohens_d_identical(self):
        a = [0.5, 0.6, 0.7]
        d = ranklab.cohens_d(a, a)
        assert abs(d) < 1e-9

    def test_cohens_d_numpy(self):
        a = np.array([0.5, 0.6, 0.7])
        b = np.array([0.4, 0.5, 0.6])
        d = ranklab.cohens_d(a, b)
        assert isinstance(d, float)

    def test_cohens_d_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ranklab.cohens_d([1, 2, 3], [1, 2])


class TestTrecIO:
    def test_load_trec_run(self, tmp_path):
        run_file = tmp_path / "run.txt"
        run_file.write_text(
            "1 Q0 doc1 1 0.9 run1\n"
            "1 Q0 doc2 2 0.8 run1\n"
            "2 Q0 doc3 1 0.95 run1\n"
        )
        result = ranklab.load_trec_run(str(run_file))
        assert "1" in result
        assert "2" in result
        assert len(result["1"]) == 2
        # Should be sorted by score descending
        assert result["1"][0][1] >= result["1"][1][1]

    def test_load_qrels(self, tmp_path):
        qrels_file = tmp_path / "qrels.txt"
        qrels_file.write_text(
            "1 0 doc1 2\n"
            "1 0 doc2 1\n"
            "2 0 doc3 0\n"
        )
        result = ranklab.load_qrels(str(qrels_file))
        assert "1" in result
        assert result["1"]["doc1"] == 2
        assert result["1"]["doc2"] == 1
        assert "2" in result

    def test_load_trec_run_invalid(self, tmp_path):
        run_file = tmp_path / "bad.txt"
        run_file.write_text("bad format\n")
        with pytest.raises(OSError):
            ranklab.load_trec_run(str(run_file))

    def test_load_trec_run_missing_file(self):
        with pytest.raises(OSError):
            ranklab.load_trec_run("/nonexistent/path.txt")


class TestBatchEvaluation:
    def test_evaluate_batch_basic(self):
        rankings = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        qrels = [
            ["doc1", "doc3"],
            ["doc4"],
        ]
        result = ranklab.evaluate_batch(rankings, qrels, ["ndcg@10", "precision@5"])
        assert "per_query" in result
        assert "aggregated" in result
        assert len(result["per_query"]) == 2
        assert "ndcg@10" in result["aggregated"]
        assert "precision@5" in result["aggregated"]

    def test_evaluate_batch_metrics(self):
        rankings = [["doc1", "doc2", "doc3"]]
        qrels = [["doc1", "doc3"]]
        result = ranklab.evaluate_batch(rankings, qrels, ["mrr", "ap", "err@10"])
        assert result["aggregated"]["mrr"] > 0.0
        assert result["aggregated"]["ap"] > 0.0

    def test_evaluate_trec(self, tmp_path):
        run_file = tmp_path / "run.txt"
        run_file.write_text(
            "q1 Q0 doc1 1 0.9 run1\n"
            "q1 Q0 doc2 2 0.8 run1\n"
            "q1 Q0 doc3 3 0.7 run1\n"
        )
        qrels_file = tmp_path / "qrels.txt"
        qrels_file.write_text(
            "q1 0 doc1 2\n"
            "q1 0 doc2 0\n"
            "q1 0 doc3 1\n"
        )
        result = ranklab.evaluate_trec(
            str(run_file), str(qrels_file), ["ndcg@10", "mrr"]
        )
        assert "per_query" in result
        assert "aggregated" in result
        assert len(result["per_query"]) == 1
        assert result["per_query"][0]["query_id"] == "q1"
        assert result["aggregated"]["mrr"] > 0.0
