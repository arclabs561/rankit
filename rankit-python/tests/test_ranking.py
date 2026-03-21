"""Tests for ranklab Python bindings."""

import numpy as np
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
