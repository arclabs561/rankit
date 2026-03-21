"""Tests for ranklab Python bindings."""

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
        assert ranks == []

    def test_soft_rank_single(self):
        ranks = ranklab.soft_rank([42.0], temperature=1.0)
        assert ranks == [0.0]


class TestLosses:
    def test_ranknet_loss_perfect(self):
        # Predictions match relevance ordering: loss should be low
        predictions = [0.9, 0.5, 0.1]
        relevance = [2.0, 1.0, 0.0]
        loss = ranklab.ranknet_loss(predictions, relevance)
        assert loss >= 0.0
        assert loss < 1.0

    def test_ranknet_loss_reversed(self):
        # Predictions opposite to relevance: loss should be high
        predictions = [0.1, 0.5, 0.9]
        relevance = [2.0, 1.0, 0.0]
        loss = ranklab.ranknet_loss(predictions, relevance)
        assert loss > 0.5

    def test_ranknet_loss_comparison(self):
        relevance = [2.0, 1.0, 0.0]
        good = ranklab.ranknet_loss([0.9, 0.5, 0.1], relevance)
        bad = ranklab.ranknet_loss([0.1, 0.5, 0.9], relevance)
        assert good < bad

    def test_approx_ndcg_perfect(self):
        # Perfect ranking should yield nDCG close to 1.0
        predictions = [0.9, 0.5, 0.1]
        relevance = [2.0, 1.0, 0.0]
        score = ranklab.approx_ndcg(predictions, relevance, temperature=1.0)
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


class TestGradients:
    def test_lambdarank_gradients(self):
        scores = [0.5, 0.8, 0.3]
        relevance = [3.0, 1.0, 2.0]
        grads = ranklab.compute_lambdarank_gradients(scores, relevance)
        assert len(grads) == 3
        assert any(g != 0.0 for g in grads)

    def test_ranking_svm_gradients(self):
        scores = [0.5, 0.8, 0.3]
        relevance = [3.0, 1.0, 2.0]
        grads = ranklab.compute_ranking_svm_gradients(scores, relevance)
        assert len(grads) == 3


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
        # doc1 and doc2 are relevant (value > 0), both in top-2
        assert score > 0.9

    def test_mrr_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc2": 1, "doc3": 0}
        score = ranklab.mrr(ranked, qrels)
        # First relevant doc is doc2 at position 2 -> MRR = 0.5
        assert abs(score - 0.5) < 1e-9

    def test_precision_at_k_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc1": 1, "doc3": 1}
        score = ranklab.precision_at_k(ranked, qrels, k=2)
        # Top-2: doc1 (relevant), doc2 (not relevant) -> 0.5
        assert abs(score - 0.5) < 1e-9

    def test_recall_at_k_eval(self):
        ranked = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        qrels = {"doc1": 1, "doc3": 1}
        score = ranklab.recall_at_k(ranked, qrels, k=3)
        # All 3 docs examined, 2 relevant found -> 1.0
        assert abs(score - 1.0) < 1e-9


class TestEdgeCases:
    def test_empty_inputs(self):
        assert ranklab.soft_rank([], temperature=1.0) == []
        assert ranklab.ranknet_loss([], []) == 0.0

    def test_single_element(self):
        ranks = ranklab.soft_rank([1.0], temperature=1.0)
        assert ranks == [0.0]

    def test_differentiable_topk(self):
        scores = [5.0, 1.0, 2.0, 4.0, 3.0]
        values, ranks = ranklab.differentiable_topk(scores, k=3, temperature=1.0)
        assert len(values) == 5
        assert len(ranks) == 5
