//! End-to-end retrieval pipeline: tokenize, index, score, rank.
//!
//! Composes [`textprep`] for text normalization and tokenization,
//! [`postings`] for inverted indexing with candidate retrieval,
//! and [`rankfns`] for BM25 / TF-IDF / language model scoring.
//!
//! Requires the `pipeline` feature.
//!
//! ```rust,ignore
//! use rankit::pipeline::{Pipeline, Scoring};
//!
//! let mut p = Pipeline::bm25();
//! p.add(0, "rust systems programming language");
//! p.add(1, "python machine learning data science");
//! let results = p.search("rust programming", 5);
//! ```

use postings::{DocId, PostingsIndex};

/// Relevance scoring method.
#[derive(Debug, Clone)]
pub enum Scoring {
    /// Okapi BM25 with saturation `k1` and length normalization `b`.
    Bm25 {
        /// Term frequency saturation (typical: 1.2).
        k1: f32,
        /// Document length normalization in \[0, 1\] (typical: 0.75).
        b: f32,
    },
    /// Log-scaled TF with standard IDF.
    TfIdf,
    /// Dirichlet-smoothed query likelihood language model.
    DirichletLm {
        /// Prior strength (typical: 1000--2000).
        mu: f32,
    },
    /// Jelinek-Mercer smoothed query likelihood language model:
    /// `p(t|d) = (1 - lambda) * tf / |d| + lambda * p(t|C)`.
    JelinekMercerLm {
        /// Interpolation weight for collection model in \[0, 1\].
        lambda: f32,
    },
}

impl Default for Scoring {
    fn default() -> Self {
        Scoring::Bm25 { k1: 1.2, b: 0.75 }
    }
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document identifier.
    pub doc_id: DocId,
    /// Relevance score. Higher is better for BM25/TF-IDF; less negative for LM.
    pub score: f32,
    /// 1-based rank in result list.
    pub rank: usize,
}

/// Retrieval pipeline: tokenize text, build inverted index, score and rank.
pub struct Pipeline {
    index: PostingsIndex<String>,
    scoring: Scoring,
    /// Token count over live documents (the LM collection size).
    total_corpus_tokens: u64,
}

impl Pipeline {
    /// Create a pipeline with the given scoring method.
    pub fn new(scoring: Scoring) -> Self {
        Self {
            index: PostingsIndex::new(),
            scoring,
            total_corpus_tokens: 0,
        }
    }

    /// Create a pipeline with default BM25 scoring (k1=1.2, b=0.75).
    pub fn bm25() -> Self {
        Self::new(Scoring::default())
    }

    /// Tokenize text: normalize (NFKC + lowercase + collapse whitespace) then split.
    fn tokenize(text: &str) -> Vec<String> {
        let clean = textprep::scrub_with(text, &textprep::ScrubConfig::search_key());
        textprep::tokenize::words(&clean)
            .into_iter()
            .map(|s: &str| s.to_string())
            .collect()
    }

    /// Add a document. Text is tokenized internally.
    /// Returns `Err` if `doc_id` already exists.
    pub fn add(&mut self, doc_id: DocId, text: &str) -> Result<(), postings::Error> {
        let tokens = Self::tokenize(text);
        self.index.add_document(doc_id, &tokens)?;
        self.total_corpus_tokens += tokens.len() as u64;
        Ok(())
    }

    /// Remove a document. Returns `true` if it existed.
    pub fn remove(&mut self, doc_id: DocId) -> bool {
        let len = u64::from(self.index.document_len(doc_id));
        let existed = self.index.delete_document(doc_id);
        if existed {
            self.total_corpus_tokens -= len;
        }
        existed
    }

    /// Number of indexed documents.
    pub fn num_docs(&self) -> u32 {
        self.index.num_docs()
    }

    /// Average document length in tokens.
    pub fn avg_doc_len(&self) -> f32 {
        self.index.avg_doc_len()
    }

    /// Search with OR semantics: documents containing any query term.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let tokens = Self::tokenize(query);
        let candidates = self.index.candidates(&tokens);
        self.rank(&tokens, &candidates, top_k)
    }

    /// Search with AND semantics: documents containing all query terms.
    pub fn search_all(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let tokens = Self::tokenize(query);
        let candidates = self.index.candidates_all_terms(&tokens);
        self.rank(&tokens, &candidates, top_k)
    }

    /// Access the underlying postings index (for advanced queries).
    pub fn index(&self) -> &PostingsIndex<String> {
        &self.index
    }

    /// Current scoring method.
    pub fn scoring(&self) -> &Scoring {
        &self.scoring
    }

    fn rank(
        &self,
        query_tokens: &[String],
        candidates: &[DocId],
        top_k: usize,
    ) -> Vec<SearchResult> {
        let n = self.index.num_docs();
        let avg_dl = self.index.avg_doc_len();
        // Collection probabilities for the LM scorers, once per query term.
        let p_c: Vec<f32> = match self.scoring {
            Scoring::DirichletLm { .. } | Scoring::JelinekMercerLm { .. } => {
                query_tokens.iter().map(|t| self.corpus_prob(t)).collect()
            }
            _ => Vec::new(),
        };

        let mut scored: Vec<(DocId, f32)> = candidates
            .iter()
            .map(|&id| (id, self.score_doc(id, query_tokens, &p_c, n, avg_dl)))
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);

        scored
            .into_iter()
            .enumerate()
            .map(|(i, (doc_id, score))| SearchResult {
                doc_id,
                score,
                rank: i + 1,
            })
            .collect()
    }

    fn score_doc(
        &self,
        doc_id: DocId,
        query: &[String],
        p_c: &[f32],
        n_docs: u32,
        avg_dl: f32,
    ) -> f32 {
        let dl = self.index.document_len(doc_id) as f32;

        match &self.scoring {
            Scoring::Bm25 { k1, b } => {
                let mut s = 0.0f32;
                for t in query {
                    let tf = self.index.term_frequency(doc_id, t.as_str()) as f32;
                    if tf > 0.0 {
                        let df = self.index.df(t.as_str());
                        s += rankfns::bm25_idf_plus1(n_docs, df)
                            * rankfns::bm25_tf(tf, dl, avg_dl, *k1, *b);
                    }
                }
                s
            }
            Scoring::TfIdf => {
                let mut s = 0.0f32;
                for t in query {
                    let tf = self.index.term_frequency(doc_id, t.as_str());
                    if tf > 0 {
                        let df = self.index.df(t.as_str());
                        s += rankfns::tf_transform(tf, rankfns::TfVariant::LogScaled)
                            * rankfns::idf_transform(n_docs, df, rankfns::IdfVariant::Standard);
                    }
                }
                s
            }
            Scoring::DirichletLm { mu } => self.lm_score(doc_id, query, p_c, |tf, p_c| {
                rankfns::lm_smoothed_p(tf, dl, p_c, rankfns::SmoothingMethod::Dirichlet { mu: *mu })
            }),
            Scoring::JelinekMercerLm { lambda } => self.lm_score(doc_id, query, p_c, |tf, p_c| {
                rankfns::lm_smoothed_p(
                    tf,
                    dl,
                    p_c,
                    // rankfns' lambda weights the document model; ours weights
                    // the collection model.
                    rankfns::SmoothingMethod::JelinekMercer {
                        lambda: 1.0 - *lambda,
                    },
                )
            }),
        }
    }

    fn lm_score(
        &self,
        doc_id: DocId,
        query: &[String],
        p_c: &[f32],
        smooth: impl Fn(f32, f32) -> f32,
    ) -> f32 {
        let mut log_s = 0.0f32;
        for (t, &p_c) in query.iter().zip(p_c) {
            let tf = self.index.term_frequency(doc_id, t.as_str()) as f32;
            let p = smooth(tf, p_c);
            if p > 0.0 {
                log_s += p.ln();
            }
        }
        log_s
    }

    fn corpus_prob(&self, term: &str) -> f32 {
        if self.total_corpus_tokens == 0 {
            return 0.0;
        }
        // Read from the live postings so rejected adds and removals cannot
        // leave stale collection counts.
        let tf: u64 = self
            .index
            .postings_iter(term)
            .map(|(_, w)| u64::from(w))
            .sum();
        tf as f32 / self.total_corpus_tokens as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_search() {
        let mut p = Pipeline::bm25();
        p.add(0, "the quick brown fox").unwrap();
        p.add(1, "the lazy brown dog").unwrap();
        p.add(2, "a fox jumps over the dog").unwrap();

        let results = p.search("fox", 10);
        assert!(!results.is_empty());
        // Both docs 0 and 2 mention fox.
        let ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn conjunctive_search() {
        let mut p = Pipeline::bm25();
        p.add(0, "rust programming language").unwrap();
        p.add(1, "rust belt manufacturing").unwrap();
        p.add(2, "programming in python").unwrap();

        let results = p.search_all("rust programming", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 0);
    }

    #[test]
    fn delete_removes_from_results() {
        let mut p = Pipeline::bm25();
        p.add(0, "alpha beta gamma").unwrap();
        p.add(1, "beta delta epsilon").unwrap();

        assert_eq!(p.search("beta", 10).len(), 2);
        p.remove(0);
        let results = p.search("beta", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 1);
    }

    fn lm_scores(p: &Pipeline, q: &str) -> Vec<(DocId, f32)> {
        let mut v: Vec<_> = p
            .search(q, 10)
            .into_iter()
            .map(|r| (r.doc_id, r.score))
            .collect();
        v.sort_by_key(|x| x.0);
        v
    }

    #[test]
    fn rejected_duplicate_add_leaves_lm_scores_unchanged() {
        let mut p = Pipeline::new(Scoring::DirichletLm { mu: 10.0 });
        p.add(0, "apple banana").unwrap();
        p.add(1, "cherry date").unwrap();
        let before = lm_scores(&p, "apple");
        assert!(p.add(0, "apple apple apple apple").is_err());
        assert_eq!(lm_scores(&p, "apple"), before);
    }

    #[test]
    fn remove_matches_never_added_for_lm_scores() {
        let mut a = Pipeline::new(Scoring::DirichletLm { mu: 10.0 });
        a.add(0, "apple banana").unwrap();
        a.add(1, "cherry date").unwrap();
        a.add(2, "apple apple apple apple apple apple").unwrap();
        assert!(a.remove(2));
        assert!(!a.remove(2));
        let mut b = Pipeline::new(Scoring::DirichletLm { mu: 10.0 });
        b.add(0, "apple banana").unwrap();
        b.add(1, "cherry date").unwrap();
        assert_eq!(lm_scores(&a, "apple"), lm_scores(&b, "apple"));
    }

    #[test]
    fn jelinek_mercer_lambda_weights_collection_model() {
        let mut p = Pipeline::new(Scoring::JelinekMercerLm { lambda: 0.2 });
        p.add(0, "apple banana").unwrap();
        p.add(1, "cherry date").unwrap();
        // p(apple|d0) = 0.8 * (1/2) + 0.2 * (1/4) = 0.45.
        let expected = 0.45f32.ln();
        let scores = lm_scores(&p, "apple");
        assert_eq!(scores.len(), 1);
        assert!(
            (scores[0].1 - expected).abs() < 1e-6,
            "score {} vs expected {expected}",
            scores[0].1
        );
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut p = Pipeline::bm25();
        p.add(0, "some document text").unwrap();
        assert!(p.search("", 10).is_empty());
    }

    #[test]
    fn scoring_methods_all_rank() {
        let docs = [
            (0, "information retrieval search engine"),
            (1, "machine learning neural network"),
            (2, "search engine optimization ranking"),
        ];

        for scoring in [
            Scoring::Bm25 { k1: 1.2, b: 0.75 },
            Scoring::TfIdf,
            Scoring::DirichletLm { mu: 1000.0 },
            Scoring::JelinekMercerLm { lambda: 0.7 },
        ] {
            let mut p = Pipeline::new(scoring.clone());
            for &(id, text) in &docs {
                p.add(id, text).unwrap();
            }
            let results = p.search("search engine", 3);
            assert!(
                !results.is_empty(),
                "scoring {:?} returned no results",
                scoring
            );
            // Docs 0 and 2 should appear; doc 1 should not.
            let ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();
            assert!(!ids.contains(&1), "scoring {:?} matched wrong doc", scoring);
        }
    }
}
