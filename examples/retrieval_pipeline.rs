//! End-to-end retrieval pipeline: tokenize, index, score, rank, evaluate.
//!
//! Demonstrates the `pipeline` feature composing textprep + postings + rankfns,
//! then evaluating results with rankit's IR metrics.
//!
//! Run: cargo run --example retrieval_pipeline --features pipeline,eval

use rankit::pipeline::{Pipeline, Scoring};

fn main() {
    // --- Corpus ---
    let docs: Vec<(u32, &str)> = vec![
        (
            0,
            "Rust is a systems programming language focused on safety and performance",
        ),
        (
            1,
            "Python is popular for machine learning and data science applications",
        ),
        (
            2,
            "Search engines use inverted indexes and BM25 scoring for relevance ranking",
        ),
        (
            3,
            "Rust's ownership model prevents data races at compile time",
        ),
        (
            4,
            "Information retrieval systems combine indexing with relevance scoring",
        ),
        (
            5,
            "Neural networks learn representations for natural language processing",
        ),
        (
            6,
            "The BM25 algorithm normalizes term frequency by document length",
        ),
        (7, "Rust and C++ compete for systems programming use cases"),
        (
            8,
            "Vector search uses approximate nearest neighbors for similarity retrieval",
        ),
        (
            9,
            "Language models assign probability to word sequences using smoothing",
        ),
    ];

    let queries = [
        "rust systems programming",
        "search engine ranking BM25",
        "machine learning neural network",
        "information retrieval scoring",
    ];

    // --- Compare scoring methods ---
    let methods: Vec<(&str, Scoring)> = vec![
        ("BM25", Scoring::Bm25 { k1: 1.2, b: 0.75 }),
        ("TF-IDF", Scoring::TfIdf),
        ("Dirichlet LM", Scoring::DirichletLm { mu: 1000.0 }),
        ("JM LM", Scoring::JelinekMercerLm { lambda: 0.7 }),
    ];

    for (method_name, scoring) in &methods {
        let mut p = Pipeline::new(scoring.clone());
        for &(id, text) in &docs {
            p.add(id, text).unwrap();
        }

        println!(
            "=== {method_name} ({} docs, avg_dl={:.1}) ===\n",
            p.num_docs(),
            p.avg_doc_len()
        );

        for query in &queries {
            let results = p.search(query, 3);
            println!("  query: \"{query}\"");
            for r in &results {
                let snippet = &docs[r.doc_id as usize].1;
                let truncated = if snippet.len() > 60 {
                    format!("{}...", &snippet[..57])
                } else {
                    snippet.to_string()
                };
                println!(
                    "    #{} doc={:<2} score={:>8.3}  {}",
                    r.rank, r.doc_id, r.score, truncated
                );
            }
            println!();
        }
    }

    // --- Evaluate with rankit's IR metrics ---
    #[cfg(feature = "eval")]
    {
        use std::collections::HashSet;

        println!("=== IR Evaluation ===\n");

        // Ground truth: relevant docs per query.
        let relevance: Vec<(&str, HashSet<u32>)> = vec![
            (
                "rust systems programming",
                [0, 3, 7].iter().copied().collect(),
            ),
            (
                "search engine ranking BM25",
                [2, 4, 6].iter().copied().collect(),
            ),
            (
                "machine learning neural network",
                [1, 5].iter().copied().collect(),
            ),
            (
                "information retrieval scoring",
                [2, 4, 6, 8].iter().copied().collect(),
            ),
        ];

        let mut p = Pipeline::bm25();
        for &(id, text) in &docs {
            p.add(id, text).unwrap();
        }

        println!("  {:<35} P@3    R@3    MRR    NDCG@3", "Query");
        println!("  {:<35} -----  -----  -----  ------", "");

        for (query, rel) in &relevance {
            let results = p.search(query, 3);
            let ranked: Vec<u32> = results.iter().map(|r| r.doc_id).collect();

            let p_at_3 = rankit::binary::precision_at_k(&ranked, rel, 3);
            let r_at_3 = rankit::binary::recall_at_k(&ranked, rel, 3);
            let mrr = rankit::binary::mrr(&ranked, rel);
            let ndcg = rankit::binary::ndcg_at_k(&ranked, rel, 3);

            println!(
                "  {:<35} {:.3}  {:.3}  {:.3}  {:.3}",
                query, p_at_3, r_at_3, mrr, ndcg
            );
        }
        println!();
    }

    // --- Conjunctive vs disjunctive ---
    println!("=== AND vs OR retrieval ===\n");
    let mut p = Pipeline::bm25();
    for &(id, text) in &docs {
        p.add(id, text).unwrap();
    }

    let q = "rust machine learning";
    let or_results = p.search(q, 5);
    let and_results = p.search_all(q, 5);
    println!("  query: \"{q}\"");
    println!(
        "  OR  hits: {} (docs: {:?})",
        or_results.len(),
        or_results.iter().map(|r| r.doc_id).collect::<Vec<_>>()
    );
    println!(
        "  AND hits: {} (docs: {:?})",
        and_results.len(),
        and_results.iter().map(|r| r.doc_id).collect::<Vec<_>>()
    );
}
