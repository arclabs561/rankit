//! IR evaluation metrics on sample relevance judgments.
//!
//! Demonstrates NDCG, MAP, MRR, Precision@K, and Recall@K using both
//! binary and graded relevance evaluation.

use std::collections::{HashMap, HashSet};

use rankit::eval::{binary, graded};

fn main() {
    // --- Binary relevance ---
    // A search engine returned 10 documents; 4 are relevant.
    let ranked = vec![
        "doc_1", "doc_7", "doc_3", "doc_9", "doc_2", "doc_5", "doc_8", "doc_4", "doc_6", "doc_10",
    ];
    let relevant: HashSet<&str> = ["doc_1", "doc_3", "doc_5", "doc_8"].into_iter().collect();

    println!("=== Binary relevance ===");
    println!("Ranked:   {:?}", ranked);
    println!("Relevant: {:?}\n", relevant);

    for k in [1, 3, 5, 10] {
        let p = binary::precision_at_k(&ranked, &relevant, k);
        let r = binary::recall_at_k(&ranked, &relevant, k);
        println!("  P@{k:<2} = {p:.3}    R@{k:<2} = {r:.3}");
    }

    let mrr_val = binary::mrr(&ranked, &relevant);
    let ap = binary::average_precision(&ranked, &relevant);
    let ndcg5 = binary::ndcg_at_k(&ranked, &relevant, 5);

    println!("\n  MRR   = {mrr_val:.3}");
    println!("  MAP   = {ap:.3}");
    println!("  NDCG@5 = {ndcg5:.3}");

    // --- Graded relevance ---
    // Same documents but with graded relevance (0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant).
    let ranked_scored: Vec<(String, f32)> = ranked
        .iter()
        .enumerate()
        .map(|(i, &doc)| (doc.to_string(), 1.0 - i as f32 * 0.1))
        .collect();

    let mut qrels: HashMap<String, u32> = HashMap::new();
    qrels.insert("doc_1".into(), 3); // highly relevant
    qrels.insert("doc_3".into(), 2); // relevant
    qrels.insert("doc_5".into(), 1); // marginal
    qrels.insert("doc_8".into(), 1); // marginal
                                     // doc_7, doc_9, doc_2, etc. are irrelevant (absent = 0)

    println!("\n=== Graded relevance ===");
    println!("Judgments: {:?}\n", qrels);

    for k in [3, 5, 10] {
        let ndcg = graded::compute_ndcg(&ranked_scored, &qrels, k);
        println!("  NDCG@{k:<2} = {ndcg:.4}");
    }

    let map_graded = graded::compute_map(&ranked_scored, &qrels);
    println!("  MAP     = {map_graded:.4}");
}
