#![cfg(all(feature = "pipeline", feature = "eval"))]

use rankit::binary;
use rankit::pipeline::{Pipeline, Scoring};
use std::collections::HashSet;

#[test]
fn unsmoothed_query_keeps_zero_probability_matches_at_the_end() {
    for scorer in [
        Scoring::JelinekMercerLm { lambda: 0.0 },
        Scoring::DirichletLm { mu: 0.0 },
    ] {
        let mut index = Pipeline::new(scorer);
        index.add(0, "APPLE").unwrap();
        index.add(1, "apple banana").unwrap();
        let hits = index.search("APPLE banana", 2);
        let ids: Vec<_> = hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(binary::precision_at_k(&ids, &HashSet::from([1]), 1), 1.0);
        assert!((hits[0].score - 0.25_f32.ln()).abs() < 1e-6);
        assert_eq!(hits[1].score, f32::NEG_INFINITY);
        assert_eq!(index.search("apple unindexed", 1)[0].score, 0.0);
    }
}

#[test]
fn small_collection_weight_still_smooths_absent_terms() {
    let lambda = 1e-9;
    let mut index = Pipeline::new(Scoring::JelinekMercerLm { lambda });
    index.add(0, "apple").unwrap();
    index.add(1, "apple banana").unwrap();
    let hits = index.search("apple banana", 2);
    let partial = hits.iter().find(|h| h.doc_id == 0).unwrap();
    assert!((partial.score - (lambda / 3.0).ln()).abs() < 1e-5);
}
