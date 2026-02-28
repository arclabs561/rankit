//! Export utilities for evaluation results (CSV, JSON).

use crate::eval::batch::BatchResults;
use std::collections::HashMap;
use std::io::Write;

/// Export batch results to CSV format.
pub fn export_to_csv<W: Write>(results: &BatchResults, writer: &mut W) -> std::io::Result<()> {
    if results.query_results.is_empty() {
        return Ok(());
    }

    let metric_names: Vec<&String> = results.query_results[0].metrics.keys().collect();

    write!(writer, "query_id")?;
    for metric_name in &metric_names {
        write!(writer, ",{}", metric_name)?;
    }
    writeln!(writer)?;

    for query_result in &results.query_results {
        write!(writer, "{}", query_result.query_id)?;
        for metric_name in &metric_names {
            let value = query_result.metrics.get(*metric_name).unwrap_or(&0.0);
            write!(writer, ",{:.6}", value)?;
        }
        writeln!(writer)?;
    }

    writeln!(writer)?;
    write!(writer, "mean")?;
    for metric_name in &metric_names {
        let value = results.aggregated.get(*metric_name).unwrap_or(&0.0);
        write!(writer, ",{:.6}", value)?;
    }
    writeln!(writer)?;

    Ok(())
}

/// Export batch results to JSON format.
#[cfg(feature = "serde")]
pub fn export_to_json(results: &BatchResults) -> Result<String, serde_json::Error> {
    #[derive(serde::Serialize)]
    struct ExportableResults {
        query_results: Vec<QueryResultsExport>,
        aggregated: HashMap<String, f64>,
    }

    #[derive(serde::Serialize)]
    struct QueryResultsExport {
        query_id: String,
        metrics: HashMap<String, f64>,
    }

    let exportable = ExportableResults {
        query_results: results
            .query_results
            .iter()
            .map(|qr| QueryResultsExport {
                query_id: qr.query_id.clone(),
                metrics: qr.metrics.clone(),
            })
            .collect(),
        aggregated: results.aggregated.clone(),
    };

    serde_json::to_string_pretty(&exportable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::batch::evaluate_batch_binary;
    use std::collections::HashSet;

    #[test]
    fn test_export_to_csv() {
        let rankings = vec![vec!["doc1", "doc2", "doc3"]];
        let qrels = vec![["doc1", "doc3"].into_iter().collect::<HashSet<_>>()];
        let results = evaluate_batch_binary(&rankings, &qrels, &["ndcg@10"]);

        let mut csv = Vec::new();
        export_to_csv(&results, &mut csv).unwrap();

        let csv_str = String::from_utf8(csv).unwrap();
        assert!(csv_str.contains("query_id"));
        assert!(csv_str.contains("ndcg@10"));
    }
}
