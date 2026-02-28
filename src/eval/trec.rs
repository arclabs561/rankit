//! TREC format parsing utilities.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A TREC run file entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TrecRun {
    /// Query identifier.
    pub query_id: String,
    /// Document identifier.
    pub doc_id: String,
    /// Rank (1 = best).
    pub rank: usize,
    /// Retrieval score (higher is better; must be finite).
    pub score: f32,
    /// Run tag (system name).
    pub run_tag: String,
}

/// Ground truth relevance judgments (qrels).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qrel {
    /// Query identifier.
    pub query_id: String,
    /// Document identifier.
    pub doc_id: String,
    /// Relevance label (0 = not relevant; 1+ = relevant).
    pub relevance: u32,
}

/// Load TREC run file.
///
/// Format: `query_id Q0 doc_id rank score run_tag`
pub fn load_trec_runs(path: impl AsRef<Path>) -> Result<Vec<TrecRun>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open TREC runs file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut runs = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context("Failed to read line")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            if parts.len() == 5 && parts[1] != "Q0" {
                return Err(anyhow::anyhow!(
                    "Line {}: Expected 'Q0' as second field, found '{}'. Format: query_id Q0 doc_id rank score run_tag",
                    line_num + 1, parts.get(1).unwrap_or(&"<missing>")
                ));
            }
            return Err(anyhow::anyhow!(
                "Line {}: Invalid TREC run format. Expected 6 fields, found {}. Format: query_id Q0 doc_id rank score run_tag\nLine: {}",
                line_num + 1, parts.len(), line
            ));
        }

        if parts[1] != "Q0" {
            return Err(anyhow::anyhow!(
                "Line {}: Expected 'Q0' as second field, found '{}'. Format: query_id Q0 doc_id rank score run_tag",
                line_num + 1, parts[1]
            ));
        }

        let query_id = parts[0].to_string();
        let doc_id = parts[2].to_string();
        let rank: usize = parts[3]
            .parse()
            .with_context(|| format!("Invalid rank on line {}: {}", line_num + 1, parts[3]))?;
        let score: f32 = parts[4]
            .parse()
            .with_context(|| format!("Invalid score on line {}: {}", line_num + 1, parts[4]))?;

        if !score.is_finite() {
            return Err(anyhow::anyhow!(
                "Line {}: Invalid score (NaN or Infinity): {}",
                line_num + 1,
                score
            ));
        }

        let run_tag = if parts.len() > 6 {
            parts[5..].join(" ")
        } else {
            parts[5].to_string()
        };

        runs.push(TrecRun {
            query_id,
            doc_id,
            rank,
            score,
            run_tag,
        });
    }

    Ok(runs)
}

/// Load TREC qrels file.
///
/// Format: `query_id 0 doc_id relevance`
pub fn load_qrels(path: impl AsRef<Path>) -> Result<Vec<Qrel>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open qrels file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut qrels = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context("Failed to read line")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(anyhow::anyhow!(
                "Line {}: Invalid TREC qrels format. Expected 4 fields, found {}. Format: query_id 0 doc_id relevance\nLine: {}",
                line_num + 1, parts.len(), line
            ));
        }

        if parts[1] != "0" {
            return Err(anyhow::anyhow!(
                "Line {}: Expected '0' as second field in qrels, found '{}'. Format: query_id 0 doc_id relevance",
                line_num + 1, parts[1]
            ));
        }

        let query_id = parts[0].to_string();
        let doc_id = parts[2].to_string();
        let relevance: u32 = parts[3]
            .parse()
            .with_context(|| format!("Invalid relevance on line {}: {}", line_num + 1, parts[3]))?;

        qrels.push(Qrel {
            query_id,
            doc_id,
            relevance,
        });
    }

    Ok(qrels)
}

/// Group runs by query and run tag.
///
/// Returns: query_id -> run_tag -> Vec<(doc_id, score)>, sorted by score descending.
pub fn group_runs_by_query(
    runs: &[TrecRun],
) -> HashMap<String, HashMap<String, Vec<(String, f32)>>> {
    let mut grouped: HashMap<String, HashMap<String, Vec<(String, f32)>>> = HashMap::new();

    for run in runs {
        grouped
            .entry(run.query_id.clone())
            .or_default()
            .entry(run.run_tag.clone())
            .or_default()
            .push((run.doc_id.clone(), run.score));
    }

    for query_runs in grouped.values_mut() {
        for run_results in query_runs.values_mut() {
            run_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }
    }

    grouped
}

/// Group qrels by query.
///
/// Returns: query_id -> doc_id -> relevance
pub fn group_qrels_by_query(qrels: &[Qrel]) -> HashMap<String, HashMap<String, u32>> {
    let mut grouped: HashMap<String, HashMap<String, u32>> = HashMap::new();

    for qrel in qrels {
        grouped
            .entry(qrel.query_id.clone())
            .or_default()
            .insert(qrel.doc_id.clone(), qrel.relevance);
    }

    grouped
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_load_trec_runs() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("runs.txt");
        let mut file = fs::File::create(&file_path).unwrap();

        writeln!(file, "1 Q0 doc1 1 0.9 run1").unwrap();
        writeln!(file, "1 Q0 doc2 2 0.8 run1").unwrap();
        writeln!(file, "2 Q0 doc3 1 0.95 run1").unwrap();

        let runs = load_trec_runs(&file_path).unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].query_id, "1");
        assert_eq!(runs[0].doc_id, "doc1");
    }

    #[test]
    fn test_load_qrels() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("qrels.txt");
        let mut file = fs::File::create(&file_path).unwrap();

        writeln!(file, "1 0 doc1 2").unwrap();
        writeln!(file, "1 0 doc2 1").unwrap();

        let qrels = load_qrels(&file_path).unwrap();
        assert_eq!(qrels.len(), 2);
        assert_eq!(qrels[0].relevance, 2);
    }

    #[test]
    fn test_group_runs_by_query() {
        let runs = vec![
            TrecRun {
                query_id: "1".to_string(),
                doc_id: "doc1".to_string(),
                rank: 1,
                score: 0.9,
                run_tag: "run1".to_string(),
            },
            TrecRun {
                query_id: "2".to_string(),
                doc_id: "doc3".to_string(),
                rank: 1,
                score: 0.95,
                run_tag: "run1".to_string(),
            },
        ];

        let grouped = group_runs_by_query(&runs);
        assert_eq!(grouped.len(), 2);
    }
}
