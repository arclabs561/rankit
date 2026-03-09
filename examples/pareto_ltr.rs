//! Multi-objective Learning to Rank via Pareto frontier.
//!
//! Demonstrates using pare's Pareto frontier to find ranker configurations
//! that are non-dominated across multiple objectives (NDCG, diversity, fairness).
//!
//! Run: cargo run --example pareto_ltr --features eval

use pare::{Direction, ParetoFrontier};

fn main() {
    println!("=== Multi-Objective LTR via Pareto Frontier ===\n");

    // Simulated ranker configurations: (temperature, regularization)
    // Each produces different NDCG, diversity, and fairness scores.
    let configs: Vec<(f64, f64)> = vec![
        (0.1, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (0.1, 0.1),
        (0.5, 0.1),
        (1.0, 0.1),
        (2.0, 0.1),
        (0.1, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (2.0, 0.5),
    ];

    // Simulate evaluation: each config produces (NDCG, diversity, fairness)
    let objectives: Vec<Vec<f64>> = configs
        .iter()
        .map(|&(temp, reg)| {
            // NDCG: low temperature + low regularization = high NDCG
            let ndcg = 1.0 / (1.0 + temp) * (1.0 - 0.3 * reg);
            // Diversity: higher temperature = more diverse (softened ranking)
            let diversity = 1.0 - 1.0 / (1.0 + temp) + 0.2 * reg;
            // Fairness: regularization helps fairness, extreme temps hurt
            let fairness = 0.5 + 0.4 * reg - 0.1 * (temp - 1.0).abs();
            vec![ndcg, diversity, fairness]
        })
        .collect();

    // Build Pareto frontier (maximize all three)
    let mut frontier = ParetoFrontier::new(vec![
        Direction::Maximize,
        Direction::Maximize,
        Direction::Maximize,
    ]);

    for (i, obj) in objectives.iter().enumerate() {
        let on_frontier = frontier.push(obj.clone(), i);
        let (temp, reg) = configs[i];
        println!(
            "  config(temp={temp:.1}, reg={reg:.1}) -> NDCG={:.3} div={:.3} fair={:.3} {}",
            obj[0],
            obj[1],
            obj[2],
            if on_frontier { "<- Pareto" } else { "" }
        );
    }

    println!("\n--- Pareto-optimal configurations ---\n");

    let crowding = frontier.crowding_distances();

    for (rank, (point, &cd)) in frontier.points().iter().zip(crowding.iter()).enumerate() {
        let idx = point.data;
        let (temp, reg) = configs[idx];
        println!(
            "  #{}: temp={:.1} reg={:.1} | NDCG={:.3} div={:.3} fair={:.3} | crowding={:.3}",
            rank + 1,
            temp,
            reg,
            point.values[0],
            point.values[1],
            point.values[2],
            cd
        );
    }

    println!("\nPareto frontier size: {}", frontier.len());
    println!("Total configurations evaluated: {}", configs.len());

    // Find best config for different weight preferences
    println!("\n--- Best config by preference ---\n");

    let preferences = [
        ("NDCG-focused", vec![0.8, 0.1, 0.1]),
        ("Balanced", vec![0.33, 0.33, 0.34]),
        ("Diversity-focused", vec![0.1, 0.8, 0.1]),
        ("Fairness-focused", vec![0.1, 0.1, 0.8]),
    ];

    for (name, weights) in &preferences {
        if let Some(best_idx) = frontier.best_index(weights) {
            let point = &frontier.points()[best_idx];
            let cfg_idx = point.data;
            let (temp, reg) = configs[cfg_idx];
            println!(
                "  {name}: temp={temp:.1} reg={reg:.1} (NDCG={:.3} div={:.3} fair={:.3})",
                point.values[0], point.values[1], point.values[2]
            );
        }
    }
}
