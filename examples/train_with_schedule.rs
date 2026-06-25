//! Training a ranker with `descend`'s LR schedule and EMA.
//!
//! `rankit` provides differentiable ranking losses but no optimizer; `descend`
//! provides the training-loop infrastructure (warmup/cosine learning-rate
//! schedules, EMA weight averaging, optimizers). This composes them: a
//! gradient-descent loop minimizing rankit's RankNet loss while descend's
//! `WarmupCosine` schedule drives the learning rate and a `WeightAverager`
//! tracks an averaged parameter copy. It is the loop a trainer should reuse
//! rather than hand-roll a constant learning rate.
//!
//! Run: `cargo run --example train_with_schedule`

use descend::ema::WeightAverager;
use descend::schedule::{LrSchedule, WarmupCosine};
use rankit::losses::ranknet_loss;

fn main() {
    // Eight documents with target relevance; the ranker's scores start at zero.
    let relevance = vec![3.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0, 1.0];
    let n = relevance.len();
    let mut params = vec![0.0f32; n];

    let total_steps = 200usize;
    let warmup = 20usize;
    let schedule = WarmupCosine {
        warmup_steps: warmup,
        total_steps,
        eta_min: 0.0,
    };
    let base_lr = 0.5f32;
    let mut ema = WeightAverager::new(0.9);

    let loss =
        |p: &[f32]| ranknet_loss(&p.iter().map(|&x| x as f64).collect::<Vec<_>>(), &relevance);

    let initial_loss = loss(&params);

    for step in 0..total_steps {
        // Finite-difference gradient of the ranking loss (rankit gives the loss;
        // a real trainer would supply analytic gradients).
        let base = loss(&params);
        let eps = 1e-3f32;
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let mut perturbed = params.clone();
            perturbed[i] += eps;
            grad[i] = ((loss(&perturbed) - base) / eps as f64) as f32;
        }
        // descend's schedule sets the learning rate for this step.
        let lr = schedule.lr_at(step, base_lr);
        for i in 0..n {
            params[i] -= lr * grad[i];
        }
        ema.update(&params);
    }

    let final_loss = loss(&params);
    let ema_loss = loss(ema.get());
    println!("RankNet loss: {initial_loss:.4} -> {final_loss:.4} over {total_steps} steps");
    println!("EMA-averaged model loss: {ema_loss:.4}");
    println!(
        "schedule LR: warmup-start={:.4}, peak={:.4}, end={:.4}",
        schedule.lr_at(0, base_lr),
        schedule.lr_at(warmup, base_lr),
        schedule.lr_at(total_steps - 1, base_lr)
    );

    // Training reduces the loss...
    assert!(
        final_loss < initial_loss,
        "training with the descend schedule should reduce RankNet loss"
    );
    // ...and the schedule ramps up during warmup, then cosine-decays.
    assert!(
        schedule.lr_at(0, base_lr) < schedule.lr_at(warmup, base_lr),
        "warmup: LR ramps up to the peak"
    );
    assert!(
        schedule.lr_at(total_steps - 1, base_lr) < schedule.lr_at(warmup, base_lr),
        "cosine: LR decays after the peak"
    );
    println!("  [PASS] loss decreased and the warmup-cosine schedule shaped the LR");
}
