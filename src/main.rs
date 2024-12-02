#![no_main]
sp1_zkvm::entrypoint!(main);

use serde::{Deserialize, Serialize};
use ndarray::Array2;
use std::fmt;

// Define your models as simple structs or mock implementations
#[derive(Debug, Deserialize, Serialize)]
pub struct TestData {
    quantity: f64,
    price: f64,
    discount_applied: f64,
    // other fields ...
    amount: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MyError {
    ParseError(String),
    IoError(String),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MyError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            MyError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for MyError {}

pub fn main() {
    // Simulate input data (in a real zkVM, these would be inputs)
    let test_features = vec![
        vec![1.0, 2.0, 3.0],  // Example feature data
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let actual_amounts = vec![10.0, 20.0, 30.0];

    // Convert Vec<Vec<f64>> to Array2<f64>
    let num_samples = test_features.len();
    let num_features = test_features[0].len();
    let flat_features: Vec<f64> = test_features.into_iter().flatten().collect();
    let x: Array2<f64> = Array2::from_shape_vec((num_samples, num_features), flat_features)
        .expect("Failed to create Array2 from shape");

    // Mock Scaler transformation (simplified for zkVM)
    let x_scaled = x.clone();  // Placeholder for scaling logic

    // Mock model parameters and prediction (simplified)
    let linear_pred = mock_linear_regression(&x_scaled);
    let ridge_pred = mock_ridge_regression(&x_scaled);
    let poly_ridge_pred = mock_polynomial_ridge_regression(&x_scaled);

    // Commit predictions for zkVM using the correct commit format
    sp1_zkvm::io::commit(&linear_pred);
    sp1_zkvm::io::commit(&ridge_pred);
    sp1_zkvm::io::commit(&poly_ridge_pred);

    // Compute evaluation metrics
    let mae_linear = compute_mae(&linear_pred, &actual_amounts);
    let mse_linear = compute_mse(&linear_pred, &actual_amounts);
    let rmse_linear = mse_linear.sqrt();
    let r2_linear = compute_r2(&linear_pred, &actual_amounts);

    let mae_ridge = compute_mae(&ridge_pred, &actual_amounts);
    let mse_ridge = compute_mse(&ridge_pred, &actual_amounts);
    let rmse_ridge = mse_ridge.sqrt();
    let r2_ridge = compute_r2(&ridge_pred, &actual_amounts);

    let mae_poly_ridge = compute_mae(&poly_ridge_pred, &actual_amounts);
    let mse_poly_ridge = compute_mse(&poly_ridge_pred, &actual_amounts);
    let rmse_poly_ridge = mse_poly_ridge.sqrt();
    let r2_poly_ridge = compute_r2(&poly_ridge_pred, &actual_amounts);

    // Commit evaluation metrics for zkVM using the correct commit format
    sp1_zkvm::io::commit(&mae_linear);
    sp1_zkvm::io::commit(&mse_linear);
    sp1_zkvm::io::commit(&rmse_linear);
    sp1_zkvm::io::commit(&r2_linear);

    sp1_zkvm::io::commit(&mae_ridge);
    sp1_zkvm::io::commit(&mse_ridge);
    sp1_zkvm::io::commit(&rmse_ridge);
    sp1_zkvm::io::commit(&r2_ridge);

    sp1_zkvm::io::commit(&mae_poly_ridge);
    sp1_zkvm::io::commit(&mse_poly_ridge);
    sp1_zkvm::io::commit(&rmse_poly_ridge);
    sp1_zkvm::io::commit(&r2_poly_ridge);

    
}

// Mock prediction functions for zkVM
fn mock_linear_regression(x: &Array2<f64>) -> Vec<f64> {
    vec![1.0; x.nrows()]  // Fake prediction for zkVM
}

fn mock_ridge_regression(x: &Array2<f64>) -> Vec<f64> {
    vec![2.0; x.nrows()]  // Fake prediction for zkVM
}

fn mock_polynomial_ridge_regression(x: &Array2<f64>) -> Vec<f64> {
    vec![3.0; x.nrows()]  // Fake prediction for zkVM
}

// Evaluation metrics (these need to be computed directly in zkVM)
fn compute_mae(predictions: &[f64], actuals: &[f64]) -> f64 {
    predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).abs()).sum::<f64>() / predictions.len() as f64
}

fn compute_mse(predictions: &[f64], actuals: &[f64]) -> f64 {
    predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).powi(2)).sum::<f64>() / predictions.len() as f64
}

fn compute_r2(predictions: &[f64], actuals: &[f64]) -> f64 {
    let actual_mean = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let ss_tot: f64 = actuals.iter().map(|a| (a - actual_mean).powi(2)).sum();
    let ss_res: f64 = predictions.iter().zip(actuals.iter()).map(|(p, a)| (a - p).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}
