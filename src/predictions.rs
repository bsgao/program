// src/predictions.rs

use ndarray::prelude::*;
use crate::models::{LinearRegressionParams, RidgeRegressionParams, PolynomialRidgeRegressionParams, ScalerParams};

pub struct Scaler {
    mean: Array1<f64>,
    scale: Array1<f64>,
}

impl Scaler {
    pub fn new(params: ScalerParams) -> Self {
        Scaler {
            mean: Array1::from(params.mean),
            scale: Array1::from(params.scale),
        }
    }

    pub fn transform(&self, input: &Array2<f64>) -> Array2<f64> {
        // (X - mean) / scale
        (input - &self.mean) / &self.scale
    }
}

pub struct LinearRegressionModel {
    coefficients: Array1<f64>,
    intercept: f64,
}

impl LinearRegressionModel {
    pub fn new(params: LinearRegressionParams) -> Self {
        LinearRegressionModel {
            coefficients: Array1::from(params.coefficients),
            intercept: params.intercept
        }
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        X.dot(&self.coefficients) + self.intercept
    }
}

pub struct RidgeRegressionModel {
    coefficients: Array1<f64>,
    intercept: f64,
}

impl RidgeRegressionModel {
    pub fn new(params: RidgeRegressionParams) -> Self {
        RidgeRegressionModel {
            coefficients: Array1::from(params.coefficients),
            intercept: params.intercept
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.coefficients) + self.intercept
    }
}

pub struct PolynomialRidgeRegressionModel {
    coefficients: Array1<f64>,
    intercept: f64,
    feature_names: Vec<String>,
}

impl PolynomialRidgeRegressionModel {
    pub fn new(params: PolynomialRidgeRegressionParams) -> Self {
        PolynomialRidgeRegressionModel {
            coefficients: Array1::from(params.coefficients),
            intercept: params.intercept,
            feature_names: params.feature_names,
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        // Assuming X has been preprocessed (scaled)
        // Generate polynomial features manually
        // For degree=2, include squares and pairwise products

        let num_features = x.shape()[1];
        let mut X_poly = Array2::<f64>::zeros((x.shape()[0], self.feature_names.len()));
        
        for (i, row) in x.outer_iter().enumerate() {
            for (j, feature_name) in self.feature_names.iter().enumerate() {
                // Simple parser for feature names like 'feature1', 'feature1^2', 'feature1 feature2'
                if feature_name.contains("^2") {
                    let feature = feature_name.replace("^2", "");
                    let idx = self.feature_names.iter().position(|f| f == &feature).unwrap();
                    X_poly[[i, j]] = row[idx].powi(2);
                } else if feature_name.contains(' ') {
                    let parts: Vec<&str> = feature_name.split(' ').collect();
                    let idx1 = self.feature_names.iter().position(|f| f == parts[0]).unwrap();
                    let idx2 = self.feature_names.iter().position(|f| f == parts[1]).unwrap();
                    X_poly[[i, j]] = row[idx1] * row[idx2];
                } else {
                    let idx = self.feature_names.iter().position(|f| f == feature_name).unwrap();
                    X_poly[[i, j]] = row[idx];
                }
            }
        }

        X_poly.dot(&self.coefficients) + self.intercept
    }
}
