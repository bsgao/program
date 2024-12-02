// src/models.rs

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct LinearRegressionParams {
    pub coefficients: Vec<f64>,
    pub intercept: f64// intercept is a single value but using Vec for consistency
}

#[derive(Debug, Deserialize)]
pub struct RidgeRegressionParams {
    pub coefficients: Vec<f64>,
    pub intercept: f64
}

#[derive(Debug, Deserialize)]
pub struct PolynomialRidgeRegressionParams {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ScalerParams {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct TestData {
    pub quantity: f64,
    pub price: f64,
    pub discount_applied: f64,
    pub IsAL: f64,
    pub IsAK: f64,
    pub IsAZ: f64,
    pub IsAR: f64,
    pub IsCA: f64,
    pub IsCO: f64,
    pub IsCT: f64,
    pub IsDE: f64,
    pub IsFL: f64,
    pub IsGA: f64,
    pub IsHI: f64,
    pub IsID: f64,
    pub IsIL: f64,
    pub IsIN: f64,
    pub IsIA: f64,
    pub IsKS: f64,
    pub IsKY: f64,
    pub IsLA: f64,
    pub IsME: f64,
    pub IsMD: f64,
    pub IsMA: f64,
    pub IsMI: f64,
    pub IsMN: f64,
    pub IsMS: f64,
    pub IsMO: f64,
    pub IsMT: f64,
    pub IsNE: f64,
    pub IsNV: f64,
    pub IsNH: f64,
    pub IsNJ: f64,
    pub IsNM: f64,
    pub IsNY: f64,
    pub IsNC: f64,
    pub IsND: f64,
    pub IsOH: f64,
    pub IsOK: f64,
    pub IsOR: f64,
    pub IsPA: f64,
    pub IsRI: f64,
    pub IsSC: f64,
    pub IsSD: f64,
    pub IsTN: f64,
    pub IsTX: f64,
    pub IsUT: f64,
    pub IsVT: f64,
    pub IsVA: f64,
    pub IsWA: f64,
    pub IsWV: f64,
    pub IsWI: f64,
    pub IsWY: f64,
    pub IsCash: f64,
    pub IsPayPal: f64,
    pub IsDebitCard: f64,
    pub IsCreditCard: f64,
    pub IsBooks: f64,
    pub IsHomeDecor: f64,
    pub IsElectronics: f64,
    pub IsClothing: f64,
    pub amount: f64, // Target variable
}


