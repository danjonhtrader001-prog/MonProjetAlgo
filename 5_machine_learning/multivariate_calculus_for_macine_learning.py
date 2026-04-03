// Script for calculating the Jacobian and Hessian of matrix-valued functions

// This script uses math.js for matrix operations. Please ensure it is available in your environment.

function jacobian(f, x, h = 1e-5) {
    // f: function accepting x (array or math.js vector), returning array or vector.
    // x: point at which to evaluate Jacobian (array or math.js vector)
    // h: step for finite differencing
    const n = x.length;
    const y = f(x);
    const m = Array.isArray(y) ? y.length : y.size()[0]; // Support array or math.js output
    const J = [];
    for (let i = 0; i < m; i++) {
        J.push([]);
    }
    for (let j = 0; j < n; j++) {
        const xph = x.slice();
        xph[j] += h;
        const xmh = x.slice();
        xmh[j] -= h;
        const fph = f(xph);
        const fmh = f(xmh);
        for (let i = 0; i < m; i++) {
            J[i][j] = (fph[i] - fmh[i]) / (2 * h);
        }
    }
    return J;
}

function hessian(f, x, h = 1e-5) {
    // f: function accepting x (array or math.js vector), returning scalar
    // x: point at which to evaluate Hessian (array)
    // h: step for finite differencing
    const n = x.length;
    const H = math.zeros(n, n)._data;  // Use math.js zeros, or fallback if unavailable
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < n; ++j) {
            const x_pp = x.slice(), x_pm = x.slice(), x_mp = x.slice(), x_mm = x.slice();
            x_pp[i] += h; x_pp[j] += h;
            x_pm[i] += h; x_pm[j] -= h;
            x_mp[i] -= h; x_mp[j] += h;
            x_mm[i] -= h; x_mm[j] -= h;
            H[i][j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h);
        }
    }
    return H;
}

// Example usage (uncomment below):
/*
const math = require('mathjs');

// Example 1: Jacobian of vector function
function fvec(x) {
    // f(x) = [x1^2, sin(x2)]
    return [x[0]*x[0], Math.sin(x[1])];
}

console.log("Jacobian at [2, π/4]:", jacobian(fvec, [2, Math.PI/4]));

// Example 2: Hessian of scalar function
function fscal(x) {
    // f(x) = x1^2 + 3*x1*x2 + x2^2
    return x[0]*x[0] + 3*x[0]*x[1] + x[1]*x[1];
}
// Example: Using Jacobian, Hessian, and multivariate calculus rules in algorithmic trading

const math = require('mathjs');

// 1. Jacobian: Sensitivity of a custom portfolio to underlying factors (factor-model risk)
function factorModel(weights) {
    // Simulate a portfolio as a function of two factors (e.g., Market and Volatility)
    // Portfolio value = 2*w1*Market + 3*w2*Vol + 0.5*w1*w2*Market*Vol
    // Here, weights[0] = exposure to Market, weights[1] = exposure to Volatility
    const Market = 1.5;
    const Vol = 0.7;
    return [
        2*weights[0]*Market + 3*weights[1]*Vol + 0.5*weights[0]*weights[1]*Market*Vol
    ];
}
const weights = [0.8, 0.6];
console.log("Jacobian (Portfolio sensitivity to weights):", jacobian(factorModel, weights));
// The Jacobian tells you how the portfolio changes when you change exposure to underlying risk factors.

// 2. Hessian: Curvature of a loss function, useful in risk or optimization (e.g., quadratic utility/risk)
function portfolioVariance(x) {
    // x: weights of assets
    // Covariance matrix of 2-asset universe (toy example)
    const Sigma = [
        [0.01, 0.002],
        [0.002, 0.02]
    ];
    // Portfolio variance: x'Σx
    return math.multiply(math.multiply([x], Sigma), math.transpose([x]))[0][0];
}
console.log("Hessian (portfolio variance curvature):", hessian(portfolioVariance, [0.7, 0.3]));
// The Hessian here is the covariance matrix: key to risk management and quadratic programming in trading.

// 3. Multivariate Chain Rule: Propagation of derivatives through a market impact model
// E.g., final trading cost as a function of algo parameters via market impact and volatility
function marketImpact(x) {
    // x[0]: order size (Q), x[1]: volatility (σ)
    // Model: Impact = k * Q^alpha * σ^beta
    const k = 0.1, alpha = 0.7, beta = 0.5;
    return [ k * Math.pow(x[0], alpha) * Math.pow(x[1], beta) ];
}
const Q_sigma = [1e5, 0.02];
console.log("Gradient (cost sensitivity to size, vol):", jacobian(marketImpact, Q_sigma));
// A trader could use this to optimize order sizing dynamically with real-time volatility estimates.

// Hessian of the same (nonlinear) market impact model
console.log("Hessian (impact curvature):", hessian((x) => marketImpact(x)[0], Q_sigma));


console.log("Hessian at [1, 1]:", hessian(fscal, [1,1]));
*/