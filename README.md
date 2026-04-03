## Project: Algorithmic Trading and Probability

### General introduction

This project gathers several **quantitative finance Python scripts** illustrating key concepts from **probability**, **statistics**, and **integration** applied to algorithmic trading: standard probability laws, the birthday paradox, entropy, concentration inequalities, stochastic dominance, Lebesgue integration applied to payoffs, and more.  
Each module provides either a **Monte Carlo simulation** or a **graphical visualization** (using `matplotlib`), and systematically highlights a **financial interpretation** (PnL, tail risks, diversification, capital allocation…).

### Repository layout

| Folder | Role |
|--------|------|
| `1_probability_fundamentals/` | Core probability ideas: combinatorics, games, entropy, information |
| `2_distributions_and_limits/` | Discrete/continuous laws, tails, random‑matrix limit (Wigner) |
| `3_measure_and_inequalities/` | Integration (Lebesgue), concentration inequalities, stochastic orders |
| `4_quantitative_finance/` | Markets, options/Greeks, implied vol, factors, regression on finance data |
| `5_machine_learning/` | Neural nets (backprop), multivariate calculus utilities for ML |

### Overview of the modules

#### `1_probability_fundamentals/`

- **Birthday paradox** (`1_probability_fundamentals/birthday_paradox.py`): Monte Carlo simulation of the probability that two individuals in a group share the same birthday as the group size increases.
- **Blackjack and Monte Carlo** (`1_probability_fundamentals/black_jack.py`): simulation of many Blackjack hands to estimate win, loss, and push probabilities for a simple strategy.
- **Probabilistic entropy and diversification** (`1_probability_fundamentals/entropy_diversification.py`): **Shannon entropy** over market scenarios and portfolio weights.

#### `2_distributions_and_limits/`

- **Standard discrete laws** (`2_distributions_and_limits/discrete_distributions.py`): **Bernoulli**, **binomial**, **Poisson**, **geometric**; trades and order arrivals.
- **Standard continuous laws** (`2_distributions_and_limits/continuous_distributions.py`): normal, exponential, Student, Gumbel, Cauchy, etc.
- **Gaussian tail simulation** (`2_distributions_and_limits/gaussian_tail_simulation.py`): positive half of a Gaussian (truncation / tail shape).
- **Wigner semicircle law** (`2_distributions_and_limits/wigner_semicircle_law.py`): eigenvalue spectrum of large Wigner matrices vs semicircle density.

#### `3_measure_and_inequalities/`

- **Markov, Jensen, and Chebyshev** (`3_measure_and_inequalities/markov_jensen_chebyshev.py`): concentration bounds on simulated PnL and returns.
- **Lebesgue vs Riemann** (`3_measure_and_inequalities/lebesgue.py`): expectation of `max(X-K,0)` via grid integral vs Monte Carlo.
- **Stochastic dominance** (`3_measure_and_inequalities/stochastic_dominance.py`): first‑order dominance between two normal PnL‑like laws.

#### `4_quantitative_finance/`

- **Mean and variance on real data** (`4_quantitative_finance/application_var_exp.py`): daily returns of `AAPL` via `yfinance`.
- **Bose–Einstein capital allocation** (`4_quantitative_finance/bose_einstein.py`): analogy between condensation and concentrated capital.
- **Gram–Schmidt / orthogonal factors** (`4_quantitative_finance/gram_shimt_process.py`): orthogonalizing correlated return series (factor‑style intuition).
- **Newton–Raphson / implied volatility** (`4_quantitative_finance/newton_raphson.py`): root‑finding (e.g. Black–Scholes inversion).
- **Taylor series / option approximation** (`4_quantitative_finance/taylor_series.py`): Black–Scholes call and Taylor expansion in \(S\).
- **Linear regression pipeline** (`4_quantitative_finance/optimization_regression_model.py`): sklearn regression on synthetic then finance‑style features.

#### `5_machine_learning/`

- **Backpropagation demo** (`5_machine_learning/backpropagation_nn.py`): small network and option‑payoff toy (from former `back_propagation/Nn`).
- **Multivariate calculus (Jacobian / Hessian)** (`5_machine_learning/multivariate_calculus_for_macine_learning.py`): numerical derivatives for vector functions (note: file is JavaScript‑style; rename or port to Python if you standardize the repo).

---

### Birthday paradox

#### Mathematical intuition

The birthday paradox is based on **combinatorics**. For a group of \( n \) people, we compute the probability that at least two of them share the same birthday, assuming 365 possible days (non‑leap year) and **independent, equally likely** dates. We start from the complementary probability (no shared birthday), which is the product of decreasing terms, and then deduce the desired probability. As soon as the group reaches 23 people, this probability exceeds 50%, which is highly counterintuitive.

#### Link with algorithmic trading

This paradox shows how our intuition can fail in the presence of **rare but combinatorial events**, similar to **price coincidences**, **hash collisions**, or clusters of events on markets. It emphasizes the importance of reasoning in probabilities over large sets (orders, ticks, signals) rather than relying on naive intuition when assessing the likelihood of supposedly “improbable” scenarios.

#### Python implementation

The script `1_probability_fundamentals/birthday_paradox.py` performs a **Monte Carlo simulation**: for each group size from 1 to 60, it runs many trials (by default 1,000) drawing random birthdays. It estimates how often at least one coincidence occurs and plots the estimated probability as a function of group size, with a reference line at 50%.

---

### Standard discrete laws

#### Mathematical intuition

The **discrete** laws Bernoulli, binomial, Poisson, and geometric respectively model:  
- a single binary success/failure trial (Bernoulli);  
- the number of successes in a finite series of independent trials (binomial);  
- the number of events in a given time interval (Poisson);  
- the number of failures before the first success (geometric).  
They form the basis of many probabilistic models in finance and risk management.

#### Link with algorithmic trading

In algorithmic trading, these laws are used to model:  
- the outcome of a single trade (Bernoulli);  
- the distribution of the number of winning trades over a given period (binomial);  
- the random arrival of orders or “spikes” in the order book (Poisson);  
- the waiting time (in number of attempts) until a winning trade or a specific market event (geometric).  
They enable **backtesting** and **stress testing** of strategies by characterizing the variability of their results.

#### Python implementation

The script `2_distributions_and_limits/discrete_distributions.py` generates many samples from each law using `numpy.random` and displays comparative **histograms**. Comments document each law’s typical use in trading (win rate, order flow, waiting times, etc.), making the script a pedagogical bridge between discrete theory and market practice.

---

### Standard continuous laws

#### Mathematical intuition

Continuous laws such as the **normal**, **exponential**, **uniform**, **gamma**, **Student**, **Weibull**, **Laplace**, **Gumbel**, and **Cauchy** distributions provide a range of models for real‑valued variables: fluctuations around a mean, waiting times, accumulation phenomena, heavy tails, extreme values, or ultra‑erratic behavior (infinite variance).

#### Link with algorithmic trading

These laws are used to model **returns**, **inter‑trade durations**, **times to incidents** or crashes, and **extreme outcomes** (drawdowns, volatility spikes). In practice, using an inappropriate distribution (e.g., normal instead of Student or Cauchy) leads to underestimating tail risks, which is critical for **risk management** and strategy calibration.

#### Python implementation

The script `2_distributions_and_limits/continuous_distributions.py` simulates, for each law, a large number of samples and plots **normalized histograms** on a grid of subplots. Each subplot is annotated with the law’s financial role (standard returns, waiting times, accumulated risks, fat tails, extreme values, highly erratic markets, etc.).

---

### Probabilistic entropy and diversification

#### Mathematical intuition

**Shannon entropy** measures the level of **uncertainty** or **disorder** of a discrete probability distribution. It is minimal when the distribution is highly concentrated (one almost certain scenario) and maximal when all scenarios are equally likely. For a law \((p_i)_i\), entropy is defined as \(-\sum_i p_i \log_2 p_i\).

#### Link with algorithmic trading

In finance, entropy can quantify:  
- the uncertainty of **market scenarios** (up, flat, down);  
- the degree of **diversification** of a **portfolio**: a highly concentrated allocation has low entropy, whereas an equally‑weighted portfolio maximizes entropy.  
It is a conceptual tool connecting **information**, **diversification**, and **uncertainty**.

#### Python implementation

The script `1_probability_fundamentals/entropy_diversification.py` defines an entropy function, evaluates several distributions representing directional, uncertain, or intermediate markets, and then plots entropy values as bars. It also compares different portfolio allocations (all in one asset, two equal assets, diversified portfolios, concentrated portfolios), showing how entropy increases with diversification.

---

### Markov, Jensen, and Chebyshev inequalities

#### Mathematical intuition

The **Markov**, **Jensen**, and **Chebyshev** inequalities provide general bounds on the probabilities of extreme events and on expectations of convex functions.  
- Markov bounds \( \mathbb{P}(X \ge a) \) by \( \mathbb{E}[X]/a \) for \( X \ge 0 \).  
- Jensen compares \( f(\mathbb{E}[X]) \) to \( \mathbb{E}[f(X)] \) for a convex function \( f \).  
- Chebyshev bounds the probability of large deviations from the mean in terms of the variance.

#### Link with algorithmic trading

These inequalities help **bound tail risks** and “rare” events without knowing the exact distribution. For a log‑normal PnL or simulated returns, they provide simple upper bounds on the probabilities of extreme losses or gains and help reason about how **concentrated** a distribution is. They are useful in **risk management**, **conservative backtesting**, and communication of “worst‑case” style bounds.

#### Python implementation

The script `3_measure_and_inequalities/markov_jensen_chebyshev.py` simulates a log‑normal PnL and returns, computes empirical probabilities of certain events (PnL beyond a threshold, deviations from the mean), and compares them with Markov and Chebyshev bounds. It also illustrates Jensen using \( f(x) = e^x \). Several plots show the PnL distribution, the extreme region considered, and lines for the mean and multiples of the standard deviation.

---

### Lebesgue vs Riemann integral for an option payoff

#### Mathematical intuition

The **expectation** of a payoff \( f(X) \) can be viewed either as a **Riemann integral** (summing \( f(x)p(x)\,dx \) over the price axis) or as a **Lebesgue integral** (summing contributions from individual realizations of \( X \)). In probability theory, the Lebesgue integral is the natural framework for defining expectation.

#### Link with algorithmic trading

In option pricing, the expectation of the payoff \( \max(X-K, 0) \) under a given probability measure is central to **pricing** and **risk‑neutral valuation**. Understanding the equivalence between these two viewpoints (integral over the density vs Monte Carlo average) is key to justifying numerical pricing methods.

#### Python implementation

The script `3_measure_and_inequalities/lebesgue.py` simulates a Gaussian return, constructs the payoff `max(x-K, 0)`, and compares:  
- a **Riemann** approximation via a grid `x_grid` and the sum of \( f(x) p(x)\,dx \) (using `scipy.stats.norm`);  
- a **Lebesgue** approximation via the Monte Carlo mean `np.mean(payoff)`.  
Plots display the normal density, the weighted payoff area, and the sample contributions to the Lebesgue integral.

---

### Stochastic dominance

#### Mathematical intuition

A random variable \( Y \) **stochastically dominates** \( X \) (first order) if, for every \( x \), \( F_X(x) \ge F_Y(x) \). Intuitively, \( Y \) delivers more high values than \( X \) and is thus “better” in terms of risk‑return for any investor who prefers more to less.

#### Link with algorithmic trading

When comparing strategies or portfolios, stochastic dominance allows us to compare PnL distributions: if one strategy dominates another, it is preferred by any **risk‑averse** investor with a standard utility. It is an important theoretical tool for **strategy ranking** and decision theory.

#### Python implementation

The script `3_measure_and_inequalities/stochastic_dominance.py` simulates two normal laws \( X \sim \mathcal{N}(0,1) \) and \( Y \sim \mathcal{N}(1,1) \), computes their **empirical distribution functions**, and plots them on the same axes. It also shows empirical densities to highlight the rightward shift of \( Y \), numerically illustrating stochastic dominance.

---

### Wigner semicircle law

#### Mathematical intuition

In **random matrix theory**, the spectrum (set of eigenvalues) of large symmetric matrices with Gaussian entries, once normalized, converges to the **Wigner semicircle law**. Its density is supported on a compact interval and has a half‑circle shape.

#### Link with algorithmic trading

Random matrices arise in **portfolio risk management** (correlation/covariance matrices of many assets). Understanding the typical spectrum of such matrices helps separate **random noise** from **structural factors** (significant extreme eigenvalues), which is crucial for dimensionality reduction and managing multi‑asset portfolios.

#### Python implementation

The script `2_distributions_and_limits/wigner_semicircle_law.py` generates a Wigner matrix (symmetric with Gaussian entries), computes its eigenvalues, normalizes them, and plots the empirical histogram together with the theoretical semicircle density. It illustrates the convergence of the spectrum to this law as the matrix dimension grows.

---

### Gaussian tail simulation

#### Mathematical intuition

A **Gaussian tail** is the truncated part of a normal distribution, e.g. \( X \sim \mathcal{N}(0,1) \) conditioned on \( X > 0 \). It models phenomena that can only take positive values but inherit a Gaussian‑like shape over their support.

#### Link with algorithmic trading

Such variables can model **magnitudes** of positive moves, strictly positive times, or other quantities naturally bounded below by zero. Visualizing the Gaussian tail helps understand how **truncation** alters the distribution and thus risk evaluation.

#### Python implementation

The script `2_distributions_and_limits/gaussian_tail_simulation.py` simulates a standard Gaussian, filters strictly positive values, and plots the normalized histogram of the right half, highlighting the shape of the tail.

---

### Blackjack and Monte Carlo simulation

#### Mathematical intuition

Blackjack is a **probabilistic** game where the player faces the dealer under simple drawing rules. Its structure naturally lends itself to **Monte Carlo simulation** to estimate probabilities of winning, losing, or pushing under various strategies.

#### Link with algorithmic trading

Modeling this game is a good sandbox for testing ideas about **risk management**, **decision rules** (hit or stand), and illustrating how simple rules generate complex outcome distributions. It provides intuition close to **trading systems** (position sizing, stop‑loss rules, mechanical strategies).

#### Python implementation

The script `1_probability_fundamentals/black_jack.py` defines a hand representation, a simple strategy for both player and dealer, and then runs many hands (by default 100,000). It returns the observed frequencies of wins, losses, and pushes for the player, giving an empirical estimate of the house edge for this naive strategy.

---

### Mean and variance on real data

#### Mathematical intuition

**Mean** and **variance** are, respectively, the average and the dispersion measure of a series of returns. On real market data, their estimation quantifies an asset’s **average gain** and **risk**.

#### Link with algorithmic trading

These quantities underpin **portfolio management** (risk/return ratio, volatility), **indicator construction**, and model calibration (e.g., constant variance vs stochastic volatility). They are also the starting point for risk‑adjusted performance measures.

#### Python implementation

The script `4_quantitative_finance/application_var_exp.py` downloads daily prices for the `AAPL` stock via `yfinance`, computes **daily returns**, and then their mean and variance using `numpy`. Results are displayed in percentage terms for direct interpretation.

---

### Capital allocation and Bose–Einstein condensation

#### Mathematical intuition

The **Bose–Einstein** distribution describes the allocation of indistinguishable particles over energy levels, with possible **condensation** on the ground state. By analogy, a given amount of capital can concentrate on a few highly attractive companies.

#### Link with algorithmic trading

This module shows how “increasing attractiveness” mechanisms can lead to **strong capital concentration** in a handful of names (superstar effect). It offers a useful analogy for thinking about sector concentration, asset bubbles, and systemic fragility.

#### Python implementation

The script `4_quantitative_finance/bose_einstein.py` defines a set of companies with increasing **attractiveness**, computes a Bose–Einstein‑type population over their energy levels, then normalizes it to a given total capital. It plots a bar chart showing the capital share allocated to each company.

---

### How to run the scripts

#### Prerequisites

- **Python 3.x**
- Python libraries: `numpy`, `matplotlib`, `yfinance`, `scipy`, `scikit-learn` (for regression / some demos)

#### Installing dependencies

From the project directory:

```bash
pip install numpy matplotlib yfinance scipy scikit-learn
```

#### Example commands

Run scripts from the **project root** (`Trading Algo project /`) so that paths resolve as below:

```bash
# --- 1_probability_fundamentals ---
python 1_probability_fundamentals/birthday_paradox.py
python 1_probability_fundamentals/black_jack.py
python 1_probability_fundamentals/entropy_diversification.py

# --- 2_distributions_and_limits ---
python 2_distributions_and_limits/discrete_distributions.py
python 2_distributions_and_limits/continuous_distributions.py
python 2_distributions_and_limits/gaussian_tail_simulation.py
python 2_distributions_and_limits/wigner_semicircle_law.py

# --- 3_measure_and_inequalities ---
python 3_measure_and_inequalities/markov_jensen_chebyshev.py
python 3_measure_and_inequalities/lebesgue.py
python 3_measure_and_inequalities/stochastic_dominance.py

# --- 4_quantitative_finance ---
python 4_quantitative_finance/application_var_exp.py
python 4_quantitative_finance/bose_einstein.py
python 4_quantitative_finance/gram_shimt_process.py
python 4_quantitative_finance/newton_raphson.py
python 4_quantitative_finance/taylor_series.py
python 4_quantitative_finance/optimization_regression_model.py

# --- 5_machine_learning ---
python 5_machine_learning/backpropagation_nn.py
```

All Python modules use the `.py` extension. Only `README.md` sits at the repository root beside the numbered folders.
