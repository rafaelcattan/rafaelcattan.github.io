---
title: "US Unemployemnt Forecasting: a use-case"
date: 2025-12-05
classes: wide  
layout: single
share: false
related: false
usemathjax: true    
---

In this post I'll explore a time-series forecasting problem. This is not only a classic data-science problem, but also an economics one. Whereas in the first, companies are interested in forecasting sales, conversions, net revenue, or any other relevant KPI, in economics it is often related to macroeconomic series.

Since it is easier to access public API-based macroeconomic data-sets I'll stick with the second.

In the post I'll adress the following question: can we forecast the US unemployment rate? 

First things first, lets dive into the data. I'm using US monthly unemployment rate from 2014-12-16 until 2024-12-01 (that's all I've got in free version of the [Bureau Of Labour and Statistics ](https://api.bls.gov)), making only 240 observations.

## Some Context

Before any methodological overview, lets have quick look at the time series under analysis:

<p align="center">
  <img src="/assets/images/forecasting/us_unemployment_with_ci.png" alt="Forecast Results" width="800">
</p>

Besides the time series itself I have added its confidence interval, a rolling 12-month average (more useful for high-volatility series), and two distinct periods that diserve some attention: the 2008 financial crash and the 2020 Covid pandemics.

Since these events are not endogenous economic events, there is a good chance that our model will fail badly in those periods.

Despite these shocks, 80% of the observations fall between a 3.4-7.7% range, with a 5.8% mean and a 2.12 standard deviation. Meaning that under "normal periods" the serie is relatively stable.

Another interesting observation is how unemployment behaved differently across the two periods. First the 2008 financial crisis took longer to reach its maximum level and more so to converge back to the period's average. The Covid shock, on the other hand, halved ocuppations extremely fast, but was also faster to converge. We can better illustrate this with the following plot:


<p align="center">
  <img src="/assets/images/forecasting/crisis_comparison.png" alt="" width="800">
</p>


We can see the GFC takes impressive 82 months from its assumed beggining (2007-12-01) until it converges to the period mean (2014-09-01). This means a 9-year sluggish recover. The Covid pandemics, on its turn, takes approximately 15 months from its beggining, in February 2020, until it converges to the period mean again, in May 2021, despite hitting a much higher unemployment level of nearly 15%.

In terms of forecastability, hence, the period shows a double challenge, not only two massive shocks, but two very distinct patterns. Having contextualized the the unemployment rate during the period of data availabilty, we should ask ourselves: **is the US unemployment rate between (2014/12 - 2024/12) predictable?**


## Measuring Predictability

One of the most standard ways of measuring how predicatable a time-series is through Coefficient of Variation (CoV). The idea is simple: compare the series' standard deviation to its mean: $$CoV = \frac{\sigma}{\mu}$$. A value smaller than 0.5 should be considerd relatively smooth, $0.5<CoV<1$ can be considered unstable, whereas a $CoV>1$ can be considered quite unstable.

For our series, this ratio if of about 0.36. This metric, however, is quite poor since it does not grasp any time-dependece of the series. That is, the data distribution is simply considered as independent, with no attention to time-specific dimensions, such as seasonality, trend, or even order itself. A simple look at the series can give us an idea of how  $CoV$ can be misleading.

A hands-on approach that considers time-specific structure is the Mean Absolute Scaled Error, formaly defined as : 

$$
\text{MASE} =
\frac{\frac{1}{n}\sum_{t=1}^{n} \lvert y_t - \hat{y}_t \rvert}
{\frac{1}{n-1}\sum_{t=2}^{n} \lvert y_t - y_{t-1} \rvert}
$$

Simply put, this method highlights *how better is my model against the last observed value?*

Since $y_t$ and $y_{t-1}$ are given, what we need to estimate is $\hat{y}_t$. One approach is to use a naive basis, such as the mean value of the series, the mean of the n past values, or as simple univariate estimate, like ARIMA. Estimating theses values for our problem I cound find:

| Model | MASE |
|:-------|:------:|
| Naive Estimator (Overall Mean) | 8.8 |
| Naive Estimator (Past 6 Months Mean) | 1.5 |
| ARIMA | 1.13 |

It is interesting to notice that, since MASE values >1 are assumed to be tricky, we confirm that out problem is not an easy one. 

Second, it is iteresting to notice how the 6 month mean performed much better than the overall mean. This suggests that our series is time-dependent, meaning that past values influence future values.

A common time-series diagnosis for such patter is to use the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). They measure the effect of $y_{t-k}$ on $y_{t}$. The first disregards the intermediate effects of $t-k-1$ events, whereas the second controls for each individual time effect between past and current events, as shown below:


<p align="center">
  <img src="/assets/images/forecasting/pacf_pac.png" alt="" width="800">
</p>

We can see that the the ACF decreases exponentially, whereas the PACF has a cut-off after the second lag. This indicates an AR(2) process: the series is time dependent and the effect of lags greater than 2 are rather small.

So our series is auto-correlated and its one hard to estimate according to different MASE metrics. 


## Predicting


Before picking a standard method and running .fit() lets first understand better the problem. A common starting point is to decompose the time series into trend, seasonality, and residuals. The [statsmodel](https://www.statsmodels.org/stable/index.html) lib provides a built-in method for this called "seasonal_decompose" which fits a simple model $Y_t = T_t + S_t + e_t$. All three elements can be seen below:


<p align="center">
  <img src="/assets/images/forecasting/seasonal_decomposition.png" alt="" width="800">
</p>

As previously observed the trend itself is quite erratic - but due to two very meaningful shocks. The seasonality has been well grasped. Intuitively I've calculated the time between the serie's nineth decile ($q=.9$) and they returned an yearly seasonality:

```python
# Get seasonal component values with absolute value > 90th percentile
high_seasonal = decomposition.seasonal[abs(decomposition.seasonal) > decomposition.seasonal.abs().quantile(0.9)]    

# Calculate time differences between these dates
time_diffs = high_seasonal.index.to_series().diff() 
time_diffs.mean()

Timedelta('365 days 06:18:56.842105264')

``` 

Which has been further corroborated by a standard a Fast Fourier Transform:


```python
from scipy.fft import rfft, rfftfreq

yf = rfft(decomposition.seasonal.values - decomposition.seasonal.values.mean())
xf = rfftfreq(len(df), d=1) 

# Finding highest power
idx = np.argmax(np.abs(yf))
dominant_period = 1 / xf[idx]
12.0
``` 

Wrapping up what we have seen so far we can say that our series is time-dependent, non-stationary (from the exponential decay of the PAC plot), has a yearly seasonality, and suffered two strong but distinct shocks during the period of analysis. With that in mind we'll explore three different approaches taking that information into account.

## Fitting

When tackling a macro-economic indicator like the US unemployment rate (2005–2025), choosing a model is less about "which is better" and more about "which mathematical assumptions do we trust?" Having considered the nature of our problem, we will compare three different approaches.

First things, first: The **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) model. This method treats the time series as a linear stochastic process and is a kind of go-to method when you want to have a first predictability idea. Methodologically, it relies on the assumption that the future is a linear combination of past observations and past errors, be them explained by the series itself or by its seasonality.

The specific identification of our SARIMA model, **$(0,2,1) \times (0,0,1)_{12}$** (via the AIC criterion), corresponds to a specific expansion of the lag operator. The equation we are fitting to the US unemployment data is summarized as:

$$(1-L)^2 y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \Theta_1 \epsilon_{t-12} + \theta_1 \Theta_1 \epsilon_{t-13}$$


* **Second-Order Differencing $(1-L)^2$**: This term represents the "acceleration" of the series. Mathematically, it expands to $y_t - 2y_{t-1} + y_{t-2}$, which effectively removes non-linear stochastic trends from the unemployment rate.

* **Short-Term Error ($\theta_1 \epsilon_{t-1}$)**: This captures the impact of the residual (shock) from the immediate previous month.

* **Seasonal Error ($\Theta_1 \epsilon_{t-12} a$)** and ($\theta_1 \Theta_1 \epsilon_{t-13}$)**: Which identifies the seasonal residual from 12 and 13 months ago, allowing the model to correct for annual cycles.

So, in this framework, the current "accelerated" change in unemployment is explained not by past values themselves, but by a combination of recent shocks ($\theta$) and yearly seasonal residuals ($\Theta$).


In contrast to the regressive nature of SARIMA, **Prophet** views forecasting as a curve-fitting exercise. Its methodology is built on a **Generalized Additive Model (GAM)**. Rather than looking for autocorrelation, it decomposes the signal into distinct structural components. Its functional form is:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$


While the components are **summed** together (making it additive), the individual components are non-linear: the trend can be fitted as a logistic growth curve or a growth-rate adjusted at given states. Whereas the seasonality is a **Fourier Series** which by definition is not linear.

If this strategy allows it to handle irregular spacing and structural breaks—like economic crises—more better, the model still departures from a given functional form, meaning that the algorithm effort is to find the best parameters that match that specific functional form.

Our third model specification, on the other hand, starts from a different problem: given the data, which functional form fits best?

The **RandomForestRegressor** shifts the paradigm from temporal sequences to a supervised learning problem. Since decision trees are inherently "time-blind," the methodology requires manual feature engineering to create a lag-matrix ($y_{t-1}, y_{t-2}, \dots$). So basically it excels at capturing non-linear interactions between lags—something SARIMA cannot do—but it lacks an internal mechanism to handle trends (extrapolation). Its functional form is an average of $B$ individual tree predictions:

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Where:
* $T_b(x)$ is the output of a single decision tree grown on a bootstrap sample.
* $x$ is the input vector of lagged features.

```python
from sklearn.ensemble import RandomForestRegressor

# Example: Converting the series to a supervised problem
# target = y_t, features = [y_{t-1}, y_{t-2}, y_{t-3}]
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10)
model_rf.fit(X_train_lags, y_train)