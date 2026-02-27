---
title: "US Unemployment Forecasting: a use-case"
date: 2025-12-05
classes: wide  
layout: single
share: false
related: false
usemathjax: true    
---

In this post I'll explore a time-series forecasting problem. This is not only a classic data-science problem, but also an economic one. Whereas in the first, companies are interested in forecasting sales, conversions, net revenue, or any other relevant KPI, in economics it is often related to macroeconomic series.

Since it is easier to access public API-based macroeconomic datasets, I'll focus on the economic problem.

In this post I'll address the following question: can we forecast the US unemployment rate?

More than simply trying to find the best performance metric, this post will explore some of the nuances behind time-series forecasting. For instance, how "shocks" affect your prediction? How do pre-defined forecasting functions perform against model-agnostic algorithms? How to measure predictability?



## Some Context

First things first, let's dive into the data. I'm using the US monthly unemployment rate from 2014-12-16 to 2024-12-01 (that's all I've got in the free version of the [Bureau of Labor Statistics](https://api.bls.gov)), resulting in only 240 observations.

Before any methodological overview, let's have a quick look at the time series under analysis:

<p align="center">
  <img src="/assets/images/forecasting/us_unemployment_with_ci.png" alt="Forecast Results" width="800">
</p>

In addition to the time series, I added its confidence interval, a 12‑month rolling average (especially useful for high‑volatility series), and two notable periods: the 2008 Great Financial Crisis (GFC) and the 2020 COVID‑19 pandemic.
Since these events are not endogenous economic events, there is a good chance that our model will fail badly in those periods.

Despite these shocks, 80% of the observations fall between a 3.4-7.7% range, with a 5.8% mean and a standard deviation of 2.12, meaning that under "normal periods" the series is relatively stable.

Another interesting observation is how unemployment behaved differently across the two periods. First, the 2008 financial crisis took longer to reach its maximum level and longer to converge back to the period average. The COVID-19 shock, on the other hand, halved employment extremely fast, but was also faster to converge. We can better illustrate this with the following plot:


<p align="center">
  <img src="/assets/images/forecasting/crisis_comparison.png" alt="" width="800">
</p>


We can see the GFC takes an impressive 82 months from its assumed beginning (2007-12-01) until it converges to the period mean (2014-09-01). This means a 9-year sluggish recovery. The COVID-19 pandemic, in turn, takes approximately 15 months from its beginning in February 2020 to converge to the period mean again in May 2021, despite hitting a much higher unemployment level of nearly 15%.

In terms of forecastability, thus, the period shows a double challenge: not only two massive shocks but also two very distinct patterns. Having contextualized the unemployment rate during the period of data availability, we should ask ourselves: **is the US unemployment rate between (2014/12 - 2024/12) predictable?**


## Measuring Predictability

One of the most standard ways to measure the predictability of a time-series is through the Coefficient of Variation (CoV). The idea is simple: compare the series' standard deviation to its mean: $$CoV = \frac{\sigma}{\mu}$$. A value smaller than 0.5 indicates relative smoothness; $0.5<CoV<1$ indicates instability; and $CoV>1$ indicates high instability.

For our series, this ratio is about 0.36. This metric, however, is quite poor since it does not capture any time-dependence of the series. That is, the data distribution is simply considered as independent, with no attention to time-dependent dimensions, such as seasonality, trend, or even order itself. A simple look at the series can give us an idea of how $CoV$ can be misleading.

A hands-on approach that considers time-specific structure is the Mean Absolute Scaled Error, formally defined as:

$$
\text{MASE} =
\frac{\frac{1}{n}\sum_{t=1}^{n} \lvert y_t - \hat{y}_t \rvert}
{\frac{1}{n-1}\sum_{t=2}^{n} \lvert y_t - y_{t-1} \rvert}
$$

Simply put, this method highlights *how much better my model is compared to the last observed value.*

Since $y_t$ and $y_{t-1}$ are given, what we need to estimate is $\hat{y}_t$. One approach is to use a naive basis, such as the mean value of the series, the mean of the n past values, or a simple univariate estimate, like ARIMA. For our problem, I estimated these values and found:

| Model | MASE |
|:-------|:------:|
| Naive Estimator (Overall Mean) | 8.8 |
| Naive Estimator (Past 6 Months Mean) | 1.5 |
| ARIMA | 1.13 |

It is interesting to note that, since MASE values >1 represent high-volatility series, this confirms not only that CV can mislead you but that our problem is not an easy one.

Secondly, it is interesting to observe that the 6-month mean performed much better than the overall mean. This suggests that our series is time-dependent, meaning that past values influence future values. Since this is key for a good model proposition, we need to understand this dependence on time.

A common time-series diagnosis for such a pattern is to use the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). They measure the effect of $y_{t-k}$ on $y_t$. The first disregards the intermediate effects of $y_{t-k-1}$ events, whereas the second controls for each individual time effect between past and current events, as shown below:


<p align="center">
  <img src="/assets/images/forecasting/pacf_pac.png" alt="" width="800">
</p>

We can see that the ACF decreases exponentially, whereas the PACF has a cut-off after the second lag. This indicates an AR(2) process: the series is time-dependent and the effects of lags greater than 2 are rather small.

So our series is auto-correlated and difficult to estimate, as indicated by different MASE metrics. But how bad can that be? Since this is an exploratory exercise we will accept the challenge!



## Predicting

Before picking a standard method and running .fit(), let's first better understand the problem. A common starting point is to decompose the time series into trend, seasonality, and residuals. The [statsmodel](https://www.statsmodels.org/stable/index.html) library provides a built-in method for this called "seasonal_decompose" which fits a simple model $Y_t = T_t + S_t + e_t$. All three elements can be seen below:


<p align="center">
  <img src="/assets/images/forecasting/seasonal_decomposition.png" alt="" width="800">
</p>

As previously observed, the trend itself is quite erratic, but due to two very significant shocks. The seasonality has been well captured. Intuitively, I calculated the time between peaks of the seasonal component (using the ninth decile $q=.9$ threshold), which indicated a yearly seasonality:

```python
# Get seasonal component values with absolute value > 90th percentile
high_seasonal = decomposition.seasonal[abs(decomposition.seasonal) > decomposition.seasonal.abs().quantile(0.9)]    

# Calculate time differences between these dates
time_diffs = high_seasonal.index.to_series().diff() 
time_diffs.mean()

Timedelta('365 days 06:18:56.842105264')

``` 

Which has been further corroborated by a standard Fast Fourier Transform:


```python
from scipy.fft import rfft, rfftfreq

yf = rfft(decomposition.seasonal.values - decomposition.seasonal.values.mean())
xf = rfftfreq(len(df), d=1) 

# Finding highest power
idx = np.argmax(np.abs(yf))
dominant_period = 1 / xf[idx]
12.0
``` 

Wrapping up what we have seen so far, we can say that our series is time-dependent, non-stationary (from the exponential decay of the PACF plot), has a yearly seasonality, and suffered two strong but distinct shocks during the period of analysis. With that in mind, we'll explore three different approaches.


### The Methodological Approach

When tackling a macroeconomic indicator like the US unemployment rate (2005–2025), choosing a model is less about "which is better" and more about "which mathematical assumptions better represent the data generation process of this series?" Having considered the nature of our problem, we will compare three different approaches.

First things first: the **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) model. This method treats the time series as a linear stochastic process and is a go-to method for a preliminary forecast assessment. Methodologically, it relies on the assumption that the future is a linear combination of past observations and past errors, whether they are explained by the series itself or by its seasonality.

The parameterization of our SARIMA model, **$(0,2,1) \times (0,0,1)_{12}$** (found via the AIC criterion), corresponds to an expansion of the lag operator. The equation we are fitting to the US unemployment data is summarized as:

$$(1-L)^2 y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \Theta_1 \epsilon_{t-12} + \theta_1 \Theta_1 \epsilon_{t-13}$$

Where we observe a **Second-Order Differencing $(1-L)^2$**: This term represents the "acceleration" of the series. Mathematically, it expands to $y_t - 2y_{t-1} + y_{t-2}$, which effectively removes stochastic trends of order up to 2 from the original unemployment rate (possibly due to the two observed shocks).

Another aspect of the SARIMA model specification is the **Short-Term Error ($\theta_1 \epsilon_{t-1}$)**: This captures the impact of the residual (shock) from the previous month.

Finally the model foresees a **Seasonal Error ($\Theta_1 \epsilon_{t-12}$)** and ($\theta_1 \Theta_1 \epsilon_{t-13}$)**: These identify the seasonal residuals from 12 and 13 months ago, allowing the model to correct for annual cycles.

So, in this framework, the current "accelerated" change in unemployment is explained not by past values themselves, but by a combination of recent shocks ($\theta_1$) and yearly seasonal residuals ($\Theta_1$,$\Theta_2$ ).

In contrast to the regressive nature of SARIMA, **Prophet** views forecasting as a curve-fitting exercise. Its methodology is built on a **Generalized Additive Model (GAM)**. Rather than looking for autocorrelation, it decomposes the signal into distinct structural components. Its functional form is:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$


While the components are **summed** together (making it additive), the individual components are nonlinear: the trend ($$g(t)$$) can be fitted as a logistic growth curve or a growth rate adjusted at given changepoints. The seasonality ($$s(t)$$) is a **Fourier Series**, which by definition is not linear, whereas $$h(t)$$ are exogenous, user-defined, components (normally holidays or relevant events).

This strategy allows it to handle irregular spacing and structural breaks—like economic crises—with ease, but the model still relies on a given functional form. That is, the algorithm's effort is to find the best parameters that match that specific functional form.

Our third model specification, on the other hand, starts from a different problem: given the data, which functional form fits best?

The **RandomForestRegressor** shifts the paradigm from temporal sequences to a supervised learning problem. Since decision trees are inherently "time-agnostic," the methodology requires manual feature engineering to create a lag-matrix ($y_{t-1}, y_{t-2}, \dots$). Thus, it excels at capturing nonlinear interactions between lags—something SARIMA cannot do—but it lacks an internal mechanism to handle trends (extrapolation). Its functional form is an average of $B$ individual tree predictions:

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
```

### The Estimates

For simplicity, the first approach was to train our model on half the time period (2005-01-01 to 2014-12-01) and estimate on a hold-out set (the next 120 periods).

I constructed confidence intervals for all of them and compared them on a single plot:


<p align="center">
  <img src="/assets/images/forecasting/forecasting.png" alt="" width="800">
</p>

First, none of the models could anticipate the unemployment spike during COVID‑19, which is unsurprising given there were no prior observations resembling that shock and pandemics are not predictable seasonal events.

Second, both SARIMA and Prophet overfitted the decreasing trend from the 2008 crisis. This is explained by their own nature: both assume a given function that depends on either residuals or time-based components: trend and seasonality. This over-reliance on past data structure has caused both to fail to forecast future unemployment rates (let alone the COVID-19 crisis). In the end, the models forecast negative unemployment rates, which are impossible. Even though Prophet can capture regime changes in its trend function, it clearly failed at this task.

The *Random Forest* forecast (purple dotted line) is the only model that remained "realistic," hovering around the historical mean. Because trees cannot extrapolate beyond the range of the training data, the RF model produced a horizontal, oscillatory forecast. Additionally, its [MAPIE](https://mapie.readthedocs.io/en/stable/) confidence interval is much tighter and more realistic than the massive SARIMA confidence interval (red), which exploded because the model became increasingly "unsure" as it drifted further from the training mean.

The errors clearly illustrate how different the predictions were, on average, from the observed data:


<p align="center">
  <img src="/assets/images/forecasting/forecasting_errors.png" alt="" width="800">
</p>

Using standard (continuous-value) metrics, RF outperforms its peers: other models' RMSEs are approximately **2.07–2.80×** RF's (i.e., **107%–180% higher**), so often **more than twice as large**; their MAEs are roughly **1.60–2.21×** RF's (i.e., **60%–121% higher**).

With current estimates, we can mislead unemployment rate by roughly 1.9 p.p, which is not great, for this time horizon, on average. If we take the pre-test period's mean and create an interval between mean+-1.9, the observed data would be withing this range 90% of the time. Since 1.9 is greater than the train period's standard deviation, this result is of little value.

Regarding the model's comparison, however, can we say that Random Forest is the clear winner for forecasting US unemployment? The short answer is no. These results are heavily shaped by three key decisions: how much historical data we use for training, *when* that training period starts, and how far ahead we're trying to forecast.

In order to address this fact I have estimated 19 different models: the first model is trained in the first year and tested in the following 18, the second model was trained in the first two years, and tested in the following 17, and so on. In the plot below, "Config 1" represents the one-year training and 17-year hold-out set:

<p align="center">
  <img src="/assets/images/forecasting/forecasting_errors_by_train_year.png" alt="" width="800">
</p>


We can see that the best-performing model depends strongly on the train–test split. SARIMA and Prophet perform poorly with small training sets: up to Config 6 (84 training months) they often diverge substantially from the observed series. This likely reflects that these models explicitly fit trend + seasonality, so the 2008 shock impaired parameter estimates, resulting on strong bias.

Although I expected another error spike after the strong COVID shock between Configs 15 and 16, the errors instead show a steady decline — likely because that shock was shorter and was combined to a much smaller extrapolation horizon. This finding is corroborated visually:

<p align="center">
  <img src="/assets/images/forecasting/forecasting_errors_by_train_year_lineplot.png" alt="" width="800">
</p>

While RF+MAPIE performs well during crisis periods, SARIMAX (and Prophet) outperform Random Forest in the last four train–test configurations, reflecting a better learning curve when extrapolation is limited. Counting wins by configuration confirms RF+MAPIE wins most often (RF+MAPIE: 9, SARIMA: 7, Prophet: 2).

Overall, the error patterns reflect each model’s inductive bias under varying train–test horizons: for short training windows and long extrapolations (Configs 1–6), SARIMA and Prophet show high variance and occasional divergence; as the training span increases and forecast horizons shrink (Configs 14–18), SARIMA becomes competitive and often yields the lowest RMSE.



## Conclusion

This post used the monthly US unemployment rate (2014–2024) as a practical forecasting example rather than to declare a single winning method. The series under scrutinity shows clear yearly seasonality and was shaped by two important shocks (the 2008 crisis and COVID‑19), which makes forecasting both challenging and instructive rather than simply a performance exercise.

Key findings and diagnostics:

- Predictability is limited: CoV ≈ 0.36 and MASE diagnostics indicate the task is difficult (Naive overall mean MASE ≈ 8.8; 6‑month mean ≈ 1.5; ARIMA ≈ 1.13). MASE values above 1 highlight that simple baselines are often hard to beat.  
- Error magnitudes depend on the context: compared to RF, other models' RMSEs were roughly **107%–180% higher** and their MAEs **60%–121% higher**. These differences reflect a trade‑off: RF+MAPIE produced conservative, plausible forecasts and tighter empirically calibrated intervals in crisis scenarios, while SARIMA/Prophet sometimes captured trend changes better with ample data but were more sensitive to shocks and extrapolation.

Finally, always put into perspective your problem. Is your series random? Are there structural breaks? How relevant are exogenous events explaining your data generation process? All of these questions should be asked to calibrate you error expectations. Treat you problem as unique and explore its shape before running .fit() and you will likely have a better understaing of your prediction power.