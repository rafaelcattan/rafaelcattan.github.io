---
title: "US Unemployemnt Forecasting: a use-case"
date: 2025-12-05
classes: wide  
layout: single
usemathjax: true

---

In this post I'll explore a time-series forecasting problem. This is not only a classic data-science problem, but also an economics one. Whereas in the first, companies are interested in forecasting sales, conversions, net revenue, or any other relevant KPI, in economics it is often related to macroeconomic series.

Since it is easier to access public API-based macroeconomic data-sets I'll stick with the second.

In the post I'll adress the following question: can I forecast the US unemployment rate? 

First things first, lets dive into the data. I'm using US monthly unemployment rate from 2014-12-16 until 2024-12-01 (that's all I've got in free version of the [Bureau Of Labour and Statistics ](https://api.bls.gov)), making only 240 observations.

## Some Context

Before any methodological overview, lets have quick look at the time series under analysis:

<p align="center">
  <img src="/analysis/us_unemployment_with_ci.png" alt="Forecast Results" width="800">
</p>

Besides the time series itself I have added its confidence interval, a rolling 12-month average (more useful for high-volatility seires), and two distinct periods that diserve some attention: the 2008 financial crash and the 2020 Covid pandemics.

Since theses are not endogenous economic events, there is a good chance that our model will fail badly in those periods.

Despite these shocks, 80% of the observations fall between a 3.4-7.7% range, with a 5.8% mean and a 2.12 standard deviation. Meaning that under "normal periods" the serie is relatively stable.

Another interesting observation is how unemployment behaved differently across these two periods. First the 2008 financial crisis took longer to reach its maximum level and also too longer to converge back to the period average. The Covid shock, on the other hand, halved ocuppations extremely fast, but was also faster to converge. We can better illustrate this with the following plot:


<p align="center">
  <img src="/analysis/crisis_comparison.png" alt="" width="800">
</p>


We can see the GFC takes impressive 83 months from its assumed beggining (2007-12-01) until it converges to the period mean (2014-10-01). This means a 9-year sluggish recover. The Covid pandemics, on its turn, takes approximately 23 months from its beggining, in February 2020, until it converges to the period mean again, in December 2021, despite hitting a much higher unemployment level of nearly 15%.

In terms of forecastability, hence, the period shows a double challenge, not only two massive shocks, they showed very distinct patterns. Having contextualized the period' highlights the question we should be askin is: **is the US unemployment rate between (2014/12 - 2024/12) predictable?**


## Measuring Predictability

One of the most standard ways of measuring how predicatable a time-series is through Coefficient of Variation (CoV). The idea is simple: compare the series' standard deviation to its mean: $$CoV = \frac{\sigma}{\mu}$$. A value smaller than 0.5 should be considerd relatively smooth, $0.5<CoV<1$ can be considered unstable, whereas a $CoV>1$ can be considered quite unstable.

For our series, this ratio if of about 0.36. This metric, however, is quite poor since it does not grasp any time-dependece of the series: that is the data distribution is simply considered as independent, with no attention to time-specific dimensions, such as seasonality, trend, or even order itself. A look at the series shows and $CoV$ can be misleading.

A hands-on approach that considers time-specific structure is the Mean Absolute Scaled Error, formaly defined as : 

$$
\text{MASE} =
\frac{\frac{1}{n}\sum_{t=1}^{n} \lvert y_t - \hat{y}_t \rvert}
{\frac{1}{n-1}\sum_{t=2}^{n} \lvert y_t - y_{t-1} \rvert}
$$

Simply put, this method highlights \it{how better (in the best scenario) is my model against the last observed value?}

Since $y_t$$ and $y_{t-1}$ are given, what we need to estimate is $$\hat{y}_t$$. One approach is to use a naive basis, such as the mean value of the series, the mean of the n past values, or as simple univariate estimate, like ARIMA. Estimating theses values for our problem I cound find: 8.8, 1.5 and 1.13, respectively. As it is standard, MASE values >1 are assumed to be tricky. 

But hang in there, we still have a lot of room to find out how far can we can in terms of forecasting the US unemployment rate.



