---
title: "US Unemployemnt Forecasting: a use-case"
date: 2025-12-05
classes: wide  
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

Since theses are not endogenous economic events, there is a good chance that our model will fail badly in those periods. So the first question we should be askin is: **is the US unemployment rate between (2014/12 - 2024/12) predictable?**

Despite these shocks, 80% of the observations fall between a 3.4-7.7% range, with a 5.8% mean and a 2.12 standard deviation. Meaning that under "normal periods" the serie is relatively stable.

Another interesting observation is how unemployment behaved differently across these two periods. First the 2008 financial crisis took longer to reach its maximum level and also too longer to converge back to the period average. The Covid shock, on the other hand, halved ocuppations extremely fast, but was also faster to converge. We can better illustrate this with the following plot:


<p align="center">
  <img src="/analysis/crisis_comparison.png" alt="" width="800">
</p>

