---
title: "US Unemployemnt Forecasting: a use-case"
date: 2025-12-05
---

In this post I'll explore a time-series forecasting problem. This is a classic data-science (and economics) problem. Whereas in the first, companies are interested in forecasting sales, conversions, net revenue, or any other relevant KPI, in economics it is often related to macroeconomic series.

Since it is easier to access public API-based macroeconomic data-sets I'll stick with the second.

In the post I'll adress the following question: can I forecast the US unemployment rate? 

First things first, lets dive into the data. I'm using US monthly unemployment rate from 2014-12-16 until 2024-12-01 (that's all I've got in free version of the [Bureau Of Labour and Statistics ](https://api.bls.gov)), making only 240 observations.