#!/usr/bin/env python3
import re

with open('_posts/2025-12-05-forecasting.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Grammar corrections
replacements = [
    # line 11
    ("In this post I'll explore a time-series forecasting problem. This is not only a classic data-science problem, but also an economics one.",
     "In this post I'll explore a time-series forecasting problem. This is not only a classic data-science problem, but also an economic one."),
    # line 15
    ("In the post I'll address the following question: can we forecast the US unemployment rate?",
     "In this post I'll address the following question: can we forecast the US unemployment rate?"),
    # line 17
    ("More than simplying trying to find the best performance metric, this post will explore some of the nuances behind time-series forecasting.",
     "More than simply trying to find the best performance metric, this post will explore some of the nuances behind time-series forecasting."),
    # line 19
    ('For instance, how "shocks" affect you prediction? How pre-defined forecasting functions perform against model-agnostic models? How to measure predictability?',
     'For instance, how "shocks" affect your prediction? How do pre-defined forecasting functions perform against model-agnostic models? How to measure predictability?'),
    # line 24
    ("First things first, let's dive into the data. I'm using US monthly unemployment rate from 2014-12-16 until 2024-12-01 (that's all I've got in free version of the [Bureau of Labor Statistics](https://api.bls.gov)), making only 240 observations.",
     "First things first, let's dive into the data. I'm using the US monthly unemployment rate from 2014-12-16 to 2024-12-01 (that's all I've got in the free version of the [Bureau of Labor Statistics](https://api.bls.gov)), resulting in only 240 observations."),
    # line 32
    ("Besides the time series itself I have added its confidence interval, a rolling 12-month average (more useful for high-volatility series), and two distinct periods that deserve some attention: the 2008 financial crash and the 2020 COVID-19 pandemic.",
     "Besides the time series itself, I have added its confidence interval, a rolling 12-month average (more useful for high-volatility series), and two distinct periods that deserve some attention: the 2008 financial crash and the 2020 COVID-19 pandemic."),
    # line 36
    ('Despite these shocks, 80% of the observations fall between a 3.4-7.7% range, with a 5.8% mean and a standard deviation of 2.12. Meaning that under "normal periods" the series is relatively stable.',
     'Despite these shocks, 80% of the observations fall between a 3.4-7.7% range, with a 5.8% mean and a standard deviation of 2.12, meaning that under "normal periods" the series is relatively stable.'),
    # line 38
    ("Another interesting observation is how unemployment behaved differently across the two periods. First the 2008 financial crisis took longer to reach its maximum level and more so to converge back to the period's average.",
     "Another interesting observation is how unemployment behaved differently across the two periods. First, the 2008 financial crisis took longer to reach its maximum level and longer to converge back to the period average."),
    # line 46
    ("We can see the GFC takes an impressive 82 months from its assumed beginning (2007-12-01) until it converges to the period mean (2014-09-01). This means a 9-year sluggish recovery. The COVID-19 pandemic, on its turn, takes approximately 15 months from its beginning, in February 2020, until it converges to the period mean again, in May 2021, despite hitting a much higher unemployment level of nearly 15%.",
     "We can see the GFC takes an impressive 82 months from its assumed beginning (2007-12-01) until it converges to the period mean (2014-09-01). This means a 9-year sluggish recovery. The COVID-19 pandemic, in turn, takes approximately 15 months from its beginning in February 2020 to converge to the period mean again in May 2021, despite hitting a much higher unemployment level of nearly 15%."),
    # line 48
    ("In terms of forecastability, hence, the period shows a double challenge, not only two massive shocks, but two very distinct patterns. Having contextualized the unemployment rate during the period of data availability, we should ask ourselves: **is the US unemployment rate between (2014/12 - 2024/12) predictable?**",
     "In terms of forecastability, thus, the period shows a double challenge, not only two massive shocks but also two very distinct patterns. Having contextualized the unemployment rate during the period of data availability, we should ask ourselves: **is the US unemployment rate between (2014/12 - 2024/12) predictable?**"),
    # line 53
    ("One of the most standard ways of measuring how predictable a time-series is through Coefficient of Variation (CoV).",
     "One of the most standard ways to measure the predictability of a time-series is through the Coefficient of Variation (CoV)."),
    # line 53 continued (next sentence)
    ("The idea is simple: compare the series' standard deviation to its mean: $$CoV = \\frac{\\sigma}{\\mu}$$. A value smaller than 0.5 should be considered relatively smooth, $0.5<CoV<1$ can be considered unstable, whereas a $CoV>1$ can be considered quite unstable.",
     "The idea is simple: compare the series' standard deviation to its mean: $$CoV = \\frac{\\sigma}{\\mu}$$. A value smaller than 0.5 indicates relative smoothness; $0.5<CoV<1$ indicates instability; and $CoV>1$ indicates high instability."),
    # line 55
    ("For our series, this ratio is about 0.36. This metric, however, is quite poor since it does not grasp any time-dependence of the series. That is, the data distribution is simply considered as independent, with no attention to time-specific dimensions, such as seasonality, trend, or even order itself. A simple look at the series can give us an idea of how  $CoV$ can be misleading.",
     "For our series, this ratio is about 0.36. This metric, however, is quite poor since it does not capture any time-dependence of the series. That is, the data distribution is simply considered as independent, with no attention to time-dependent dimensions, such as seasonality, trend, or even order itself. A simple look at the series can give us an idea of how $CoV$ can be misleading."),
    # line 65
    ("Simply put, this method highlights *how much better is my model compared to the last observed value?*",
     "Simply put, this method highlights *how much better my model is compared to the last observed value.*"),
    # line 67
    ("Since $y_t$ and $y_{t-1}$ are given, what we need to estimate is $\\hat{y}_t$. One approach is to use a naive basis, such as the mean value of the series, the mean of the n past values, or a simple univariate estimate, like ARIMA. Estimating these values for our problem I could find:",
     "Since $y_t$ and $y_{t-1}$ are given, what we need to estimate is $\\hat{y}_t$. One approach is to use a naive basis, such as the mean value of the series, the mean of the n past values, or a simple univariate estimate, like ARIMA. For our problem, I estimated these values and found:"),
    # line 75
    ("It is interesting to notice that, since MASE values >1 are assumed to be tricky, we confirm that our problem is not an easy one.",
     "It is interesting to note that, since MASE values >1 are considered poor, this confirms that our problem is not an easy one."),
    # line 77
    ("Second, it is interesting to notice how the 6-month mean performed much better than the overall mean. This suggests that our series is time-dependent, meaning that past values influence future values.",
     "Secondly, it is interesting to observe that the 6-month mean performed much better than the overall mean. This suggests that our series is time-dependent, meaning that past values influence future values."),
    # line 79
    ("A common time-series diagnosis for such pattern is to use the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). They measure the effect of $y_{t-k}$ on $y_{t}$. The first disregards the intermediate effects of $t-k-1$ events, whereas the second controls for each individual time effect between past and current events, as shown below:",
     "A common time-series diagnosis for such a pattern is to use the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). They measure the effect of $y_{t-k}$ on $y_t$. The first disregards the intermediate effects of $y_{t-k-1}$ events, whereas the second controls for each individual time effect between past and current events, as shown below:"),
    # line 88
    ("So our series is auto-correlated and it's one hard to estimate according to different MASE metrics.",
     "So our series is auto-correlated and difficult to estimate, as indicated by different MASE metrics."),
    # line 94
    ("Before picking a standard method and running .fit() let's first understand better the problem.",
     "Before picking a standard method and running .fit(), let's first better understand the problem."),
    # line 101 (multiple sentences)
    ("As previously observed the trend itself is quite erratic - but due to two very meaningful shocks.",
     "As previously observed, the trend itself is quite erratic, but due to two very significant shocks."),
    ("The seasonality has been well grasped.",
     "The seasonality has been well captured."),
    ("Intuitively I've calculated the time between the series' ninth decile ($q=.9$) and it returned a yearly seasonality:",
     "Intuitively, I calculated the time between peaks of the seasonal component exceeding the ninth decile ($q=.9$), which indicated a yearly seasonality:"),
    # line 130
    ("Wrapping up what we have seen so far we can say that our series is time-dependent, non-stationary (from the exponential decay of the PACF plot), has a yearly seasonality, and suffered two strong but distinct shocks during the period of analysis. With that in mind we'll explore three different approaches taking that information into account.",
     "Wrapping up what we have seen so far, we can say that our series is time-dependent, non-stationary (from the exponential decay of the PACF plot), has a yearly seasonality, and suffered two strong but distinct shocks during the period of analysis. With that in mind, we'll explore three different approaches."),
    # line 136
    ("First things, first: The **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) model.",
     "First things first: the **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) model."),
    ("This method treats the time series as a linear stochastic process and is a kind of go-to method when you want to have a first predictability idea.",
     "This method treats the time series as a linear stochastic process and is a go-to method for a preliminary assessment of predictability."),
    # line 140
    ("The specific identification of our SARIMA model, **$(0,2,1) \\times (0,0,1)_{12}$** (via the AIC criterion), corresponds to a specific expansion of the lag operator. The equation we are fitting to the US unemployment data is summarized as:",
     "The parameterization of our SARIMA model, **$(0,2,1) \\times (0,0,1)_{12}$** (via the AIC criterion), corresponds to an expansion of the lag operator. The equation we are fitting to the US unemployment data is summarized as:"),
    # line 147
    ("* **Short-Term Error ($\\theta_1 \\epsilon_{t-1}$)**: This captures the impact of the residual (shock) from the immediate previous month.",
     "* **Short-Term Error ($\\theta_1 \\epsilon_{t-1}$)**: This captures the impact of the residual (shock) from the previous month."),
    # line 149
    ("* **Seasonal Error ($\\Theta_1 \\epsilon_{t-12}$)** and ($\\theta_1 \\Theta_1 \\epsilon_{t-13}$)**: This identifies the seasonal residuals from 12 and 13 months ago, allowing the model to correct for annual cycles.",
     "* **Seasonal Error ($\\Theta_1 \\epsilon_{t-12}$)** and ($\\theta_1 \\Theta_1 \\epsilon_{t-13}$)**: These identify the seasonal residuals from 12 and 13 months ago, allowing the model to correct for annual cycles."),
    # line 158
    ("While the components are **summed** together (making it additive), the individual components are nonlinear: the trend can be fitted as a logistic growth curve or a growth-rate adjusted at given states. Whereas the seasonality is a **Fourier Series** which by definition is not linear.",
     "While the components are **summed** together (making it additive), the individual components are nonlinear: the trend can be fitted as a logistic growth curve or a growth-rate adjusted at given states. The seasonality is a **Fourier Series**, which by definition is not linear."),
    # line 164
    ("The **RandomForestRegressor** shifts the paradigm from temporal sequences to a supervised learning problem. Since decision trees are inherently \"time-blind,\" the methodology requires manual feature engineering to create a lag-matrix ($y_{t-1}, y_{t-2}, \\dots$). So basically it excels at capturing nonlinear interactions between lags—something SARIMA cannot do—but it lacks an internal mechanism to handle trends (extrapolation). Its functional form is an average of $B$ individual tree predictions:",
     "The **RandomForestRegressor** shifts the paradigm from temporal sequences to a supervised learning problem. Since decision trees are inherently \"time-blind,\" the methodology requires manual feature engineering to create a lag-matrix ($y_{t-1}, y_{t-2}, \\dots$). Thus, it excels at capturing nonlinear interactions between lags—something SARIMA cannot do—but it lacks an internal mechanism to handle trends (extrapolation). Its functional form is an average of $B$ individual tree predictions:"),
    # line 182
    ("For simplicity, the first approach was to train our model on half the time (2005/01/01 - 2014-12-01) and estimate on a hold-out set (the next 120 periods).",
     "For simplicity, the first approach was to train our model on half the time period (2005-01-01 to 2014-12-01) and estimate on a hold-out set (the next 120 periods)."),
    # line 184
    ("I have constructed confidence intervals for all of them, and compared them on a single plot:",
     "I constructed confidence intervals for all of them and compared them on a single plot:"),
    # line 193
    ("Two things deserve attention. First, none could foresee the unemployment rate during COVID-19, which should be obvious, since there is no previous pattern to be replicated. There are no past values even close to that change, there is no seasonality in pandemics (I would guess).",
     "Two things deserve attention. First, none of the models could foresee the unemployment rate during COVID-19, which should be obvious, since there is no previous pattern to replicate. There are no past values even close to that change, and there is no seasonality in pandemics (I would guess)."),
    # line 195
    ("Second, both SARIMA and Prophet overfitted the decreasing trend from the 2008 crisis. This is explained by their own nature: both assume a given function that depends on either residuals or time-based modules: trend and seasonality. This over-reliance on past data structure has failed them both to forecast future unemployment rates (let alone the COVID-19 crisis). In the end, the model forecasts negative unemployment rates which are impossible. Even though Prophet is able to capture regime changes in its trend function, it has clearly failed that task.",
     "Second, both SARIMA and Prophet overfitted the decreasing trend from the 2008 crisis. This is explained by their own nature: both assume a given function that depends on either residuals or time-based components: trend and seasonality. This over-reliance on past data structure has caused both to fail to forecast future unemployment rates (let alone the COVID-19 crisis). In the end, the models forecast negative unemployment rates, which are impossible. Even though Prophet can capture regime changes in its trend function, it clearly failed at this task."),
    # line 200
    ("The errors clearly illustrate how different, on average, the predictions were from observed data:",
     "The errors clearly illustrate how different the predictions were, on average, from the observed data:"),
    # line 206
    ("We can see that using standard (continuous-value) error metrics such as MAE and RMSE, the RF model outperforms its peers, with error differences ranging between 107-180% for RMSE and 60-121% for MAE.",
     "We can see that using standard (continuous-value) error metrics such as MAE and RMSE, the RF model outperforms its peers, with error differences ranging from 107% to 180% for RMSE and from 60% to 121% for MAE."),
    # line 208
    ("But that is not the full story. These results are greatly impacted by three major choices: the length of the training data, the data-point of this training data - that is the date itself - and lastly, the lenght of the test-set, the one we are comparing our estimates against with.",
     "But that is not the full story. These results are greatly impacted by three major choices: the length of the training data, the data point of this training data - that is the date itself - and lastly, the length of the test-set, the one we are comparing our estimates against."),
    # line 210
    ("In order to adress this fact I have estimated 19 different models: the first model is trainned in the first year and teste in the following 18, the second model was trained in the first two years, and tested in the following 17, and so on. In the plot bellow, the first value represents the one-year-17-year train and hold-out set:",
     "In order to address this fact I have estimated 19 different models: the first model is trained in the first year and tested in the following 18, the second model was trained in the first two years, and tested in the following 17, and so on. In the plot below, the first value represents the one-year training and 17-year hold-out set:"),
    # line 216
    ("We can see that \"best results\" change reasonably depending on the train-test combination. The second noticeable fact is that SARIMA and PROPHET have performed quite poorly for small training data, as up to Config 6 (84 training months), SARIMA and specially PROPHET perform quite poorly. One of the explanations is that since these models fit, in a macro-sense a trend+seasonal effect, the shock effect of the 2008 crisis have undermined their performance, in combination with a weak learning process of the trend and seasonal components.",
     "We can see that \"best results\" change substantially depending on the train-test combination. The second noticeable fact is that SARIMA and PROPHET perform quite poorly with small training data, as up to Config 6 (84 training months), SARIMA and especially PROPHET perform quite poorly. One of the explanations is that since these models fit, in a macro-sense a trend+seasonal effect, the shock effect of the 2008 crisis undermined their performance, combined with a weak learning process of the trend and seasonal components."),
    # line 218
    ("On the other hand, whereas the RF+MAPIE algo did well on crisis periods, SARIMAX (and Prophet) have outperformed the Random Forest model in the last 4 train-test config. This can be associated to a better learning curve compared to the \"miopic\" stand point from a tree-based model. With little extrapolation, SARIMA could find better parameters, interpret better seasonal effects and trend, and with smaller extrapolation, provided the best fit.",
     "On the other hand, whereas the RF+MAPIE algorithm did well on crisis periods, SARIMAX (and Prophet) have outperformed the Random Forest model in the last 4 train-test configurations. This can be associated with a better learning curve compared to the \"myopic\" standpoint of a tree-based model. With little extrapolation, SARIMA could find better parameters, better interpret seasonal effects and trends, and with less extrapolation, provided the best fit."),
    # line 222
    ("If we pick the best model for each period and count the frequency they win we can see that RF+MAPI still outperforms the two models, altough SARIMA does not lag behind much.",
     "If we pick the best model for each period and count the frequency with which they win we can see that RF+MAPIE still outperforms the two models, although SARIMA does not lag behind much."),
    # line 247
    ("Adding MAPIE (conformal prediction) to Random Forest provided an additional edge: empirically calibrated uncertainty quantification. Rather than assuming error distributions, MAPIE learned from actual historical prediction errors, providing realistic prediction intervals that widened appropriately over the forecast horizon.",
     "Adding MAPIE (conformal prediction) to Random Forest provided an additional edge: empirically calibrated uncertainty quantification. Rather than assuming error distributions, MAPIE learns from actual historical prediction errors, providing realistic prediction intervals that widen appropriately over the forecast horizon."),
]

# Apply replacements sequentially (order matters)
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
    else:
        # Try with newline variations
        old_alt = old.replace('\n', ' ')
        if old_alt in content:
            content = content.replace(old_alt, new)
        else:
            print(f"Warning: pattern not found: {old[:50]}...")

# Additional fluency improvements (more subjective)
# We'll do a few more replacements for better flow
fluency_replacements = [
    # Replace "Since it is easier to access public API-based macroeconomic datasets, I'll stick with the second."
    # with "Since it is easier to access public API-based macroeconomic datasets, I'll focus on the economic problem."
    ("Since it is easier to access public API-based macroeconomic datasets, I'll stick with the second.",
     "Since it is easier to access public API-based macroeconomic datasets, I'll focus on the economic problem."),
    # Replace "halved occupations extremely fast" with "halved employment extremely fast"
    ("halved occupations extremely fast",
     "halved employment extremely fast"),
    # Replace "time-specific dimensions" with "time-dependent dimensions"
    ("time-specific dimensions",
     "time-dependent dimensions"),
    # Replace "time-based modules" with "time-based components"
    ("time-based modules",
     "time-based components"),
    # Replace "growth-rate adjusted at given states" with "growth rate adjusted at given changepoints"
    ("growth-rate adjusted at given states",
     "growth rate adjusted at given changepoints"),
    # Replace "time-blind" with "time-agnostic"
    ("time-blind",
     "time-agnostic"),
    # Replace "So basically" with "Thus"
    ("So basically",
     "Thus"),
]

for old, new in fluency_replacements:
    if old in content:
        content = content.replace(old, new)

# Write to new file
output_path = '_posts/2025-12-05-forecasting_corrected.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created corrected file at {output_path}")