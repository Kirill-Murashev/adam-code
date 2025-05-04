import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.metrics import mean_squared_error, mean_absolute_error


def regression_diagnostics(df, formula, out_prefix):
    """
    Fit OLS model, compute diagnostics, and save outputs in 'primary_models' subfolder.
    """
    # Ensure output directory exists
    out_dir = 'primary_models'
    os.makedirs(out_dir, exist_ok=True)

    # 1. Fit model
    model = smf.ols(formula, data=df).fit()
    y_true = model.model.endog
    y_pred = model.fittedvalues
    resid = model.resid

    # 2. Basic error metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # 3. Heteroscedasticity tests
    bp_p = het_breuschpagan(resid, model.model.exog)[3]
    white_p = het_white(resid, model.model.exog)[3]

    # 4. Autocorrelation test (Ljung-Box)
    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True).iloc[0]['lb_pvalue']

    # 5. Specification tests
    reset_p = linear_reset(model, use_f=True).pvalue
    try:
        jtest_p = sm.stats.diagnostic.compare_j(model).pvalue
    except Exception:
        jtest_p = np.nan

    # 6. Multicollinearity: VIF
    exog = model.model.exog
    vif_vals = [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]
    vif_df = pd.Series(vif_vals, index=model.model.exog_names, name='VIF')
    vif_df.to_csv(f"{out_dir}/{out_prefix}_vif.csv")

    # 7. Influence measures
    infl = OLSInfluence(model)
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    dfbetas = infl.dfbetas
    dffits = infl.dffits[0]

    influence_df = pd.DataFrame({
        'cooks_d': cooks,
        'leverage': leverage,
        'dffits': dffits
    }, index=df.index)
    for i, name in enumerate(model.params.index):
        influence_df[f'dfbeta_{name}'] = dfbetas[:, i]
    influence_df.to_csv(f"{out_dir}/{out_prefix}_influence.csv")

    # 8. Collect global metrics
    metrics = {
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'BP_pvalue': bp_p,
        'White_pvalue': white_p,
        'LB_pvalue': lb_p,
        'RESET_pvalue': reset_p,
        'JTest_pvalue': jtest_p,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{out_dir}/{out_prefix}_metrics.csv", index=False)

    # 9. Save model summary
    with open(f"{out_dir}/{out_prefix}_summary.txt", 'w') as f:
        f.write(model.summary().as_text())

    return metrics_df, influence_df
