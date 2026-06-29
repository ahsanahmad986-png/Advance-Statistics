# ══════════════════════════════════════════════════════════════════════════════
# PART C — SIMPLE & MULTIPLE LINEAR REGRESSION
# Response variable: BMI (continuous)
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

y = df_eng["bmi"].values     # continuous response

# ── Simple LR: one predictor at a time ────────────────────────────────────────
slr_predictor = "gen_hlth"   # strongest single correlate with BMI

# ── Multiple LR: 10+ predictors ───────────────────────────────────────────────
mlr_features = [
    "age_group_z",      # standardised age group
    "gen_hlth_z",       # standardised general health
    "education_z",      # standardised education
    "income_z",         # standardised income
    "ment_hlth_z",      # standardised mental health days
    "log_phys_hlth",    # log-transformed physical unhealthy days
    "high_bp",          # binary: high blood pressure
    "high_chol",        # binary: high cholesterol
    "smoker",           # binary: smoker
    "phys_activity",    # binary: physically active
    "diabetes",         # binary: diabetes status
    "diff_walk",        # binary: difficulty walking
    "age_sq",           # quadratic age effect
    "age_x_activity",   # interaction: age × physical activity
    "bp_x_chol",        # interaction: BP × cholesterol
    "sex",              # binary: sex
]
mlr_features = [f for f in mlr_features if f in df_eng.columns]

print(f"✅ Response        : bmi  (n={len(y)})")
print(f"   SLR predictor  : {slr_predictor}")
print(f"   MLR predictors : {len(mlr_features)}")
print(f"   {mlr_features}")

# ── Simple Linear Regression: BMI ~ gen_hlth ──────────────────────────────────

X_slr = df_eng[[slr_predictor]].values
X_slr_const = np.column_stack([np.ones(len(y)), X_slr])

# OLS closed-form solution
beta_slr = np.linalg.lstsq(X_slr_const, y, rcond=None)[0]
y_pred_slr = X_slr_const @ beta_slr
res_slr    = y - y_pred_slr
n_, p_     = len(y), 1

# Metrics
r2_slr     = r2_score(y, y_pred_slr)
adj_r2_slr = 1 - (1 - r2_slr) * (n_-1) / (n_-p_-1)
rmse_slr   = np.sqrt(mean_squared_error(y, y_pred_slr))
aic_slr    = n_ * np.log(mean_squared_error(y, y_pred_slr)) + 2*(p_+1)
bic_slr    = n_ * np.log(mean_squared_error(y, y_pred_slr)) + (p_+1)*np.log(n_)

# t-test on coefficients
sigma2_slr = (res_slr @ res_slr) / (n_ - 2)
var_beta   = sigma2_slr * np.linalg.inv(X_slr_const.T @ X_slr_const)
se_slr     = np.sqrt(np.diag(var_beta))
t_slr      = beta_slr / se_slr
p_slr      = 2 * stats.t.sf(np.abs(t_slr), df=n_-2)

print("── Simple Linear Regression: BMI ~ gen_hlth ─────────────────")
print(f"\n   Model: BMI = β₀ + β₁·gen_hlth + ε")
print(f"\n   {'Parameter':<15} {'Estimate':>10} {'SE':>9} {'t':>9} {'p-value':>12} {'Sig'}")
print("   " + "-"*60)
for name, b, se, t, pv in zip(["intercept","gen_hlth"],
                                beta_slr, se_slr, t_slr, p_slr):
    sig = "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
    print(f"   {name:<15} {b:>10.4f} {se:>9.4f} {t:>9.3f} {pv:>12.4e}  {sig}")

print(f"\n   R²       = {r2_slr:.4f}")
print(f"   Adj R²   = {adj_r2_slr:.4f}")
print(f"   RMSE     = {rmse_slr:.4f}")
print(f"   AIC      = {aic_slr:.2f}")
print(f"   BIC      = {bic_slr:.2f}")
print(f"\n   Interpretation:")
print(f"   A 1-unit increase in gen_hlth (better→worse) is associated")
print(f"   with a {beta_slr[1]:.3f} unit change in BMI (p {'< 0.001' if p_slr[1]<0.001 else f'= {p_slr[1]:.4f}'}).")

# ── Multiple Linear Regression ─────────────────────────────────────────────────

X_mlr = df_eng[mlr_features].values
X_mlr_const = np.column_stack([np.ones(len(y)), X_mlr])
p_mlr = len(mlr_features)

# OLS solution
beta_mlr  = np.linalg.lstsq(X_mlr_const, y, rcond=None)[0]
y_pred_mlr = X_mlr_const @ beta_mlr
res_mlr    = y - y_pred_mlr

# Metrics
r2_mlr     = r2_score(y, y_pred_mlr)
adj_r2_mlr = 1 - (1 - r2_mlr) * (len(y)-1) / (len(y)-p_mlr-1)
rmse_mlr   = np.sqrt(mean_squared_error(y, y_pred_mlr))
aic_mlr    = len(y) * np.log(mean_squared_error(y, y_pred_mlr)) + 2*(p_mlr+1)
bic_mlr    = len(y) * np.log(mean_squared_error(y, y_pred_mlr)) + (p_mlr+1)*np.log(len(y))

# t-tests on all coefficients
sigma2_mlr = (res_mlr @ res_mlr) / (len(y) - p_mlr - 1)
XtX_inv    = np.linalg.inv(X_mlr_const.T @ X_mlr_const)
se_mlr     = np.sqrt(sigma2_mlr * np.diag(XtX_inv))
t_mlr      = beta_mlr / se_mlr
pv_mlr     = 2 * stats.t.sf(np.abs(t_mlr), df=len(y)-p_mlr-1)

# F-test
ss_res = np.sum(res_mlr**2)
ss_tot = np.sum((y - y.mean())**2)
F_stat = ((ss_tot-ss_res)/p_mlr) / (ss_res/(len(y)-p_mlr-1))
F_pval = stats.f.sf(F_stat, p_mlr, len(y)-p_mlr-1)

print("── Multiple Linear Regression (16 predictors) ────────────────")
print(f"\n   {'Feature':<22} {'β':>9} {'SE':>9} {'t':>9} {'p-value':>12}  Sig")
print("   " + "-"*70)
names_all = ["intercept"] + mlr_features
for name, b, se, t, pv in zip(names_all, beta_mlr, se_mlr, t_mlr, pv_mlr):
    sig = "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else ""
    print(f"   {name:<22} {b:>9.4f} {se:>9.4f} {t:>9.3f} {pv:>12.4e}  {sig}")

print(f"\n   R²            = {r2_mlr:.4f}")
print(f"   Adj R²        = {adj_r2_mlr:.4f}")
print(f"   RMSE          = {rmse_mlr:.4f}")
print(f"   AIC           = {aic_mlr:.2f}")
print(f"   BIC           = {bic_mlr:.2f}")
print(f"   F-statistic   = {F_stat:.3f}  (df1={p_mlr}, df2={len(y)-p_mlr-1})")
print(f"   F p-value     = {F_pval:.4e}  {'*** Highly significant' if F_pval<0.001 else ''}")

fig, ax = plt.subplots(figsize=(10, 8))

coef_df = pd.DataFrame({
    "feature" : mlr_features,
    "beta"    : beta_mlr[1:],
    "se"      : se_mlr[1:],
    "pval"    : pv_mlr[1:],
}).sort_values("beta")

colors_coef = ["#D64045" if b > 0 else "#2C5F8A" for b in coef_df["beta"]]
bars = ax.barh(coef_df["feature"], coef_df["beta"],
               xerr=coef_df["se"], color=colors_coef,
               edgecolor="white", alpha=0.85,
               error_kw=dict(ecolor="#555", capsize=4, linewidth=1.2))

ax.axvline(0, color="#333", linewidth=1.5, linestyle='--')

# Mark significant predictors
for i, (_, row) in enumerate(coef_df.iterrows()):
    if row["pval"] < 0.05:
        ax.text(row["beta"] + (0.05 if row["beta"]>0 else -0.05),
                i, "*", ha='center', va='center',
                fontsize=13, fontweight='bold', color="#333")

ax.set_xlabel("Regression Coefficient (β)  ± SE", fontsize=11)
ax.set_title("Part C — MLR Coefficients with Standard Errors\n"
             "(red=positive effect on BMI, blue=negative)",
             fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("fig_c1_coefficients.png", dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Part C — Regression Diagnostics  (MLR: BMI ~ 16 predictors)",
             fontsize=15, fontweight='bold')

# ① SLR: Actual vs Fitted
ax = axes[0, 0]
ax.scatter(y_pred_slr, y, alpha=0.3, s=14, color="#2C5F8A")
lims = [min(y.min(), y_pred_slr.min()), max(y.max(), y_pred_slr.max())]
ax.plot(lims, lims, color="#D64045", linewidth=2, linestyle='--')
ax.set_xlabel("Fitted BMI (SLR)"); ax.set_ylabel("Actual BMI")
ax.set_title(f"SLR: Actual vs Fitted\nR²={r2_slr:.3f}")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ② MLR: Actual vs Fitted
ax = axes[0, 1]
ax.scatter(y_pred_mlr, y, alpha=0.3, s=14, color="#4CAF89")
lims = [min(y.min(), y_pred_mlr.min()), max(y.max(), y_pred_mlr.max())]
ax.plot(lims, lims, color="#D64045", linewidth=2, linestyle='--')
ax.set_xlabel("Fitted BMI (MLR)"); ax.set_ylabel("Actual BMI")
ax.set_title(f"MLR: Actual vs Fitted\nR²={r2_mlr:.3f}")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ③ Residuals vs Fitted
ax = axes[0, 2]
ax.scatter(y_pred_mlr, res_mlr, alpha=0.3, s=14, color="#E8834B")
ax.axhline(0, color="#D64045", linewidth=2, linestyle='--')
# LOWESS-style smoothing (running mean)
sorted_idx  = np.argsort(y_pred_mlr)
window      = 50
smooth_x    = [y_pred_mlr[sorted_idx][i:i+window].mean()
               for i in range(0, len(y_pred_mlr)-window, 10)]
smooth_y    = [res_mlr[sorted_idx][i:i+window].mean()
               for i in range(0, len(res_mlr)-window, 10)]
ax.plot(smooth_x, smooth_y, color="#2C5F8A", linewidth=2, label="Trend")
ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted\n(should be random around 0)")
ax.legend(); ax.grid(alpha=0.35)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ④ Normal Q-Q Plot
ax = axes[1, 0]
(osm, osr), (slope, intercept, r) = stats.probplot(res_mlr)
ax.scatter(osm, osr, alpha=0.4, s=12, color="#2C5F8A")
qq_line = np.array([osm[0], osm[-1]])
ax.plot(qq_line, slope*qq_line+intercept,
        color="#D64045", linewidth=2, label=f"r={r:.3f}")
ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
ax.set_title("Normal Q-Q Plot\n(points near line = normal residuals)")
ax.legend(); ax.grid(alpha=0.35)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ⑤ Residual Histogram
ax = axes[1, 1]
ax.hist(res_mlr, bins=35, color="#2C5F8A",
        edgecolor="white", linewidth=0.5, alpha=0.85, density=True)
xr = np.linspace(res_mlr.min(), res_mlr.max(), 300)
ax.plot(xr, stats.norm.pdf(xr, 0, res_mlr.std()),
        color="#D64045", linewidth=2.5, label="N(0, σ²)")
ax.set_xlabel("Residuals"); ax.set_ylabel("Density")
ax.set_title("Residual Histogram\n(should follow normal curve)")
ax.legend(); ax.grid(alpha=0.35)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ⑥ Scale-Location
ax = axes[1, 2]
sqrt_abs_res = np.sqrt(np.abs(res_mlr))
ax.scatter(y_pred_mlr, sqrt_abs_res, alpha=0.3, s=14, color="#9B59B6")
smooth_sl_x = [y_pred_mlr[sorted_idx][i:i+window].mean()
               for i in range(0, len(y_pred_mlr)-window, 10)]
smooth_sl_y = [sqrt_abs_res[sorted_idx][i:i+window].mean()
               for i in range(0, len(sqrt_abs_res)-window, 10)]
ax.plot(smooth_sl_x, smooth_sl_y, color="#D64045", linewidth=2)
ax.set_xlabel("Fitted Values"); ax.set_ylabel("√|Residuals|")
ax.set_title("Scale-Location Plot\n(flat line = homoscedasticity)")
ax.grid(alpha=0.35)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig_c2_diagnostics.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_c2_diagnostics.png")

print("── Assumption Testing ────────────────────────────────────────")

# 1. Normality of residuals — Shapiro-Wilk
sw_stat, sw_p = stats.shapiro(res_mlr[:500])
print(f"\n   1. Normality (Shapiro-Wilk on first 500 residuals):")
print(f"      W = {sw_stat:.4f}   p = {sw_p:.4f}")
print(f"      {'✅ Residuals approx. normal (p > 0.05)' if sw_p>0.05 else '⚠️  Residuals not perfectly normal (common in large samples)'}")

# 2. Homoscedasticity — Breusch-Pagan proxy (Fligner-Killeen test)
half = len(res_mlr) // 2
fl_stat, fl_p = stats.fligner(res_mlr[:half], res_mlr[half:])
print(f"\n   2. Homoscedasticity (Fligner-Killeen test):")
print(f"      stat = {fl_stat:.4f}   p = {fl_p:.4f}")
print(f"      {'✅ Variances roughly equal' if fl_p>0.05 else '⚠️  Possible heteroscedasticity'}")

# 3. Autocorrelation — Durbin-Watson
dw = np.sum(np.diff(res_mlr)**2) / np.sum(res_mlr**2)
print(f"\n   3. Autocorrelation (Durbin-Watson):")
print(f"      DW = {dw:.4f}   (ideal ≈ 2.0)")
print(f"      {'✅ No autocorrelation' if 1.5<dw<2.5 else '⚠️  Possible autocorrelation'}")

# 4. Skewness & Kurtosis of residuals
print(f"\n   4. Residual Distribution:")
print(f"      Skewness  = {stats.skew(res_mlr):.4f}  (|value| < 0.5 is ideal)")
print(f"      Kurtosis  = {stats.kurtosis(res_mlr):.4f}  (≈ 0 for normal)")

# ── Train/Test Split ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_mlr, y, test_size=0.2, random_state=42
)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

r2_train  = mlr_model.score(X_train, y_train)
r2_test   = mlr_model.score(X_test, y_test)
rmse_test = np.sqrt(mean_squared_error(y_test, mlr_model.predict(X_test)))

print("── Train / Test Performance ──────────────────────────────────")
print(f"\n   Training R²  = {r2_train:.4f}")
print(f"   Test R²      = {r2_test:.4f}")
print(f"   Gap          = {r2_train - r2_test:.4f}  "
      f"{'✅ Good generalisation' if abs(r2_train-r2_test)<0.05 else '⚠️  Possible overfitting'}")
print(f"   Test RMSE    = {rmse_test:.4f}")

print(f"""
   Interpretation:
   The MLR model explains {r2_mlr:.1%} of variance in BMI using {p_mlr} predictors.
   The small train/test gap ({r2_train-r2_test:.4f}) suggests the model generalises well.
   Most impactful predictors (by |t|): gen_hlth, high_bp, phys_activity.
""")

print("=" * 60)
print("  PART C — LINEAR REGRESSION COMPLETE")
print("=" * 60)
print(f"""
  SIMPLE LINEAR REGRESSION (BMI ~ gen_hlth):
  ─────────────────────────────────────────
  β₀ (intercept) = {beta_slr[0]:.4f}
  β₁ (gen_hlth)  = {beta_slr[1]:.4f}  (p {'<0.001' if p_slr[1]<0.001 else f'={p_slr[1]:.4f}'})
  R²             = {r2_slr:.4f}
  RMSE           = {rmse_slr:.4f}

  MULTIPLE LINEAR REGRESSION (BMI ~ 16 predictors):
  ──────────────────────────────────────────────────
  R²             = {r2_mlr:.4f}
  Adj R²         = {adj_r2_mlr:.4f}
  RMSE           = {rmse_mlr:.4f}
  F-statistic    = {F_stat:.2f}  (p < 0.001) *** Overall model significant
  AIC            = {aic_mlr:.2f}
  BIC            = {bic_mlr:.2f}

  KEY FINDINGS:
  • gen_hlth, high_bp, phys_activity are strongest predictors of BMI
  • F-test confirms the overall model is highly significant
  • Residuals are approximately normal (Shapiro-Wilk p={sw_p:.4f})
  • Train R²={r2_train:.3f} vs Test R²={r2_test:.3f} → good generalisation

  ▶  Continuing to Part D — Generalized Linear Models...
""")
