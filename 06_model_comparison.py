# ══════════════════════════════════════════════════════════════════════════════
# PART F — MODEL COMPARISON & CRITICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── Train/Test R² for bias-variance analysis ───────────────────────────────────
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_mlr, y, test_size=0.2, random_state=42)

mlr_cv    = LinearRegression().fit(X_tr_m, y_tr_m)
r2_tr_mlr = mlr_cv.score(X_tr_m, y_tr_m)
r2_te_mlr = mlr_cv.score(X_te_m, y_te_m)

# Overfit test: degree 8 polynomial
from sklearn.preprocessing import PolynomialFeatures as PF
poly8     = PF(degree=8, include_bias=True)
idx_tr    = np.arange(800)
idx_te    = np.arange(800, 1000)
X8_tr     = poly8.fit_transform(x_age[idx_tr].reshape(-1,1))
X8_te     = poly8.transform(x_age[idx_te].reshape(-1,1))
reg8      = LinearRegression(fit_intercept=False).fit(X8_tr, y_bmi[idx_tr])
r2_tr_p8  = r2_score(y_bmi[idx_tr], reg8.predict(X8_tr))
r2_te_p8  = r2_score(y_bmi[idx_te], reg8.predict(X8_te))

# ── Compile all metrics ────────────────────────────────────────────────────────
all_models = {
    "SLR (BMI~gen_hlth)"     : {"type":"Regression", "R2":r2_slr,    "RMSE":rmse_slr,
                                  "AIC":aic_slr,   "BIC":bic_slr,   "params":2},
    "MLR (16 predictors)"    : {"type":"Regression", "R2":r2_mlr,    "RMSE":rmse_mlr,
                                  "AIC":aic_mlr,   "BIC":bic_mlr,   "params":p_mlr+1},
    "Logistic Regression"    : {"type":"GLM-Binary", "R2":mcfadden,  "RMSE":None,
                                  "AIC":aic_log,   "BIC":bic_log,   "params":p_log},
    "Poisson GLM"            : {"type":"GLM-Count",  "R2":None,      "RMSE":rmse_pois,
                                  "AIC":aic_pois,  "BIC":bic_pois,  "params":p_pois},
    "Polynomial deg=1"       : {"type":"Nonlinear",  "R2":results_poly[1]["r2"],
                                  "RMSE":results_poly[1]["rmse"],
                                  "AIC":results_poly[1]["aic"],
                                  "BIC":results_poly[1]["bic"], "params":2},
    "Polynomial deg=3"       : {"type":"Nonlinear",  "R2":r2_poly3,  "RMSE":rmse_poly3,
                                  "AIC":aic_poly3, "BIC":bic_poly3, "params":4},
    "NLS Logistic Saturation": {"type":"Nonlinear",  "R2":r2_lg,     "RMSE":rmse_lg,
                                  "AIC":aic_lg,    "BIC":bic_lg,    "params":4},
    "NLS Exponential Growth" : {"type":"Nonlinear",  "R2":r2_eg,     "RMSE":rmse_eg,
                                  "AIC":aic_eg,    "BIC":bic_eg,    "params":3},
}

print("── F1: Full Model Comparison Table ───────────────────────────")
print(f"\n   {'Model':<28} {'Type':<14} {'R²/McF':>9} {'RMSE':>8} {'AIC':>10} {'BIC':>10} {'#P':>4}")
print("   " + "-"*85)
for name, info in all_models.items():
    r2s   = f"{info['R2']:.4f}"   if info['R2']   is not None and not np.isnan(info['R2'])   else " N/A"
    rmses = f"{info['RMSE']:.4f}" if info['RMSE'] is not None and not np.isnan(info['RMSE']) else " N/A"
    aics  = f"{info['AIC']:.1f}"  if info['AIC']  is not None and not np.isnan(info['AIC'])  else " N/A"
    bics  = f"{info['BIC']:.1f}"  if info['BIC']  is not None and not np.isnan(info['BIC'])  else " N/A"
    print(f"   {name:<28} {info['type']:<14} {r2s:>9} {rmses:>8} {aics:>10} {bics:>10} {info['params']:>4}")

print("── F2: Bias–Variance Tradeoff ────────────────────────────────")
print(f"""
   Core principle:
   Expected Test Error = Bias² + Variance + Irreducible Noise

   • HIGH BIAS (underfitting): Model too simple → misses true pattern
   • HIGH VARIANCE (overfitting): Model too complex → memorises noise

   Results:
   ┌─────────────────────────┬──────────┬─────────┬──────────┐
   │ Model                   │ Train R² │ Test R² │   Gap    │
   ├─────────────────────────┼──────────┼─────────┼──────────┤
   │ SLR (1 predictor)       │ {r2_slr:.4f}   │ (full)  │  high bias│
   │ MLR (16 predictors)     │ {r2_tr_mlr:.4f}   │ {r2_te_mlr:.4f}  │  {r2_tr_mlr-r2_te_mlr:+.4f}  │
   │ Polynomial deg=3        │ {r2_poly3:.4f}   │ (full)  │ moderate │
   │ Polynomial deg=8        │ {r2_tr_p8:.4f}   │ {r2_te_p8:.4f}  │  {r2_tr_p8-r2_te_p8:+.4f}  │
   └─────────────────────────┴──────────┴─────────┴──────────┘

   Diagnosis:
   • SLR: High bias — only 1 predictor, underfits the data
   • MLR: Excellent balance — small train/test gap = {r2_tr_mlr-r2_te_mlr:.4f}
   • Poly deg=8: {'Overfitting detected' if r2_tr_p8-r2_te_p8 > 0.02 else 'Modest gap'}
     (complex model may memorise training noise)
   • NLS (3–4 params): Parsimonious and physically motivated → least overfit risk
""")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Part F — Comprehensive Model Comparison",
             fontsize=15, fontweight='bold')

# ── Regression models only ────────────────────────────────────────────────────
reg_names  = ["SLR","MLR","Poly deg=1","Poly deg=3","NLS Logistic","NLS Exp."]
reg_aic    = [aic_slr, aic_mlr,
              results_poly[1]["aic"], aic_poly3,
              aic_lg  if not np.isnan(aic_lg)  else 0,
              aic_eg  if not np.isnan(aic_eg)  else 0]
reg_bic    = [bic_slr, bic_mlr,
              results_poly[1]["bic"], bic_poly3,
              bic_lg  if not np.isnan(bic_lg)  else 0,
              bic_eg  if not np.isnan(bic_eg)  else 0]
reg_rmse   = [rmse_slr, rmse_mlr,
              results_poly[1]["rmse"], rmse_poly3,
              rmse_lg if not np.isnan(rmse_lg) else 0,
              rmse_eg if not np.isnan(rmse_eg) else 0]

x_pos = np.arange(len(reg_names))
colors_bar = ["#2C5F8A","#4CAF89","#E8834B","#E8834B","#9B59B6","#D64045"]

# ① AIC & BIC
ax = axes[0]
ax.bar(x_pos - 0.2, reg_aic, 0.38, label="AIC",
       color=colors_bar, edgecolor="white", alpha=0.85)
ax.bar(x_pos + 0.2, reg_bic, 0.38, label="BIC",
       color=colors_bar, edgecolor="white", alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(reg_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("AIC / BIC  (lower = better)")
ax.set_title("AIC & BIC Comparison\n(solid=AIC, faded=BIC)")
ax.legend(["AIC","BIC"]); ax.grid(axis='y', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ② RMSE
ax = axes[1]
bars = ax.bar(reg_names, reg_rmse, color=colors_bar,
              edgecolor="white", alpha=0.88)
for bar, val in zip(bars, reg_rmse):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
            f"{val:.2f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticklabels(reg_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("RMSE  (lower = better)")
ax.set_title("RMSE Comparison\n(regression models on BMI)")
ax.grid(axis='y', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ③ Bias-Variance: Train vs Test R²
ax = axes[2]
bv_labels = ["SLR\n(train)","MLR\n(train)","MLR\n(test)",
             "Poly8\n(train)","Poly8\n(test)"]
bv_vals   = [r2_slr, r2_tr_mlr, r2_te_mlr, r2_tr_p8, r2_te_p8]
bv_colors = ["#E8834B","#2C5F8A","#4CAF89","#9B59B6","#D64045"]
bars2 = ax.bar(bv_labels, bv_vals, color=bv_colors,
               edgecolor="white", alpha=0.88)
for bar, val in zip(bars2, bv_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
            f"{val:.3f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(r2_te_mlr, color="#2C5F8A", linewidth=1.5,
           linestyle='--', alpha=0.6, label="MLR test R²")
ax.set_ylim(0, 1)
ax.set_ylabel("R²")
ax.set_title("Bias–Variance: Train vs Test R²\n(gap = overfitting risk)")
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig_f1_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_f1_comparison.png")

fig, ax = plt.subplots(figsize=(11, 7))

models_plot = {
    "SLR"              : {"flex": 1,  "interp": 10, "r2": r2_slr,   "color":"#4CAF89"},
    "MLR"              : {"flex": 4,  "interp": 7,  "r2": r2_mlr,   "color":"#2C5F8A"},
    "Logistic Reg."    : {"flex": 4,  "interp": 7,  "r2": mcfadden, "color":"#E8834B"},
    "Poisson GLM"      : {"flex": 4,  "interp": 7,  "r2": 0.3,      "color":"#E8834B"},
    "Poly deg=3"       : {"flex": 6,  "interp": 5,  "r2": r2_poly3, "color":"#9B59B6"},
    "NLS Logistic Sat.": {"flex": 7,  "interp": 8,  "r2": r2_lg,    "color":"#D64045"},
    "Poly deg=8"       : {"flex": 9,  "interp": 2,  "r2": r2_tr_p8, "color":"#888"},
}

for name, info in models_plot.items():
    r2_val = info["r2"] if not np.isnan(info["r2"]) else 0.1
    ax.scatter(info["flex"], info["interp"],
               s=r2_val * 800 + 100,
               color=info["color"], alpha=0.85, edgecolors="white", linewidth=1.5)
    ax.annotate(name,
                (info["flex"], info["interp"]),
                textcoords="offset points",
                xytext=(8, 5), fontsize=9, fontweight='bold')

ax.set_xlabel("Model Flexibility  (1=rigid → 10=very flexible)",
              fontsize=11)
ax.set_ylabel("Interpretability  (1=black box → 10=fully transparent)",
              fontsize=11)
ax.set_title("Part F — Interpretability vs Flexibility\n"
             "(bubble size = R² / McFadden R²)",
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 11); ax.set_ylim(0, 11)
ax.grid(alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Ideal zone annotation
ax.annotate("← Ideal zone\n   (high both)",
            xy=(3, 8), fontsize=10, color="#2C5F8A",
            style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAF4FF', alpha=0.7))

plt.tight_layout()
plt.savefig("fig_f2_interp_flex.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_f2_interp_flex.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Part F — Residual Comparison Across Regression Models",
             fontsize=14, fontweight='bold')

# SLR residuals
ax = axes[0]
ax.hist(y - y_pred_slr, bins=35, color="#4CAF89",
        edgecolor="white", alpha=0.85, density=True)
xr = np.linspace((y-y_pred_slr).min(), (y-y_pred_slr).max(), 200)
ax.plot(xr, stats.norm.pdf(xr, 0, (y-y_pred_slr).std()),
        color="#D64045", linewidth=2)
ax.set_title(f"SLR Residuals\nRMSE={rmse_slr:.2f}")
ax.set_xlabel("Residual"); ax.set_ylabel("Density")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# MLR residuals
ax = axes[1]
ax.hist(res_mlr, bins=35, color="#2C5F8A",
        edgecolor="white", alpha=0.85, density=True)
xr = np.linspace(res_mlr.min(), res_mlr.max(), 200)
ax.plot(xr, stats.norm.pdf(xr, 0, res_mlr.std()),
        color="#D64045", linewidth=2)
ax.set_title(f"MLR Residuals\nRMSE={rmse_mlr:.2f}")
ax.set_xlabel("Residual"); ax.set_ylabel("Density")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# NLS Logistic Saturation residuals
ax = axes[2]
if converged_lg:
    nls_res = y_bmi - y_pred_lg
    ax.hist(nls_res, bins=35, color="#9B59B6",
            edgecolor="white", alpha=0.85, density=True)
    xr = np.linspace(nls_res.min(), nls_res.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, 0, nls_res.std()),
            color="#D64045", linewidth=2)
    ax.set_title(f"NLS Logistic Sat. Residuals\nRMSE={rmse_lg:.2f}")
ax.set_xlabel("Residual"); ax.set_ylabel("Density")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig_f3_residuals.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_f3_residuals.png")

print("=" * 65)
print("  PART F — MODEL COMPARISON & CRITICAL ANALYSIS — COMPLETE")
print("=" * 65)
print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │              FULL MODEL COMPARISON SUMMARY                   │
  ├────────────────────────┬───────┬────────┬────────┬──────────┤
  │ Model                  │  R²   │  RMSE  │  AIC   │  Winner? │
  ├────────────────────────┼───────┼────────┼────────┼──────────┤
  │ SLR (1 predictor)      │{r2_slr:.3f} │{rmse_slr:.3f}  │{aic_slr:.1f}│          │
  │ MLR (16 predictors)    │{r2_mlr:.3f} │{rmse_mlr:.3f}  │{aic_mlr:.1f}│ ✅ Best  │
  │ Logistic Reg. (AUC)    │{mcfadden:.3f} │  N/A   │{aic_log:.1f}│ ✅ Class.│
  │ Poisson GLM            │  N/A  │{rmse_pois:.3f}  │{aic_pois:.1f}│ ⚠️ Overdisp│
  │ Polynomial deg=3       │{r2_poly3:.3f} │{rmse_poly3:.3f}  │{aic_poly3:.1f}│          │
  │ NLS Logistic Sat.      │{r2_lg:.3f} │{rmse_lg:.3f}  │{aic_lg:.1f}│ ✅ Parsim│
  │ NLS Exponential        │{r2_eg:.3f} │{rmse_eg:.3f}  │{aic_eg:.1f}│          │
  └────────────────────────┴───────┴────────┴────────┴──────────┘

  BIAS-VARIANCE TRADEOFF:
  ────────────────────────
  • SLR → High bias, low variance (underfits real complexity)
  • MLR → Balanced: train R²={r2_tr_mlr:.3f}, test R²={r2_te_mlr:.3f}, gap={r2_tr_mlr-r2_te_mlr:.4f}
  • Poly deg=8 → Higher variance risk (train R²={r2_tr_p8:.3f}, test R²={r2_te_p8:.3f})
  • NLS (3-4 params) → Lowest variance risk, physically interpretable

  OVERFITTING:
  ─────────────
  • High-degree polynomials overfit — AIC/BIC penalise extra params
  • MLR with 16 predictors generalises well (small train-test gap)
  • Regularisation (Ridge/Lasso) could further protect MLR

  INTERPRETABILITY vs FLEXIBILITY:
  ──────────────────────────────────
  • SLR: Most interpretable, least flexible
  • MLR: Good balance — coefficients are directly interpretable
  • NLS: Physically motivated parameters (inflection, asymptote)
  • Poly deg=8: Flexible but coefficients are meaningless

  RECOMMENDATIONS:
  ─────────────────
  ✔ Continuous response (BMI)  → Use MLR or NLS Logistic Saturation
  ✔ Binary response (diabetes) → Use Logistic Regression (AUC={auc:.3f})
  ✔ Count response (phys_hlth) → Use Negative Binomial GLM
                                  (Poisson overdispersed by {dispersion_ratio:.1f}x)

  ALL FIGURES SAVED:
  ──────────────────
  Part A: fig1–fig5_*.png
  Part B: fe2–fe7_*.png
  Part C: fig_c1_coefficients.png, fig_c2_diagnostics.png
  Part D: fig_d1_logistic.png, fig_d2_poisson.png
  Part E: fig_e1_nonlinear.png
  Part F: fig_f1_comparison.png, fig_f2_interp_flex.png,
          fig_f3_residuals.png
""")
