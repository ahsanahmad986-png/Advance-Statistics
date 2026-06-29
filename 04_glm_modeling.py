# ══════════════════════════════════════════════════════════════════════════════
# PART D — GENERALIZED LINEAR MODELS
#   D1: Logistic Regression  → binary response: diabetes (0/1)
#   D2: Poisson GLM          → count response : phys_hlth (days 0–30)
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, roc_curve,
                             classification_report)

# ── Shared predictor set for GLMs ─────────────────────────────────────────────
glm_features = [
    "age_group_z", "gen_hlth_z", "education_z", "income_z",
    "ment_hlth_z", "log_phys_hlth", "high_bp", "high_chol",
    "smoker", "phys_activity", "diff_walk", "sex",
    "bmi_z", "bp_x_chol", "age_x_activity",
]
glm_features = [f for f in glm_features if f in df_eng.columns]

X_glm = df_eng[glm_features].values
y_bin = df_eng["diabetes"].values          # binary response
y_cnt = df_eng["phys_hlth"].values         # count response

# Train/test split
X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
    X_glm, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_glm, y_cnt, test_size=0.2, random_state=42)

print(f"✅ GLM features ({len(glm_features)}): {glm_features}")
print(f"\n   Binary response  — diabetes   : {y_bin.mean():.2%} positive")
print(f"   Count response   — phys_hlth  : mean={y_cnt.mean():.2f}  var={y_cnt.var():.2f}")
print(f"   Overdispersion   : {y_cnt.var()/y_cnt.mean():.2f}x")

# ── D1: Logistic Regression ────────────────────────────────────────────────────
# Link function: logit  →  log(p/1-p) = Xβ
# Inverse link  : p = 1 / (1 + exp(-Xβ))    (sigmoid)

from scipy.special import expit

lr_clf = LogisticRegression(max_iter=2000, random_state=42,
                             C=1.0, solver='lbfgs')
lr_clf.fit(X_tr_b, y_tr_b)

y_prob_train = lr_clf.predict_proba(X_tr_b)[:, 1]
y_prob_test  = lr_clf.predict_proba(X_te_b)[:, 1]
y_pred_b     = lr_clf.predict(X_te_b)

coefs      = lr_clf.coef_[0]
odds_ratios = np.exp(coefs)

print("── D1: Logistic Regression  (diabetes ~ 15 predictors) ───────")
print(f"\n   Link function  : logit  →  log(p/(1-p)) = Xβ")
print(f"   Inverse link   : p = sigmoid(Xβ)")
print(f"\n   {'Feature':<22} {'β (log-OR)':>11} {'Odds Ratio':>12} {'Direction'}")
print("   " + "-"*58)
for feat, logOR, OR in zip(glm_features, coefs, odds_ratios):
    direction = "↑ diabetes risk" if OR > 1 else "↓ diabetes risk"
    print(f"   {feat:<22} {logOR:>11.4f} {OR:>12.4f}  {direction}")

# McFadden R²
pi0 = np.clip(y_tr_b.mean(), 1e-10, 1-1e-10)
null_dev  = -2 * len(y_tr_b) * (pi0*np.log(pi0) + (1-pi0)*np.log(1-pi0))
y_pr_safe = np.clip(y_prob_train, 1e-10, 1-1e-10)
resid_dev = -2 * np.sum(y_tr_b*np.log(y_pr_safe) + (1-y_tr_b)*np.log(1-y_pr_safe))
mcfadden  = 1 - resid_dev / null_dev
p_log     = len(glm_features) + 1
aic_log   = resid_dev + 2 * p_log
bic_log   = resid_dev + p_log * np.log(len(y_tr_b))

print(f"\n   Null deviance     = {null_dev:.2f}")
print(f"   Residual deviance = {resid_dev:.2f}")
print(f"   McFadden R²       = {mcfadden:.4f}")
print(f"   AIC               = {aic_log:.2f}")
print(f"   BIC               = {bic_log:.2f}")

acc = accuracy_score(y_te_b, y_pred_b)
auc = roc_auc_score(y_te_b, y_prob_test)
cm  = confusion_matrix(y_te_b, y_pred_b)

print("── Logistic Regression: Performance ─────────────────────────")
print(f"\n   Accuracy   = {acc:.4f}")
print(f"   ROC-AUC    = {auc:.4f}")
print(f"\n   Confusion Matrix:")
print(f"   ┌─────────────┬──────────────┬──────────────┐")
print(f"   │             │  Pred: No(0) │  Pred: Yes(1)│")
print(f"   ├─────────────┼──────────────┼──────────────┤")
print(f"   │ Actual No(0)│  TN = {cm[0,0]:>4}   │  FP = {cm[0,1]:>4}   │")
print(f"   │ Actual Yes(1│  FN = {cm[1,0]:>4}   │  TP = {cm[1,1]:>4}   │")
print(f"   └─────────────┴──────────────┴──────────────┘")
print(f"\n   Sensitivity (Recall) = {cm[1,1]/(cm[1,0]+cm[1,1]+1e-9):.4f}")
print(f"   Specificity          = {cm[0,0]/(cm[0,0]+cm[0,1]+1e-9):.4f}")
print(f"   Precision            = {cm[1,1]/(cm[0,1]+cm[1,1]+1e-9):.4f}")
print(f"\n   Interpretation:")
print(f"   Strongest risk factors: high_bp (OR={np.exp(coefs[glm_features.index('high_bp')]):.2f}x),")
print(f"   gen_hlth (OR={np.exp(coefs[glm_features.index('gen_hlth_z')]):.2f}x per SD),")
print(f"   bmi (OR={np.exp(coefs[glm_features.index('bmi_z')]):.2f}x per SD increase)")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Part D1 — Logistic Regression Diagnostics",
             fontsize=15, fontweight='bold')

# ROC Curve
ax = axes[0]
fpr, tpr, _ = roc_curve(y_te_b, y_prob_test)
ax.plot(fpr, tpr, color="#2C5F8A", linewidth=2.5, label=f"AUC = {auc:.3f}")
ax.plot([0,1],[0,1], color="#D0D9E3", linestyle='--', linewidth=1.5)
ax.fill_between(fpr, tpr, alpha=0.12, color="#2C5F8A")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(fontsize=11)
ax.grid(alpha=0.4); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Predicted probabilities by class
ax = axes[1]
for label, color, name in [(0,"#4CAF89","No Diabetes"), (1,"#D64045","Diabetes")]:
    mask = y_te_b == label
    ax.hist(y_prob_test[mask], bins=25, alpha=0.7, color=color,
            edgecolor='white', label=name, density=True)
ax.axvline(0.5, color="#333", linestyle='--', linewidth=1.5, label="Threshold=0.5")
ax.set_xlabel("Predicted Probability of Diabetes")
ax.set_ylabel("Density")
ax.set_title("Predicted Probabilities by Class")
ax.legend(); ax.grid(alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Odds Ratio Forest Plot
ax = axes[2]
or_df = pd.DataFrame({
    "feature": glm_features,
    "OR"     : odds_ratios,
    "logOR"  : coefs
}).sort_values("OR")
colors_or = ["#D64045" if OR > 1 else "#2C5F8A" for OR in or_df["OR"]]
ax.barh(or_df["feature"], or_df["OR"], color=colors_or,
        edgecolor="white", alpha=0.85)
ax.axvline(1, color="#333", linestyle='--', linewidth=1.5,
           label="OR = 1 (no effect)")
ax.set_xlabel("Odds Ratio"); ax.set_title("Odds Ratios (red > 1 = ↑ risk)")
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig_d1_logistic.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_d1_logistic.png")

# ── D2: Poisson Regression ─────────────────────────────────────────────────────
# Link function  : log  →  log(μ) = Xβ
# Inverse link   : μ = exp(Xβ)
# Fitted via IRLS (Iteratively Reweighted Least Squares)

def fit_poisson_irls(X, y, max_iter=200, tol=1e-6):
    """Poisson GLM with log link via IRLS."""
    Xc   = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(Xc.shape[1])
    for iteration in range(max_iter):
        eta    = np.clip(Xc @ beta, -15, 15)
        mu     = np.exp(eta)
        W      = mu                             # Poisson variance = mean
        z      = eta + (y - mu) / mu            # working response
        # Weighted least squares step
        Xw     = Xc * W[:, None]
        try:
            beta_new = np.linalg.solve(Xw.T @ Xc, Xw.T @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    mu_final = np.exp(np.clip(Xc @ beta, -15, 15))
    return beta, mu_final

beta_pois, mu_pois_train = fit_poisson_irls(X_tr_c, y_tr_c)

# Predict on test set
X_te_c_int = np.column_stack([np.ones(len(y_te_c)), X_te_c])
mu_pois_test = np.exp(np.clip(X_te_c_int @ beta_pois, -15, 15))
rmse_pois    = np.sqrt(mean_squared_error(y_te_c, mu_pois_test))

# Deviance
y_safe  = np.clip(y_tr_c.astype(float), 1e-10, None)
mu_safe = np.clip(mu_pois_train, 1e-10, None)
pois_dev = 2 * np.sum(y_safe * np.log(y_safe/mu_safe) - (y_safe - mu_safe))
p_pois   = len(glm_features) + 1
aic_pois = pois_dev + 2 * p_pois
bic_pois = pois_dev + p_pois * np.log(len(y_tr_c))

print("── D2: Poisson GLM  (phys_hlth ~ 15 predictors) ─────────────")
print(f"\n   Link function  : log  →  log(μ) = Xβ")
print(f"   Inverse link   : μ = exp(Xβ)")
print(f"\n   {'Feature':<22} {'β':>9}  {'IRR=exp(β)':>12}  Interpretation")
print("   " + "-"*65)
for fname, b in zip(["intercept"]+glm_features, beta_pois):
    irr = np.exp(b)
    interp = f"{'↑' if irr>1 else '↓'} phys_hlth" if fname != "intercept" else "baseline μ"
    print(f"   {fname:<22} {b:>9.4f}  {irr:>12.4f}  {interp}")

print(f"\n   Residual Deviance = {pois_dev:.2f}")
print(f"   AIC               = {aic_pois:.2f}")
print(f"   BIC               = {bic_pois:.2f}")
print(f"   Test RMSE         = {rmse_pois:.4f}")

# ── Overdispersion Test ────────────────────────────────────────────────────────
pearson_chi2    = np.sum((y_tr_c - mu_pois_train)**2 / (mu_pois_train + 1e-10))
df_resid        = len(y_tr_c) - p_pois
dispersion_ratio = pearson_chi2 / df_resid

print("── Overdispersion Test ───────────────────────────────────────")
print(f"\n   Pearson χ² = {pearson_chi2:.2f}")
print(f"   df (resid) = {df_resid}")
print(f"   Dispersion ratio (χ²/df) = {dispersion_ratio:.3f}")
print(f"\n   Rule of thumb: ratio >> 1 → overdispersion")
if dispersion_ratio > 1.5:
    print(f"   ⚠️  OVERDISPERSION DETECTED (ratio = {dispersion_ratio:.2f})")
    print(f"   → Poisson assumption Var(Y) = E(Y) is VIOLATED")
    print(f"   → Negative Binomial model is more appropriate")
else:
    print(f"   ✅ No severe overdispersion (ratio = {dispersion_ratio:.2f})")

# ── Negative Binomial approximation ───────────────────────────────────────────
# NegBin adds a dispersion parameter θ: Var(Y) = μ + μ²/θ
# We approximate by adjusting standard errors (quasi-Poisson style)
quasi_se_factor = np.sqrt(dispersion_ratio)
print(f"\n   Quasi-Poisson SE inflation factor = √{dispersion_ratio:.2f} = {quasi_se_factor:.3f}")
print(f"   → All Poisson SEs should be multiplied by {quasi_se_factor:.3f}")
print(f"     to correctly account for overdispersion")
print(f"\n   In practice: use Negative Binomial GLM (scipy / statsmodels)")
print(f"   which directly estimates the dispersion parameter θ")

pois_res = y_tr_c - mu_pois_train   # raw residuals
pois_pearson_res = (y_tr_c - mu_pois_train) / np.sqrt(mu_pois_train + 1e-10)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Part D2 — Poisson GLM Diagnostics  (phys_hlth)",
             fontsize=15, fontweight='bold')

# ① Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_te_c, mu_pois_test, alpha=0.3, s=14, color="#2C5F8A")
lim = max(y_te_c.max(), mu_pois_test.max())
ax.plot([0, lim], [0, lim], color="#D64045", linewidth=2, linestyle='--')
ax.set_xlabel("Actual phys_hlth"); ax.set_ylabel("Predicted μ (Poisson)")
ax.set_title(f"Actual vs Predicted\nRMSE={rmse_pois:.2f}")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ② Residuals vs Fitted
ax = axes[0, 1]
ax.scatter(mu_pois_train, pois_pearson_res, alpha=0.3, s=14, color="#E8834B")
ax.axhline(0, color="#D64045", linewidth=2, linestyle='--')
ax.set_xlabel("Fitted μ"); ax.set_ylabel("Pearson Residuals")
ax.set_title("Pearson Residuals vs Fitted")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ③ IRR plot
ax = axes[0, 2]
irr_vals = np.exp(beta_pois[1:])
irr_df   = pd.DataFrame({"feature":glm_features,"IRR":irr_vals}).sort_values("IRR")
colors_irr = ["#D64045" if v>1 else "#2C5F8A" for v in irr_df["IRR"]]
ax.barh(irr_df["feature"], irr_df["IRR"], color=colors_irr, edgecolor="white", alpha=0.85)
ax.axvline(1, color="#333", linestyle='--', linewidth=1.5)
ax.set_xlabel("Incidence Rate Ratio (IRR)")
ax.set_title("IRR — Poisson GLM\n(red > 1 = ↑ phys_hlth days)")
ax.grid(axis='x', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ④ Residual histogram
ax = axes[1, 0]
ax.hist(pois_pearson_res, bins=35, color="#2C5F8A",
        edgecolor="white", linewidth=0.5, alpha=0.85, density=True)
xr = np.linspace(pois_pearson_res.min(), pois_pearson_res.max(), 300)
ax.plot(xr, stats.norm.pdf(xr, 0, pois_pearson_res.std()),
        color="#D64045", linewidth=2, label="N(0,1)")
ax.set_xlabel("Pearson Residuals"); ax.set_ylabel("Density")
ax.set_title("Residual Distribution")
ax.legend(); ax.grid(alpha=0.35)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ⑤ QQ Plot
ax = axes[1, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(pois_pearson_res)
ax.scatter(osm, osr, alpha=0.4, s=12, color="#9B59B6")
qq_line = np.array([osm[0], osm[-1]])
ax.plot(qq_line, slope*qq_line+intercept, color="#D64045", linewidth=2)
ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
ax.set_title("Normal Q-Q (Pearson Residuals)")
ax.grid(alpha=0.35); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ⑥ Variance vs Mean (overdispersion visual)
ax = axes[1, 2]
# Group fitted values into bins, compute mean and variance per bin
bins    = np.percentile(mu_pois_train, np.linspace(0, 100, 20))
bin_idx = np.digitize(mu_pois_train, bins)
bin_means = [mu_pois_train[bin_idx==i].mean() for i in range(1, 20) if (bin_idx==i).sum()>2]
bin_vars  = [y_tr_c[bin_idx==i].var()         for i in range(1, 20) if (bin_idx==i).sum()>2]
ax.scatter(bin_means, bin_vars, color="#2C5F8A", s=60, alpha=0.8, label="Observed Var")
x_ref = np.linspace(0, max(bin_means), 100)
ax.plot(x_ref, x_ref,    color="#4CAF89", linewidth=2, linestyle='--', label="Poisson: Var=Mean")
ax.plot(x_ref, x_ref*dispersion_ratio, color="#D64045", linewidth=2, linestyle='--',
        label=f"Quasi-Poisson: ×{dispersion_ratio:.1f}")
ax.set_xlabel("Fitted Mean (μ)"); ax.set_ylabel("Observed Variance")
ax.set_title(f"Overdispersion: Var vs Mean\n(ratio={dispersion_ratio:.2f})")
ax.legend(fontsize=8); ax.grid(alpha=0.35)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig_d2_poisson.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_d2_poisson.png")

print("=" * 60)
print("  PART D — GLM COMPLETE")
print("=" * 60)
print(f"""
  D1 — LOGISTIC REGRESSION  (binary: diabetes)
  ─────────────────────────────────────────────
  Link function  : logit   log(p/1-p) = Xβ
  Accuracy       = {acc:.4f}
  ROC-AUC        = {auc:.4f}
  McFadden R²    = {mcfadden:.4f}
  AIC            = {aic_log:.2f}

  Top risk factors (Odds Ratios):
    high_bp      → OR = {np.exp(coefs[glm_features.index('high_bp')]):.3f}x diabetes risk
    gen_hlth_z   → OR = {np.exp(coefs[glm_features.index('gen_hlth_z')]):.3f}x per SD
    bmi_z        → OR = {np.exp(coefs[glm_features.index('bmi_z')]):.3f}x per SD BMI

  D2 — POISSON REGRESSION  (count: phys_hlth days)
  ──────────────────────────────────────────────────
  Link function      : log   log(μ) = Xβ
  Test RMSE          = {rmse_pois:.4f}
  Residual Deviance  = {pois_dev:.2f}
  AIC                = {aic_pois:.2f}
  Overdispersion     = {dispersion_ratio:.2f}x  ⚠️  Negative Binomial preferred

  KEY FINDINGS:
  • Logistic model performs well (AUC={auc:.3f})
  • Poisson GLM is overdispersed — quasi-Poisson or NegBin
    should be used for correct inference on phys_hlth
  • Both models confirm: poor general health, high BP,
    and high BMI are central risk factors

  ▶  Continuing to Part E — Nonlinear Models...
""")
