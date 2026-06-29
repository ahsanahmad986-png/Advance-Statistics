# ══════════════════════════════════════════════════════════════════════════════
# PART B — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.preprocessing import StandardScaler

df_eng = df.copy()

# ── FE1: Polynomial Features ──────────────────────────────────────────────────
# WHY: BMI may have a nonlinear (quadratic) relationship with age_group and
#      gen_hlth. Squaring captures the curve that a linear term misses.

df_eng["age_sq"]      = df_eng["age_group"] ** 2
df_eng["gen_hlth_sq"] = df_eng["gen_hlth"]  ** 2

print("── FE1: Polynomial Features ──────────────────────────────────")
print(f"   Corr(age_group,  bmi) = {df['age_group'].corr(df['bmi']):.4f}  ← linear")
print(f"   Corr(age_sq,     bmi) = {df_eng['age_sq'].corr(df_eng['bmi']):.4f}  ← quadratic")
print(f"   Corr(gen_hlth,   bmi) = {df['gen_hlth'].corr(df['bmi']):.4f}  ← linear")
print(f"   Corr(gen_hlth_sq,bmi) = {df_eng['gen_hlth_sq'].corr(df_eng['bmi']):.4f}  ← quadratic")
print("\n   ✅ age_sq and gen_hlth_sq added")

# ── FE2: Log Transformation ────────────────────────────────────────────────────
# WHY: phys_hlth and ment_hlth are zero-inflated count variables with heavy
#      right tails. log(x+1) compresses the tail and stabilises variance.
#      (+1 avoids log(0) errors)

df_eng["log_phys_hlth"] = np.log1p(df_eng["phys_hlth"])
df_eng["log_ment_hlth"] = np.log1p(df_eng["ment_hlth"])

print("── FE2: Log Transformation ───────────────────────────────────")
print(f"   phys_hlth     skewness: {df['phys_hlth'].skew():.3f}  → "
      f"log(phys+1): {df_eng['log_phys_hlth'].skew():.3f}")
print(f"   ment_hlth     skewness: {df['ment_hlth'].skew():.3f}  → "
      f"log(ment+1): {df_eng['log_ment_hlth'].skew():.3f}")
print("\n   ✅ log_phys_hlth and log_ment_hlth added")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("FE2 — Log Transformation of Count Variables", fontweight='bold')

axes[0,0].hist(df["phys_hlth"], bins=31, color="#2C5F8A", edgecolor="white", alpha=0.85)
axes[0,0].set_title(f"phys_hlth  (skew={df['phys_hlth'].skew():.2f})")
axes[0,0].set_xlabel("Days"); axes[0,0].set_ylabel("Frequency")
axes[0,0].spines['top'].set_visible(False); axes[0,0].spines['right'].set_visible(False)

axes[0,1].hist(df_eng["log_phys_hlth"], bins=31, color="#4CAF89", edgecolor="white", alpha=0.85)
axes[0,1].set_title(f"log(phys_hlth+1)  (skew={df_eng['log_phys_hlth'].skew():.2f})")
axes[0,1].set_xlabel("log(Days+1)"); axes[0,1].set_ylabel("Frequency")
axes[0,1].spines['top'].set_visible(False); axes[0,1].spines['right'].set_visible(False)

axes[1,0].hist(df["ment_hlth"], bins=31, color="#2C5F8A", edgecolor="white", alpha=0.85)
axes[1,0].set_title(f"ment_hlth  (skew={df['ment_hlth'].skew():.2f})")
axes[1,0].set_xlabel("Days"); axes[1,0].set_ylabel("Frequency")
axes[1,0].spines['top'].set_visible(False); axes[1,0].spines['right'].set_visible(False)

axes[1,1].hist(df_eng["log_ment_hlth"], bins=31, color="#4CAF89", edgecolor="white", alpha=0.85)
axes[1,1].set_title(f"log(ment_hlth+1)  (skew={df_eng['log_ment_hlth'].skew():.2f})")
axes[1,1].set_xlabel("log(Days+1)"); axes[1,1].set_ylabel("Frequency")
axes[1,1].spines['top'].set_visible(False); axes[1,1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fe2_log_transform.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FE3: Interaction Terms ─────────────────────────────────────────────────────
# WHY: The effect of physical activity on BMI may differ by age group —
#      older inactive people are at greater risk (age × phys_activity).
#      High BP combined with high cholesterol compounds cardiovascular risk
#      and both relate to BMI (high_bp × high_chol).

df_eng["age_x_activity"]  = df_eng["age_group"]  * df_eng["phys_activity"]
df_eng["bp_x_chol"]       = df_eng["high_bp"]    * df_eng["high_chol"]
df_eng["age_x_genhlth"]   = df_eng["age_group"]  * df_eng["gen_hlth"]

print("── FE3: Interaction Terms ────────────────────────────────────")
print(f"   age × phys_activity  corr with bmi      : {df_eng['age_x_activity'].corr(df_eng['bmi']):.4f}")
print(f"   high_bp × high_chol  corr with diabetes : {df_eng['bp_x_chol'].corr(df_eng['diabetes']):.4f}")
print(f"   age × gen_hlth       corr with bmi      : {df_eng['age_x_genhlth'].corr(df_eng['bmi']):.4f}")
print("\n   ✅ age_x_activity, bp_x_chol, age_x_genhlth added")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("FE3 — Interaction Terms vs Responses", fontweight='bold')

for ax, (xvar, yvar, color, title) in zip(axes, [
    ("age_x_activity", "bmi",      "#2C5F8A", "age×activity vs BMI"),
    ("bp_x_chol",      "diabetes", "#D64045", "BP×Chol vs Diabetes"),
    ("age_x_genhlth",  "bmi",      "#4CAF89", "age×genhlth vs BMI"),
]):
    ax.scatter(df_eng[xvar], df_eng[yvar], alpha=0.25, s=14, color=color)
    m, b, r, p, _ = stats.linregress(df_eng[xvar], df_eng[yvar])
    x_line = np.linspace(df_eng[xvar].min(), df_eng[xvar].max(), 200)
    ax.plot(x_line, m*x_line+b, color="#D64045", linewidth=2,
            label=f"r={r:.2f}")
    ax.set_xlabel(xvar); ax.set_ylabel(yvar)
    ax.set_title(title, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.35)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fe3_interactions.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FE4: Dummy Variable Encoding ───────────────────────────────────────────────
# WHY: gen_hlth is an ordinal variable (1=Excellent → 5=Poor). While we can
#      treat it as numeric, creating dummies makes no assumptions about
#      equal spacing between levels and often improves model fit.
#      We drop the first category (gen_hlth=1, Excellent) as reference.

df_eng["gen_hlth_cat"] = df_eng["gen_hlth"].map({
    1:"Excellent", 2:"VeryGood", 3:"Good", 4:"Fair", 5:"Poor"
})

gen_dummies = pd.get_dummies(df_eng["gen_hlth_cat"],
                              prefix="gh",
                              drop_first=True).astype(int)
df_eng = pd.concat([df_eng, gen_dummies], axis=1)

# Also encode income as dummies (8 ordinal levels)
income_dummies = pd.get_dummies(df_eng["income"],
                                 prefix="inc",
                                 drop_first=True).astype(int)
df_eng = pd.concat([df_eng, income_dummies], axis=1)

print("── FE4: Dummy Variable Encoding ──────────────────────────────")
print(f"   gen_hlth dummies : {[c for c in df_eng.columns if c.startswith('gh_')]}")
print(f"   income dummies   : {[c for c in df_eng.columns if c.startswith('inc_')]}")
print(f"   Reference category (dropped): gen_hlth=Excellent, income=1")
print("\n   ✅ Dummy variables added")

# BMI by general health category
fig, ax = plt.subplots(figsize=(10, 5))
order    = ["Excellent","VeryGood","Good","Fair","Poor"]
colors_g = ["#4CAF89","#2C5F8A","#E8834B","#D64045","#8B0000"]
data_g   = [df_eng.loc[df_eng["gen_hlth_cat"]==cat, "bmi"].values for cat in order]

bp = ax.boxplot(data_g, patch_artist=True,
                medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp['boxes'], colors_g):
    patch.set_facecolor(color); patch.set_alpha(0.85)

ax.set_xticklabels(order, fontsize=11)
ax.set_title("FE4 — BMI by General Health Category", fontweight='bold')
ax.set_ylabel("BMI"); ax.grid(axis='y', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("fe4_dummies.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FE5: Standardisation (Z-scores) ───────────────────────────────────────────
# WHY: Predictors are on very different scales (age_group: 1–13, bmi: 12–98).
#      Z-score standardisation (mean=0, std=1) makes regression coefficients
#      directly comparable and improves convergence of iterative solvers.

scale_cols = ["age_group","gen_hlth","education","income",
              "ment_hlth","phys_hlth","bmi"]

scaler     = StandardScaler()
scaled_arr = scaler.fit_transform(df_eng[scale_cols])

for i, col in enumerate(scale_cols):
    df_eng[f"{col}_z"] = scaled_arr[:, i]

print("── FE5: Standardisation (Z-scores) ───────────────────────────")
print(f"\n   {'Feature':<15} {'Mean (orig)':>12} {'Std (orig)':>11} "
      f"{'Mean (z)':>10} {'Std (z)':>9}")
print("   " + "-"*60)
for col in scale_cols:
    print(f"   {col:<15} {df_eng[col].mean():>12.3f} {df_eng[col].std():>11.3f} "
          f"{df_eng[col+'_z'].mean():>10.5f} {df_eng[col+'_z'].std():>9.5f}")
print("\n   ✅ Z-score columns added (suffix _z)")

# ── FE6: Square Root Transformation ───────────────────────────────────────────
# WHY: age_group is discrete and slightly skewed. sqrt() gently compresses
#      higher values, reducing leverage of older age groups in regression.
#      Also applied to income to reduce right-skew in ordinal scale.

df_eng["sqrt_age"]    = np.sqrt(df_eng["age_group"])
df_eng["sqrt_income"] = np.sqrt(df_eng["income"])

print("── FE6: Square Root Transformation ───────────────────────────")
print(f"   age_group    skewness: {df['age_group'].skew():.3f}  → "
      f"sqrt: {df_eng['sqrt_age'].skew():.3f}")
print(f"   income       skewness: {df['income'].skew():.3f}  → "
      f"sqrt: {df_eng['sqrt_income'].skew():.3f}")
print("\n   ✅ sqrt_age and sqrt_income added")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("FE6 — Square Root Transformation", fontweight='bold')

axes[0,0].hist(df["age_group"], bins=13, color="#2C5F8A", edgecolor="white", alpha=0.85)
axes[0,0].set_title(f"age_group  (skew={df['age_group'].skew():.2f})")
axes[0,0].set_xlabel("Age Group"); axes[0,0].set_ylabel("Frequency")
axes[0,0].spines['top'].set_visible(False); axes[0,0].spines['right'].set_visible(False)

axes[0,1].hist(df_eng["sqrt_age"], bins=20, color="#9B59B6", edgecolor="white", alpha=0.85)
axes[0,1].set_title(f"sqrt(age_group)  (skew={df_eng['sqrt_age'].skew():.2f})")
axes[0,1].set_xlabel("√Age Group"); axes[0,1].set_ylabel("Frequency")
axes[0,1].spines['top'].set_visible(False); axes[0,1].spines['right'].set_visible(False)

axes[1,0].hist(df["income"], bins=8, color="#2C5F8A", edgecolor="white", alpha=0.85)
axes[1,0].set_title(f"income  (skew={df['income'].skew():.2f})")
axes[1,0].set_xlabel("Income Level"); axes[1,0].set_ylabel("Frequency")
axes[1,0].spines['top'].set_visible(False); axes[1,0].spines['right'].set_visible(False)

axes[1,1].hist(df_eng["sqrt_income"], bins=20, color="#9B59B6", edgecolor="white", alpha=0.85)
axes[1,1].set_title(f"sqrt(income)  (skew={df_eng['sqrt_income'].skew():.2f})")
axes[1,1].set_xlabel("√Income"); axes[1,1].set_ylabel("Frequency")
axes[1,1].spines['top'].set_visible(False); axes[1,1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fe6_sqrt.png", dpi=150, bbox_inches='tight')
plt.show()

# ── VIF Calculation ────────────────────────────────────────────────────────────
# WHY: Engineered features (especially age_sq, sqrt_age, age_z) are all
#      derived from age_group — they will be highly collinear.
#      VIF > 10 signals a problem. We document this and will select
#      only one age representation per model in Parts C–F.

from sklearn.linear_model import LinearRegression

def compute_vif(X_df):
    vif_results = []
    cols = X_df.columns.tolist()
    for col in cols:
        others = [c for c in cols if c != col]
        r2 = LinearRegression().fit(
            X_df[others].values, X_df[col].values
        ).score(X_df[others].values, X_df[col].values)
        vif = 1 / (1 - r2) if r2 < 0.9999 else np.inf
        vif_results.append({"Feature": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vif_results).sort_values("VIF", ascending=False)

# Check a representative set of engineered features
vif_cols = ["age_group_z","gen_hlth_z","education_z","income_z",
            "ment_hlth_z","age_sq","log_phys_hlth","log_ment_hlth",
            "age_x_activity","bp_x_chol","sqrt_age","sqrt_income"]
vif_cols = [c for c in vif_cols if c in df_eng.columns]

vif_df = compute_vif(df_eng[vif_cols])

print("── Multicollinearity Check (VIF) ─────────────────────────────")
print(f"\n   {'Feature':<22} {'VIF':>8}  Status")
print("   " + "-"*50)
for _, row in vif_df.iterrows():
    if row['VIF'] == np.inf or row['VIF'] > 10:
        status = "⚠️  HIGH — use only one version in models"
    elif row['VIF'] > 5:
        status = "⚡ Moderate — monitor"
    else:
        status = "✅ OK"
    vif_val = f"{row['VIF']:.2f}" if row['VIF'] != np.inf else "∞"
    print(f"   {row['Feature']:<22} {vif_val:>8}  {status}")

# Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr_fe = df_eng[vif_cols].corr()
sns.heatmap(corr_fe, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("FE — Correlation Among Engineered Features",
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig("fe7_vif_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("  PART B — FEATURE ENGINEERING COMPLETE")
print("=" * 60)

engineered = ["age_sq","gen_hlth_sq","log_phys_hlth","log_ment_hlth",
              "age_x_activity","bp_x_chol","age_x_genhlth",
              "sqrt_age","sqrt_income","age_group_z","gen_hlth_z",
              "education_z","income_z","ment_hlth_z","phys_hlth_z","bmi_z"]
present = [c for c in engineered if c in df_eng.columns]

print(f"""
  Original features  : {len(df.columns)}
  Engineered added   : {len(present)}
  Total df_eng cols  : {len(df_eng.columns)}

  ┌─────┬────────────────────────────┬───────────────────────────────┐
  │ FE# │ Feature(s)                 │ Justification                 │
  ├─────┼────────────────────────────┼───────────────────────────────┤
  │  1  │ age_sq, gen_hlth_sq        │ Nonlinear quadratic effects   │
  │  2  │ log_phys_hlth,             │ Normalise zero-inflated counts│
  │     │ log_ment_hlth              │                               │
  │  3  │ age×activity, bp×chol,     │ Real interaction effects      │
  │     │ age×gen_hlth               │                               │
  │  4  │ gen_hlth dummies,          │ No equal-spacing assumption   │
  │     │ income dummies             │ for ordinal variables         │
  │  5  │ *_z (z-scores)             │ Standardise for comparability │
  │  6  │ sqrt_age, sqrt_income      │ Variance stabilisation        │
  └─────┴────────────────────────────┴───────────────────────────────┘

  ⚠ Multicollinearity note:
    age_sq and age_group_z are highly correlated — only one will be
    used per model in Parts C–F to avoid inflated standard errors.

  ▶  Continuing to Part C — Simple & Multiple Linear Regression...
""")
