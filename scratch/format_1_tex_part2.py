import re

with open('report/chapters/1.tex', 'r') as f:
    content = f.read()

# Replace single figures
figures_single = [
    (r"\[IMAGE PLACEHOLDER: Figure 1 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/1 Distribution of purchases per customer.png}\n\\caption{Distribution of purchases per customer.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 2 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/2 Historical spending.png}\n\\caption{Historical spending distribution.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 3 - Section 3\.2: Recent Activity Distribution.*\]", ""),
    (r"\[IMAGE PLACEHOLDER: Figure 3 - Section 3\.2: Purchase Frequency Distribution.*\]", ""),
    (r"\[IMAGE PLACEHOLDER: Figure 3 - Section 3\.2: Recency Distribution.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/3 Recent activity purchase frequency recency distribution.png}\n\\caption{Recent activity, purchase frequency, and recency distribution.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 4 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/4 Target variable (revenue_2018_2019).png}\n\\caption{Target variable distribution.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 5 - Section 3\.3: Recency vs Future.*\]", ""),
    (r"\[IMAGE PLACEHOLDER: Figure 5 - Section 3\.3: Historical Spend.*\]", ""),
    (r"\[IMAGE PLACEHOLDER: Figure 5 - Section 3\.3: Purchase Frequency vs Future.*\]", ""),
    (r"\[IMAGE PLACEHOLDER: Figure 5 - Section 3\.3: Return Rate.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/5 Transaction Patterns.png}\n\\caption{Transaction patterns vs future revenue.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 6 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/6 Correlation with target.png}\n\\caption{Pearson correlation with target variable.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 7 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/7 model_mae_comparison.png}\n\\caption{MAE comparison bar plot.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 8 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/8 model_spearman_comparison.png}\n\\caption{Spearman correlation comparison.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 9 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/9 zero_rate_vs_mae.png}\n\\caption{Zero-rate vs MAE plot.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 10 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/10 model_prediction_correlation_heatmap.png}\n\\caption{Prediction correlation heatmap.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 11 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/11 calibration_deciles_all_models.png}\n\\caption{Calibration plot.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 12 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/12 predicted_vs_actual_revenue_analysis.png}\n\\caption{Predicted vs actual revenue scatterplots.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 13 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/13 residual analysis.png}\n\\caption{Residual boxplots.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 14 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/14 feature_importance_gain.png}\n\\caption{LightGBM gain importance plot.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 15 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/15.png}\n\\caption{LightGBM split importance plot.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 16 -.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/16 feature_importance_permutation.png}\n\\caption{Permutation importance plot.}\n\\end{figure}"),
    (r"\[IMAGE PLACEHOLDER: Figure 17 - Section 7\.4: PCA Visualisation.*\]", r"\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/17 pca_customer_feature_space.png}\n\\caption{PCA visualisation.}\n\\end{figure}"),
]

for pat, repl in figures_single:
    content = re.sub(pat, repl, content)

# Figure 17 side by side
fig17_repl = r"""\\begin{figure}[H]
    \\centering
    \\begin{minipage}{0.48\\textwidth}
        \\centering
        \\includegraphics[width=\\linewidth]{figures/17.1 pca_pc1_top_loadings.png}
        \\caption{PC1 top loadings.}
    \\end{minipage}\\hfill
    \\begin{minipage}{0.48\\textwidth}
        \\centering
        \\includegraphics[width=\\linewidth]{figures/17.2 pca_pc2_top_loadings.png}
        \\caption{PC2 top loadings.}
    \\end{minipage}
\\end{figure}"""
content = re.sub(r"\[IMAGE PLACEHOLDER: Figure 17.1 and 17.2.*\]", fig17_repl, content)

# Figure 18 side by side
fig18_repl = r"""\\begin{figure}[H]
    \\centering
    \\begin{minipage}{0.48\\textwidth}
        \\centering
        \\includegraphics[width=\\linewidth]{figures/18 tsne_customer_segments.png}
        \\caption{t-SNE customer segmentation.}
    \\end{minipage}\\hfill
    \\begin{minipage}{0.48\\textwidth}
        \\centering
        \\includegraphics[width=\\linewidth]{figures/18.1 tsne_all.png}
        \\caption{t-SNE components.}
    \\end{minipage}
\\end{figure}"""
content = re.sub(r"\[IMAGE PLACEHOLDER: Figure 18 and 18.1.*\]", fig18_repl, content)

# Table
table_content = """Include table:
Score
Public Score
Secondary score
Secondary Public Score
Entries
62.628
61.537
0.405
0.419
123"""

table_repl = r"""\\begin{table}[H]
    \\centering
    \\begin{tabular}{l c c c c}
        \\toprule
        Score & Public Score & Secondary score & Secondary Public Score & Entries \\\\
        \\midrule
        62.628 & 61.537 & 0.405 & 0.419 & 123 \\\\
        \\bottomrule
    \\end{tabular}
    \\caption{Final leaderboard submission results.}
    \\label{tab:leaderboard_results}
\\end{table}"""

content = content.replace(table_content, table_repl)

with open('report/chapters/1.tex', 'w') as f:
    f.write(content)
