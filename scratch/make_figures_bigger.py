import re

with open('report/chapters/1.tex', 'r') as f:
    content = f.read()

# For figures 1 to 6 in Section 3, they are currently:
# \includegraphics[width=\linewidth]{figures/1 Distribution of purchases per customer.png}
# I want to change them to:
# \makebox[\textwidth][c]{\includegraphics[width=1.2\textwidth]{figures/...}}

def repl_sect3_figs(m):
    fig_name = m.group(1)
    if any(fig_name.startswith(f"figures/{i} ") for i in [1, 2, 3, 4, 5, 6]):
        return r"\makebox[\textwidth][c]{\includegraphics[width=1.2\textwidth]{" + fig_name + r"}}"
    return m.group(0)

content = re.sub(r'\\includegraphics\[width=\\linewidth\]\{(figures/\d+.*?\.png)\}', repl_sect3_figs, content)

# For 17.1 and 17.2, they are in minipages. We will replace the whole block.
fig17_old = r"""\begin{figure}[H]
    \centering
    \begin{minipage}{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/17.1 pca_pc1_top_loadings.png}
        \caption{PC1 top loadings.}
    \end{minipage}\hfill
    \begin{minipage}{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/17.2 pca_pc2_top_loadings.png}
        \caption{PC2 top loadings.}
    \end{minipage}
\end{figure}"""

fig17_new = r"""\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/17.1 pca_pc1_top_loadings.png}
    \caption{PC1 top loadings.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/17.2 pca_pc2_top_loadings.png}
    \caption{PC2 top loadings.}
\end{figure}"""

content = content.replace(fig17_old, fig17_new)

with open('report/chapters/1.tex', 'w') as f:
    f.write(content)
