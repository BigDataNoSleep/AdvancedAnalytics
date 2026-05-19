with open('report/chapters/1.tex', 'r') as f:
    content = f.read()

fig18_old = r"""\begin{figure}[H]
    \centering
    \begin{minipage}{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/18 tsne_customer_segments.png}
        \caption{t-SNE customer segmentation.}
    \end{minipage}\hfill
    \begin{minipage}{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/18.1 tsne_all.png}
        \caption{t-SNE components.}
    \end{minipage}
\end{figure}"""

fig18_new = r"""\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/18 tsne_customer_segments.png}
    \caption{t-SNE customer segmentation.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/18.1 tsne_all.png}
    \caption{t-SNE components.}
\end{figure}"""

content = content.replace(fig18_old, fig18_new)

with open('report/chapters/1.tex', 'w') as f:
    f.write(content)
