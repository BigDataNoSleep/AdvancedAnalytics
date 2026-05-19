import re

with open('report/chapters/1.tex', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    line = line.strip()
    if not line:
        new_lines.append("")
        continue
    
    # Check for sections
    m_sec = re.match(r'^(\d+)\.\s+(.*)$', line)
    if m_sec:
        title = m_sec.group(2).strip()
        # Sentence case
        title = title[0].upper() + title[1:].lower() if title else ""
        new_lines.append(f"\\section{{{title}}}")
        continue
        
    m_subsec = re.match(r'^(\d+)\.(\d+)\s+(.*)$', line)
    if m_subsec:
        title = m_subsec.group(3).strip()
        # Sentence case
        title = title[0].upper() + title[1:].lower() if title else ""
        # Specific fixes
        title = title.replace("pca", "PCA").replace("t-sne", "t-SNE").replace("lightgbm", "LightGBM").replace("xgboost", "XGBoost").replace("catboost", "CatBoost").replace("mae", "MAE")
        new_lines.append(f"\\subsection{{{title}}}")
        continue

    # Other headings fix (if any subsubsection)
    m_subsubsec = re.match(r'^(\d+)\.(\d+)\.(\d+)\s+(.*)$', line)
    if m_subsubsec:
        title = m_subsubsec.group(4).strip()
        title = title[0].upper() + title[1:].lower() if title else ""
        new_lines.append(f"\\subsubsection{{{title}}}")
        continue

    new_lines.append(line)

with open('report/chapters/1.tex', 'w') as f:
    f.write('\n'.join(new_lines) + '\n')
