with open('report/chapters/1.tex', 'r') as f:
    content = f.read()

keywords = ['begin', 'end', 'centering', 'includegraphics', 'caption', 'label', 'toprule', 'midrule', 'bottomrule', 'textwidth', 'hfill', 'linewidth']
for kw in keywords:
    content = content.replace(r'\\' + kw, '\\' + kw)

content = content.replace(r'\\\\', r'\\')

with open('report/chapters/1.tex', 'w') as f:
    f.write(content)
