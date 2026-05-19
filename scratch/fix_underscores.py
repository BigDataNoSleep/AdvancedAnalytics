import re

with open('report/chapters/1.tex', 'r') as f:
    content = f.read()

# Temporarily remove all existing escapes just to be uniform
content = content.replace(r'\_', '_')

# Escape all underscores
content = content.replace('_', r'\_')

# Restore underscores inside \includegraphics
def repl_includegraphics(m):
    return m.group(0).replace(r'\_', '_')

content = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', repl_includegraphics, content)

# Restore underscores in \label
content = re.sub(r'\\label\{.*?\}', repl_includegraphics, content)

with open('report/chapters/1.tex', 'w') as f:
    f.write(content)
