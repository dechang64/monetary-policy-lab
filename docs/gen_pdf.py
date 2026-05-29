import markdown
import re

# Read the markdown file
with open('paper_v9_35page.md', 'r') as f:
    md_text = f.read()

# Convert markdown to HTML
html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'md_in_html'])

# Wrap math delimiters for better display
# Replace $...$ with inline math spans
html_body = re.sub(r'\$([^$]+)\$', r'<span class="math">\1</span>', html_body)
# Replace $$...$$ with block math
html_body = re.sub(r'\$\$([^$]+)\$\$', r'<div class="math-block">\1</div>', html_body)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 1in 1in 1in 1in;
    @bottom-center {{
        content: counter(page);
        font-size: 10pt;
        color: #666;
    }}
}}
body {{
    font-family: 'DejaVu Serif', 'Times New Roman', Georgia, serif;
    font-size: 11pt;
    line-height: 1.35;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}}
h1 {{
    font-size: 16pt;
    text-align: center;
    margin-top: 0;
    margin-bottom: 6pt;
    font-weight: bold;
    line-height: 1.2;
}}
h1 + p {{
    text-align: center;
    font-size: 11pt;
    margin: 2pt 0;
}}
h1 + p + p {{
    text-align: center;
    font-size: 10pt;
    font-style: italic;
    color: #444;
    margin: 2pt 0 20pt 0;
}}
h2 {{
    font-size: 13pt;
    margin-top: 24pt;
    margin-bottom: 8pt;
    font-weight: bold;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3pt;
}}
h3 {{
    font-size: 11.5pt;
    margin-top: 18pt;
    margin-bottom: 6pt;
    font-weight: bold;
    font-style: italic;
}}
p {{
    margin-top: 0;
    margin-bottom: 8pt;
    text-indent: 0;
}}
/* First paragraph after heading: no indent */
h2 + p, h3 + p {{
    text-indent: 0;
}}
/* Subsequent paragraphs: indent */
p + p {{
    text-indent: 24pt;
}}
/* No indent for list items */
li + li {{
    text-indent: 0;
}}
strong {{
    font-weight: bold;
}}
em {{
    font-style: italic;
}}
.math {{
    font-family: 'DejaVu Serif', serif;
    font-style: italic;
}}
.math-block {{
    text-align: center;
    margin: 12pt 0;
    font-style: italic;
    font-size: 11pt;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 10pt;
}}
th {{
    border-top: 2px solid #000;
    border-bottom: 1px solid #000;
    padding: 4pt 6pt;
    text-align: center;
    font-weight: bold;
    font-size: 9.5pt;
}}
td {{
    padding: 3pt 6pt;
    text-align: center;
    border-bottom: 1px solid #ddd;
}}
tr:last-child td {{
    border-bottom: 1px solid #000;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 20pt 0;
}}
ul, ol {{
    margin: 6pt 0 8pt 18pt;
    padding: 0;
}}
li {{
    margin-bottom: 4pt;
    text-indent: 0;
}}
code {{
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 1pt 3pt;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open('paper_v9_verified.html', 'w') as f:
    f.write(html)

print("HTML generated successfully")
