import re

# Exact standalone headings only; do not match these words inside prose.
END_SECTIONS = {'references', 'notes', 'notes and references', 'references and notes',
                'citations', 'bibliography', 'sources', 'further reading', 'external links', 'see also'}


def clean_article(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        heading = stripped.strip('= ').rstrip(':').casefold()
        if heading in END_SECTIONS:
            break
        if stripped.casefold().startswith(('category:', '[[category:')):
            continue
        lines.append(line.rstrip())
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
