import re
import sys
import os

def reorder_file(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping.")
        return

    print(f"Processing {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Extract citations in order of first appearance
    pattern = r"\\cite\{([^}]+)\}"
    matches = re.finditer(pattern, content)
    cited_keys = []
    seen = set()
    for match in matches:
        keys_str = match.group(1)
        keys = [k.strip() for k in keys_str.split(",")]
        for key in keys:
            if key not in seen:
                seen.add(key)
                cited_keys.append(key)

    # 2. Find the bibliography block
    bib_match = re.search(r"\\begin\{thebibliography\}\{99\}(.*?)\\end\{thebibliography\}", content, re.DOTALL)
    if not bib_match:
        print(f"Could not find thebibliography block in {filename}.")
        return

    bib_block_content = bib_match.group(1)

    # 3. Parse bibitem entries
    bib_entries = {}
    current_key = None
    current_text = []

    lines = bib_block_content.split("\n")
    for line in lines:
        m = re.match(r"\s*\\bibitem\{([^}]+)\}", line)
        if m:
            if current_key:
                bib_entries[current_key] = "\n".join(current_text).strip()
            current_key = m.group(1)
            current_text = [line]
        else:
            if current_key is not None:
                current_text.append(line)

    if current_key:
        bib_entries[current_key] = "\n".join(current_text).strip()

    # 4. Check if all cited keys exist in the bibliography
    missing_keys = [k for k in cited_keys if k not in bib_entries]
    if missing_keys:
        print(f"Warning: The following cited keys are missing from bibliography in {filename}: {missing_keys}")

    # 5. Reorder entries based on cited_keys
    reordered_bib_list = []
    for key in cited_keys:
        if key in bib_entries:
            reordered_bib_list.append(bib_entries[key])
            del bib_entries[key]

    # Add any remaining bibitems that were not cited in the text
    for key, entry in bib_entries.items():
        reordered_bib_list.append(entry)

    # 6. Construct new bibliography block
    # For caj_article.tex, there might not be a \addcontentsline{toc}{chapter}{\bibname} inside the bibliography itself,
    # let's see: if the original block has \addcontentsline{toc}{chapter}, we keep it, otherwise we don't.
    toc_line = ""
    if "\\addcontentsline" in bib_block_content:
        # Extract the addcontentsline line
        m_toc = re.search(r"\\addcontentsline\{toc\}\{[^}]+\}\{[^}]+\}", bib_block_content)
        if m_toc:
            toc_line = m_toc.group(0) + "\n"

    new_bib_block = "\\begin{thebibliography}{99}\n" + toc_line + "\n" + "\n\n".join(reordered_bib_list) + "\n\n\\end{thebibliography}"

    # 7. Replace in original content
    new_content = re.sub(
        r"\\begin\{thebibliography\}\{99\}.*?\\end\{thebibliography\}",
        new_bib_block.replace("\\", "\\\\"),
        content,
        flags=re.DOTALL
    )

    # 8. Save back
    with open(filename, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully reordered bibliography in {filename}!")

# Run for both files
reorder_file("thesis.tex")
reorder_file("caj_article.tex")
