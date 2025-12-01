import re

def process_text():
    with open('extracted_content.txt', 'r') as f:
        lines = f.readlines()

    start_line = 0
    end_line = 0
    
    # Find start and end based on known markers
    for i, line in enumerate(lines):
        if "CHAPTER 2: THEORETICAL BACKGROUND" in line and "Table of Contents" not in lines[i-5]: # Avoid TOC
             # Check if it's the actual chapter start (usually has a page break before or is distinct)
             # The TOC entry has dots "......"
             if "..." not in line:
                 start_line = i
        if "CHAPTER 3: LITERATURE REVIEW" in line:
             if "..." not in line:
                 end_line = i
                 break
    
    if start_line == 0 or end_line == 0:
        print(f"Could not find start ({start_line}) or end ({end_line})")
        # Fallback based on previous grep
        # Start around 493, End around 1031
        start_line = 492
        end_line = 1030

    content_lines = lines[start_line:end_line]
    
    latex_lines = []
    for line in content_lines:
        line = line.strip()
        
        # Skip empty lines, page numbers, form feeds
        if not line:
            continue
        if line.startswith('\x0c'): # Form feed
            line = line.replace('\x0c', '')
        if re.match(r'^\d+$', line): # Page number
            continue
        if re.match(r'^\s*xi+\s*$', line, re.IGNORECASE): # Roman numerals
            continue
            
        # Format headers
        if "CHAPTER 2: THEORETICAL BACKGROUND" in line:
            latex_lines.append("\\chapter{Theoretical Background}")
            continue
            
        # Sections 2.1, 2.2 ...
        match_sec = re.match(r'^2\.\d+\s+(.+)', line)
        if match_sec:
            latex_lines.append(f"\\section{{{match_sec.group(1)}}}")
            continue
            
        # Subsections 2.4.1 ...
        match_subsec = re.match(r'^2\.\d+\.\d+\s+(.+)', line)
        if match_subsec:
            latex_lines.append(f"\\subsection{{{match_subsec.group(1)}}}")
            continue

        # Figures
        if line.startswith("Fig."):
            latex_lines.append(f"\\begin{{figure}}[ht]")
            latex_lines.append(f"\\centering")
            latex_lines.append(f"% \\includegraphics[width=0.8\\textwidth]{{placeholder}}")
            latex_lines.append(f"\\caption{{{line}}}")
            latex_lines.append(f"\\end{{figure}}")
            continue
            
        # Replace Unicode math characters
        replacements = {
            '𝑅': 'R', '𝑒': 'e', '𝐿': 'L', '𝑈': 'U', '𝑦': 'y', '𝑚': 'm', '𝑎': 'a', '𝑥': 'x',
            '𝑖': 'i', '𝑓': 'f', '𝜇': '\\mu', '𝐵': 'B', '𝜎': '\\sigma', '∈': '\\in', '𝛽': '\\beta',
            '̅': '\\bar', '√': '\\sqrt', '≥': '\\geq', '×': '\\times', '’': "'", '“': "``", '”': "''",
            '–': '-', '—': '---', '•': '\\item'
        }
        for char, repl in replacements.items():
            line = line.replace(char, repl)

        latex_lines.append(line)
        latex_lines.append("") # Add newline for paragraph separation

    with open('MScThesis/chapters/theoretical_background.tex', 'w') as f:
        f.write('\n'.join(latex_lines))

if __name__ == "__main__":
    process_text()
