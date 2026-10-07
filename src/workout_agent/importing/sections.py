import re


def split_sections(text: str) -> dict[str, str]:
    """
    Splits a plain text file into sections based on specific header and footer markers.
    
    Sections start with '### NAME ####...' and end with a line containing only '#'.
    The blank line immediately following the header marker is skipped.
    """
    sections = {}
    
    # Regex to match the section header and capture the name
    header_pattern = re.compile(r'^### (.+?) #+$')

    # Split the text into lines and iterate through them
    lines = text.splitlines()
    current_section_name = None
    current_section_lines = []
    in_section = False
    skip_next_blank = False

    # Iterate through each line to identify sections
    for line in lines:
        # Check for section header
        header_match = header_pattern.match(line)
        if header_match:
            current_section_name = header_match.group(1)
            current_section_lines = []
            in_section = True
            skip_next_blank = True  # Flag to skip the blank line right after the header
            continue

        # Check for section footer
        if in_section and line and all(char == '#' for char in line):
            # Join the collected lines and save them under the section name
            sections[current_section_name] = '\n'.join(current_section_lines)
            in_section = False
            current_section_name = None
            continue

        # Collect content lines if inside a section
        if in_section:
            if skip_next_blank and line.strip() == '':
                skip_next_blank = False
                continue
            
            # Reset flag if the first line wasn't blank (failsafe)
            skip_next_blank = False
            current_section_lines.append(line)

        # Handle the case where the last section does not have a closing rule
        if in_section and current_section_name is not None:
            sections[current_section_name] = "\n".join(current_section_lines)

    return sections