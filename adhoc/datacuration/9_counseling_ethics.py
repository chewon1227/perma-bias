import json

def process_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    processed_data = {}
    current_section = ""
    current_subsection = ""
    buffer = []

    for line in lines:
        # Clean up the line: strip leading/trailing spaces and remove '-'
        line = line.strip().replace('-', '')
        
        # Check if this line starts a new section (e.g. "A.1.", "A.1.a.", etc.)
        if line.startswith(('A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.')):
            if buffer:
                # Save the previous section or subsection
                if current_subsection:
                    processed_data[current_subsection] = ' '.join(buffer)
                else:
                    processed_data[current_section] = ' '.join(buffer)
                buffer = []
            
            # Check if it's a subsection
            if len(line.split('.')[0]) == 3:  # For subsections like "A.1.a."
                current_subsection = line
            else:  # For main sections like "A.1."
                current_section = line
                current_subsection = ""
        else:
            buffer.append(line)

    # Add the last section/subsection to the data
    if buffer:
        if current_subsection:
            processed_data[current_subsection] = ' '.join(buffer)
        else:
            processed_data[current_section] = ' '.join(buffer)

    return processed_data

def save_as_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# File path of your txt file
txt_file_path = 'data/APA_ethics.txt'

# Process the lines from the txt file
data = process_lines(txt_file_path)

# Save the processed data into JSON format
output_json_path = 'data/data9_counseling_ethics.json'
save_as_json(data, output_json_path)

print(f"Processed data saved to {output_json_path}")
print(f"Lenght of the processed data: {len(data)}")
