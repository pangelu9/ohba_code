input_file = 'packages_list.txt'
output_file = 'filtered_packages_list.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if 'pypi' not in line:
            outfile.write(line)
