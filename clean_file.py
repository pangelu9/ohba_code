input_file = 'packages_list.txt'
output_file = 'cleaned_packages_list.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if '=' in line:
            # Split on first two '=' to remove the build and channel information
            package_info = line.split('=')[:2]
            outfile.write('='.join(package_info) + '\n')
