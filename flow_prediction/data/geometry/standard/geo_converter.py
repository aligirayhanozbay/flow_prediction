import re
import argparse

def process_file(fname):
	#patterns = ['Extrude[\s\S]*?}\n', 'Physical Volume\(\"internal\"\)']
	#replacements = ['', 'Physical Volume(\"fluid\")']
	patterns = ['Extrude[\s\S]*', '^', 'Nc = [0-9]*;', 'Ny_outer = [0-9]*;', 'Ny_inner = [0-9]*;', 'Nx_outer_right = [0-9]*;', 'Nx_outer_left = [0-9]*;', 'Nx_inner = [0-9]*;', '\{2, 21\} = Ny_inner']
	
	replacements = ["""Physical Curve("in") = {1, 2, 3};
Physical Curve("out") = {9, 8, 7};
Physical Curve("topbottom") = {4, 5, 6, 12, 11, 10};
Physical Curve("obstacle") = {28, 27, 26, 25};
Physical Surface("fluid") = {1, 2, 3, 4, 9, 5, 6, 7, 8};
	""", 'element_scale = 3;\n', 'Nc = 20*element_scale;', 'Ny_outer = 8*element_scale;', 'Ny_inner = 12*element_scale;', 'Nx_outer_right = 26*element_scale;', 'Nx_outer_left = 3*element_scale;', 'Nx_inner = 4*element_scale;', '{2, 21} = Ny_inner*0.5']
	
	with open(fname, 'r+') as f:
		contents = f.read()
		for pattern,replacement in zip(patterns,replacements):
			contents = re.sub(pattern, replacement, contents)
		f.seek(0)
		f.write(contents)
		f.truncate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser('edit .geo file to be properly 2d and compatible with pyfr')
	parser.add_argument('files', nargs ='*', type=str)
	args = parser.parse_args()
	
	for fname in args.files:
		process_file(fname)
	
