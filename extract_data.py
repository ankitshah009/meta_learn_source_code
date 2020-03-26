import os
import json
import random
import shutil
import errno
import pickle
import collections
import javac_parser
from glob import glob
from tensor2tensor.data_generators import text_encoder

'''
Description: Creates 1 percent corpus data (train, test , val and vocab). Creates individual text files and json for each of these splits.
Learns subword tokenized vocabulary from the vocab solit and tests it.
'''

#base directory path
base_dir = './'

#base data directory path. This should contain separate folders for train, test and val each containing .java files
base_data_dir = os.path.join(base_dir, 'Raw_Data')

#base data store directory path. This stores the extracted data files
base_data_store_dir = os.path.join(base_dir, 'Preprocessed_Data')

#subword vocab file
subword_vocab_filename = os.path.join(base_data_store_dir, 'subword_vocab.txt')

java = javac_parser.Java()

def java_tokenize(line):
	tokens = []
	tokens_type = []
	line = line.strip(' ').strip('\n')
	line = line.replace('\n',' ').replace('\r', ' ').replace('\t', ' ')
	parsed_line =java.lex(line)
	for i in range(len(parsed_line)-1): #use len(parsed_line)-1 to exclude EOF
		tokens.append(parsed_line[i][1].encode('ascii', 'backslashreplace').decode())
		tokens_type.append(parsed_line[i][0])
	tokenized_line = " ".join(tokens)
	tokenized_type = " ".join(tokens_type)
	#tokenized_line = tokenized_line.replace("\"","\\\"")
	return (tokenized_line, tokenized_type)

def copy(src, dest):
	"""
	Copy all files from src to dest directory
	"""
	try:
		shutil.copytree(src, dest)
	except OSError as e:
	# If the error was caused because the source wasn't a directory
		if e.errno == errno.ENOTDIR:
			shutil.copy(src, dest)
		else:
			print('Directory not copied. Error: %s' % e)

def create_vocab_split(num_vocab_proj):
	'''
	Randomly choose 1000 projects from rest of the projects excluding small train, test and val and copy them to a directory
	'''
	corpus_main_dir = os.path.join(base_data_dir, 'java_projects')
	corpus_dirs = glob(corpus_main_dir+'/*')
	not_vocab_files = os.path.join(base_data_dir, 'not_vocab_1_percent.txt')
	vocab_proj_path = os.path.join(base_data_dir, 'vocab')
	if not os.path.exists(vocab_proj_path):
		os.mkdir(vocab_proj_path)
	new_lines=[]
	lines = open(not_vocab_files, 'r').readlines()
	for line in lines:
		proj = line.strip().strip('\n')
		new_lines.append(os.path.join(corpus_main_dir, proj))
	potential_vocab_projects = [x for x in corpus_dirs if x not in new_lines]
	random.seed(42)
	vocab_projects = random.sample(potential_vocab_projects, num_vocab_proj)
	print(len(vocab_projects))
	for proj in vocab_projects:
		dest_dir = os.path.join(vocab_proj_path, proj.split('/')[-1])
		copy(proj, dest_dir)

def java_tokenize_and_write(dir_path, out_filename, type):
	'''
	Java tokenize a file and write down the corpus data into a file
	'''
	out_file = open(out_filename, 'w', encoding="utf-8")
	filenames = [y for x in os.walk(dir_path) for y in glob(os.path.join(x[0], '*.java'))]
	if type =='vocab':
		token_dict={}
	for filename in filenames:
		with open(filename, encoding="utf8", errors='backslashreplace') as f:
			lines = f.readlines()
			for line in lines:
				tokens = []
				line = line.strip().strip('\n')
				line = line.replace('\n',' ').replace('\r', ' ').replace('\t', ' ')
				if line:
					parsed_line = java.lex(line)
					for i in range(len(parsed_line)-1): #use len(parsed_line)-1 to exclude EOF
						tokens.append(parsed_line[i][1].encode('ascii', 'backslashreplace').decode()) #converts \u to \\u
				if type=='vocab':
					for token in tokens:
						if token in token_dict:
							token_dict[token]+=1
						else:
							token_dict[token]=1
				tokenized_line = " ".join(tokens)
				out_file.write(tokenized_line+"\n")
			out_file.flush()
	out_file.close()
	if type=='vocab':
		return token_dict


def get_proj_dir_file_names(data, proj_id, dir_id, file_id):
	file_name = data[proj_id]['directories'][dir_id]['files'][file_id]['file_path']
	return file_name


def build_subword_vocab(subword_vocab_file, token_counts, max_val):
	'''
	Build subword vocabulary out of the vocab split
	'''
	encoder = text_encoder.SubwordTextEncoder.build_to_target_size(2000, token_counts, 3, max_val=max_val,
				max_subtoken_length=15, num_iterations=20)
	encoder.store_to_file(subword_vocab_file)
	return encoder

def test_subword_tokenization(encoder, test_filename):
	'''
	Test subword tokenization on a test corpus
	'''
	lines = open(test_filename, encoding='utf-8', errors='backslashreplace').read()
	corpus_test = lines.replace('\n',' ').replace('\r', ' ').replace('\t', ' ')
	corpus_test = corpus_test.encode('ascii', 'backslashreplace').decode()
	encoded = encoder.encode(corpus_test)
	decoded = encoder.decode(encoded)
	assert decoded == corpus_test

def generate_json(input_folder, out_json_filename):
	'''
	Generate json file object containing info about project, directory, file, lines, tokens and token types
	'''
	proj_index = 0
	proj_dir = glob(input_folder+'/*')
	project_list=[]
	for proj in proj_dir:
		proj_index+=1
		project_name = proj.split('/')[-1]
		project_dict={}
		project_dict['project_index'] = proj_index-1
		project_dict['project_name'] = project_name+"\\"
		project_dict['project_path'] = proj+"\\"
		directories_list=[]
		filenames = [y for x in os.walk(proj) for y in glob(os.path.join(x[0], '*.java'))]
		dir_files = {}
		dir_index = 0
		file_index = 0
		for file in filenames:
			dir_name= file.split('/')[:-1]
			dir_path = '/'.join(dir_name)
			if dir_path in dir_files:
				dir_files[dir_path].append(file)
			else:
				dir_files[dir_path] = []
				dir_files[dir_path].append(file)


		dir_index=0
		for dir_path in dir_files:
			directories_dict={}
			dir_index+=1
			directories_dict['directory_index'] = dir_index-1
			directories_dict['directory_name'] = dir_path.split('/')[-1]+"\\"
			directories_dict['directory_path'] = dir_path+"\\"

			files_list=[]
			dir_file_index = 0
			for file in dir_files[dir_path]:
				files_dict={}
				dir_file_index +=1
				files_dict['file_index'] = dir_file_index-1
				files_dict['file_name'] = file.split('/')[-1]+"\\"
				files_dict['file_path'] = file+"\\"

				lines_list=[]
				lines = open(file, 'r', encoding="utf-8", errors='backslashreplace').readlines()
				file_line_index=0
				non_empty_lines = []
				for line in lines:
					line = line.strip().strip('\n')
					if line:
						non_empty_lines.append(line)

				for line in non_empty_lines:
					lines_dict={}
					file_line_index+=1
					(tokens, types) = java_tokenize(line)
					if tokens:
						lines_dict['line_index'] = file_line_index-1
						lines_dict['tokens'] = str(tokens)
						lines_dict['token_types'] = str(types)

						lines_list.append(lines_dict)

				files_dict['lines'] = lines_list
				files_list.append(files_dict)
			directories_dict['files']=files_list
			directories_list.append(directories_dict)
		project_dict['directories']=directories_list
		project_list.append(project_dict)
		print("Project " + str(proj_index))

	with open(out_json_filename, 'w', encoding='utf-8') as write_json:
		json.dump(project_list, write_json)




if __name__== "__main__":

	# Commented out portions are not required to be executed as the subword vocab and token_vocab files are already provided as part of the git repo.
	##1. Choose 1000 projects randomly to form vocab split. Should be done before lexing the vocab split.
	# print('Building vocab corpus .........')
	# create_vocab_split(1000)

	#2. Create text files for train, test, val and vocab splits
	print('Tokenizing and writing train file .........')
	java_tokenize_and_write(os.path.join(base_data_dir, 'train-lexed'), os.path.join(base_data_store_dir, 'data_train.txt'), 'train')
	print('Tokenizing and writing test file .........')
	java_tokenize_and_write(os.path.join(base_data_dir, 'test-lexed'), os.path.join(base_data_store_dir, 'data_test.txt'), 'test')
	print('Tokenizing and writing val file .........')
	java_tokenize_and_write(os.path.join(base_data_dir, 'val-lexed'), os.path.join(base_data_store_dir, 'data_val.txt'), 'val')
	# print('Tokenizing and writing vocab file .........')
	# token_dict_vocab = java_tokenize_and_write(os.path.join(base_data_dir, 'vocab-lexed'),
	# 					os.path.join(base_data_store_dir, 'data_vocab.txt'), 'vocab')


	# #3. Dump the token dict for vocab in a pickle file
	# token_dict_filename = os.path.join(base_data_store_dir, 'token_vocab.dict')
	# with open(token_dict_filename, 'wb') as f:
	# 	pickle.dump(token_dict_vocab, f)

	##4. Generate subword vocabulary using vocab split
	# print('Building Subword Vocab .........')
	# encoder = build_subword_vocab(os.path.join(base_data_store_dir, 'subword_vocab.txt'), token_dict_vocab, 10000)


	#5. Test subword tokenization by encoding and decoding a sample corpus
	encoder = text_encoder.SubwordTextEncoder(subword_vocab_filename)
	test_subword_tokenization(encoder, os.path.join(base_data_store_dir, 'data_train.txt'))
	print('Tested the subword vocab successfully........')

	#6. Create json files with hierarchy structure and token file info
	print('Creating json file for train data..........')
	generate_json(os.path.join(base_data_dir, 'train-lexed'), os.path.join(base_data_store_dir, 'data_train.json'))
	print('Creating json file for test data..........')
	generate_json(os.path.join(base_data_dir, 'test-lexed'), os.path.join(base_data_store_dir, 'data_test.json'))
	print('Creating json file for val data..........')
	generate_json(os.path.join(base_data_dir, 'val-lexed'), os.path.join(base_data_store_dir, 'data_val.json'))