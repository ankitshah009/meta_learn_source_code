import os
import csv
import random
import numpy as np
import sys
from tensor2tensor.data_generators import text_encoder

'''
Description: Script which contains all data related routines like generating episodes on the fly, loading data, creating dataset iterators, etc.
'''
#base directory path
base_dir = './'

#base data directory path
base_data_dir = os.path.join(base_dir, 'Preprocessed_Data')

#path of the subword vocab file
subword_vocab_filename = os.path.join(base_data_dir, 'subword_vocab.txt')

#tensor2tensor subword text encoder
encoder = text_encoder.SubwordTextEncoder(subword_vocab_filename)
######################################################################
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
#####################################################################

def print_tokens(window):
	window_str = ''
	for w in window:
		window_str+=encoder.all_subtoken_strings[w]+", "
	return window_str

def print_hole_and_window(hole, window):
	print("["+ print_tokens(hole)+"], ["+ print_tokens(window)+ "]")

def find_holes_in_blanked_range(target_holes, blanked_range):
	adjusted_target_holes=[]
	for hole in target_holes:
		if not check_valid_range((hole[0], hole[1]), blanked_range):
			a, b, c, d, e, f = hole
			e = e-1
			hole = (a,b,c,d,e,f)
		adjusted_target_holes.append(hole)
	return adjusted_target_holes

def find_support_tokens(target_sup_tokens, K=100, sup_def='vocab'):
	if sup_def =='proj':
		target_sup_tokens.sort(key= lambda x: x[5]/x[4], reverse=True)
		target_sup_tokens = target_sup_tokens[:K]
	elif sup_def=='vocab':
		target_sup_tokens.sort(key= lambda x: x[4])
		target_sup_tokens = target_sup_tokens[:K]
	elif sup_def=='random':
		K = min(len(target_sup_tokens), K)
		target_sup_tokens = random.sample(target_sup_tokens, K)
	elif sup_def =='unique':
		target_sup_tokens = list(set(target_sup_tokens))[:K]
	return target_sup_tokens

def check_valid_range(entry_range, blanked_range):
  blanked_range_min = blanked_range[0]
  blanked_range_max = blanked_range[1]
  entry_range_min = entry_range[0]
  entry_range_max = entry_range[1]

  if blanked_range_min <= entry_range_min <= blanked_range_max:
      return False
  if blanked_range_min <= entry_range_max <= blanked_range_max:
      return False
  if entry_range_min <= blanked_range_min <= entry_range_max:
      return False
  if entry_range_min <= blanked_range_max <= entry_range_max:
      return False
  return True

def check_valid_holes(entry, blanked_range):
	flag = False
	if check_valid_range((entry[0],entry[1]), blanked_range):
		flag = True
	return flag

def find_sup_window(sup_window, file_string, blanked_range):
	e1 = sup_window[0]
	e2 = sup_window[1]-1
	b1 = blanked_range[0]
	b2 = blanked_range[1]

	if b1 <= e1 <= b2 and e2 >= b2:
		mod_sup_window = file_string[b1-(b2+1-e1):b1] + file_string[b2+1:e2+1]
	elif (e1 <= b1 <= e2 and e1 <= b2 <= e2):
		mod_sup_window = file_string[e1:b1]+ file_string[b2+1:b2+1+(e2-b1)+1]
	elif b1 <= e1 <= b2 and b1 <= e2 <= b2:
		mod_sup_window = []
	else:
		mod_sup_window = file_string[e1:e2+1]
	return mod_sup_window


def get_hole_and_sup_window(file_string, hole, sup_tokens, hole_window, sup_window, blanked_range, K=100, sup_def='vocab'):
	def get_sup_prev_window(entry):
		window_range = (max(0,entry[0]-sup_window),entry[0])
		sup_prev_window = find_sup_window(window_range, file_string, blanked_range)
		return sup_prev_window

	def check_hole_validity(entry):
		return check_valid_holes(entry, blanked_range)

	hole_window = file_string[max(0, hole[0]-hole_window):hole[0]]
	target_sup_tokens = list(filter(check_hole_validity, sup_tokens))
	valid_sup_entries = find_support_tokens(target_sup_tokens, K, sup_def)
	valid_sup_tokens = [file_string[entry[0]:entry[1]] for entry in valid_sup_entries]
	sup_windows = list(map(get_sup_prev_window, valid_sup_entries))
	return hole_window, valid_sup_tokens, sup_windows

def get_hole_episodes(file_data, hole_id, hole_window=5):
	file_list = file_data[0]
	target_holes = file_data[1]
	hole = target_holes[hole_id]
	blanked_range = (hole[0], hole[2])
	hole_window = file_list[max(0, hole[0]-hole_window):hole[0]]
	target_hole = file_list[hole[0]:hole[1]]
	return target_hole, hole_window

def get_dyn_eval_episodes(file_data, hole_id, window_size):
	file_list = file_data[0]
	target_holes = file_data[1]
	hole = target_holes[hole_id] #Target Hole
	def get_prev_window(x):
		return file_list[max(0,x[0]-window_size):x[0]]

	target_hole = file_list[hole[0]:hole[1]]
	hole_window = get_prev_window(hole)
	candidate_sup_indices = target_holes[0:hole_id]
	sup_tokens = [file_list[x[0]:x[1]] for x in candidate_sup_indices]
	sup_windows = list(map(get_prev_window, candidate_sup_indices))
	return target_hole, hole_window, sup_tokens, sup_windows

def get_hole_and_sup_episodes(file_data, hole_id, mode, hole_window=200, sup_window=200, K=100, sup_def='vocab'):
	if mode =='dyn_eval':
		return get_dyn_eval_episodes(file_data, hole_id, hole_window)

	file_list = file_data[0]
	target_holes = file_data[1]
	hole = target_holes[hole_id]
	blanked_range = (hole[0], hole[2]-1)
	adjusted_target_holes = find_holes_in_blanked_range(target_holes, blanked_range)
	sup_tokens = adjusted_target_holes.copy()
	sup_tokens.pop(hole_id)
	hole_window, valid_sup_tokens, sup_windows = get_hole_and_sup_window(file_list, hole, sup_tokens, hole_window, sup_window, blanked_range, K, sup_def)

	# print("************************************************************")
	# print("TARGET HOLE, HOLE PREV WINDOW")
	# print("*************************************************************")
	# print_hole_and_window(file_list[hole[0]:hole[1]], hole_window)

	# print("\n **********************************************************")
	# print("FILE support HOLE, HOLE PREV WINDOW")
	# print("*************************************************************")
	# for i in range(len(valid_sup_tokens)):
	#  	print_hole_and_window(valid_sup_tokens[i], sup_windows[i])
	#  	print("\n")
	target_hole = file_list[hole[0]:hole[1]]
	return target_hole, hole_window, valid_sup_tokens, sup_windows