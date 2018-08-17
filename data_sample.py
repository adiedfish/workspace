#-*- coding:utf-8 -*-
#专门给allan那边用的

import cPickle as pkl
import numpy as np 
import sys

read_path = "for_scp/"
read_name_f = "features"
read_name_l = "labels"

save_path = "sample_2/"
save_name_f = "sampled_features"
save_name_l = "sampled_labels"

read_lis = [1,2,3,5,6,7]

pos_to_neg = 2

for i in read_lis:
	with open("for_scp_2/"+read_name_f+str(i),'rb') as f:
		features = pkl.load(f)
	sys.stdout.write("\nfeatures read done")
	sys.stdout.flush()
	with open(read_path+read_name_l+str(i),'rb') as f:
		ys = pkl.load(f)
	sys.stdout.write("\nys read done")
	sys.stdout.flush()
	sampled_features = []
	sampled_labels = []
	rep_features = []
	rep_labels = []
	for j in range(len(ys)):
		if ys[j][0] == 1:
			sampled_labels.append(ys[j])
			sampled_features.append(features[j])
		if ys[j][1] == 1:
			rep_labels.append(ys[j])
			rep_features.append(features[j])
	num = len(sampled_labels)*pos_to_neg
	for j in range(num):
		sampled_labels.append(rep_labels[2*j])
		sampled_features.append(rep_features[2*j])
	with open(save_path+save_name_l+str(i),'wb+') as f:
		pkl.dump(sampled_labels,f)
	sys.stdout.write("\nys save done")
	sys.stdout.flush()
	with open(save_path+save_name_f+str(i),'wb+') as f:
		pkl.dump(sampled_features,f)
	sys.stdout.write("\nfeatures save done")
	sys.stdout.flush()
