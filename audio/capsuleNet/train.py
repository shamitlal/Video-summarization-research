#!/usr/bin/python3
# -*- coding: utf-8 -*-
import glob,os,sys
from model import CapsuleNet


if __name__ == '__main__':
	print("Train.py")
	print(os.path.relpath(sys.path[0],'hyperparameters.json'))
	model = CapsuleNet('hyperparameters.json')
	model.train_model()



