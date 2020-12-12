import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import sys
import os
from sklearn.metrics import roc_auc_score
import argparse

sys.path.insert(0, 'src/utils')
sys.path.insert(0, 'src/models')

from data_utils import (create_patients_from_dicom, candidate_extraction, 
                        create_candidate_slices, create_test_dataset)
from test_utils import test, apply_nms, prep_dataset
from predictor import Predictor
import nodule_detector

"""
This script runs test over our validation set, which corresponds to the year 2.000 CT scan 
of each patient of the training dataset of ISBI Lung challenge 2018 (30 Patients).
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch LungCancerPredictor Test')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--dicom_dir', default='data/dicom', help="path to dicom files")
    parser.add_argument('--vols_dir', default='data/volumes', help="path to patient volumes")
    parser.add_argument('--cands_dir', default='data/candidates', help="path to extracted candidates")
    parser.add_argument('--slices_dir', default='data/slices', help="path to nodule slices")
    parser.add_argument('--sorted_slices_dir', default='output/sorted_slices', help="path to nodule sorted slices")
    parser.add_argument('--sorted_slices_jpgs_dir', default='output/sorted_slices_images/', 
                        help="path to nodule sorted slices jpgs")

    parser.add_argument('--resume', default='models/model_predictor.pth', help="path to model of 5 lanes for test")
    parser.add_argument('--csv', default='submission_test.csv', help="file name to save submission csv")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ini_t_time = time.time()

    print('\ncreating patients (dicom-->npy)...')
    create_patients_from_dicom(args.dicom_dir, args.vols_dir)

    print('\nextracting candidates...')
    candidate_extraction(args.vols_dir, args.cands_dir)

    print('\ncreating candidate slices...')
    create_candidate_slices(args.vols_dir, args.cands_dir, args.slices_dir)
   
    if not args.cuda:
        print('\nrunning nodule detector (this might take a while)...')
    else:
        print('\nrunning nodule detector...')
    nodule_detector.run(args.slices_dir) #outputs scores_detector_test.npz

    print('\napplying nms and sorting slices...')
    Sc = np.load('scores_detector_test.npz')
    apply_nms(Sc, args.vols_dir, args.cands_dir, args.slices_dir, args.sorted_slices_dir, 
                                                                 args.sorted_slices_jpgs_dir)
    
    print('\ncreating top-5 nodule dataset...')
    data, label = create_test_dataset(args.sorted_slices_dir, 1)
    testds = prep_dataset(data, label)
    test_loader = data_utils.DataLoader(testds, batch_size=args.batch_size, shuffle=False)
    print('  dataset ready!')

    model = Predictor()
    if args.cuda:
        model.cuda()

    if args.cuda:
        model.load_state_dict(torch.load(args.resume))
    else:
        model.load_state_dict(torch.load(args.resume, map_location=torch.device('cpu')))

    output, label = test(model, test_loader, args) 
    output = output.data.cpu().numpy()

    print('\nlung cancer probability: ')
    for i in range(len(output)):
        print('%.2f%%'%(output[i,1].reshape(-1,1)*100))

    print('\ntotal elapsed time: %0.2f min\n' % ((time.time() - ini_t_time)/60.0))


