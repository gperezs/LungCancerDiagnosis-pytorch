import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
import sklearn.metrics as skm
import os
import time
import imageio

import matplotlib.pyplot as plt

def test(model, test_loader, args):
    args.cuda = torch.cuda.is_available()
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
    return output, target.data.eq(1).cpu().numpy()

def prep_dataset(data,label):
    data = np.transpose(data, (4, 3, 0, 1, 2))
    #plt.imshow(data[0,0,12,:,:])
    #plt.show()
    
    label = label[0:len(data)]

    t2data = torch.from_numpy(data).float()
    t2label = torch.from_numpy(np.transpose(label)).int()
    test = data_utils.TensorDataset(t2data, t2label)
    return test

def apply_nms(Sc, vol_dir, cand_dir, in_dir, out_dir, mosaic_dir):
    D = os.listdir(cand_dir)
    D.sort()

    r = 20 #radio for NMS
    sz = 12 #size for validation of candidates (not too close to borders of the vol)
    nsc = 30 #for mosaics and saving

    for i in range(len(D)):
        start_time = time.time()
        pt = np.load(os.path.join(in_dir, D[i])) #load patient slices (ZZx24x24x24)
        pt = np.transpose(pt, (0, 1, 3, 2))
        cand = np.load(os.path.join(cand_dir, D[i])) #load patient candidates of 2000
        vol = np.load(os.path.join(vol_dir, D[i])) #load patient volume of 2000

        pt_id = np.where(Sc['arr_3'] == (i+1)*1.0) #extract ids of each patient's nodules from scores
        pt_scores = Sc['arr_1'][pt_id,1]
        sorted_scores = -np.sort(-pt_scores) #sort in descending order
        sorted_scores_idx = np.argsort(-pt_scores) #indexes of the sorted array
        patient = Sc['arr_0'][i]

        #Good centroid candidates
        tz, tx, ty = np.shape(vol)
        candsx = cand[:,1]
        candsy = cand[:,2]
        candsz = cand[:,0]
        good = np.where(np.logical_and(np.logical_and(candsx > sz , (tx - candsx) > sz) ,
             np.logical_and(np.logical_and(candsy > sz , (ty - candsy) > sz) ,
             np.logical_and(candsz > sz , (tz - candsz) > sz))))
        cand = cand[good,:]
        cand = cand.reshape(np.shape(cand)[1],np.shape(cand)[2])

        cand = cand[sorted_scores_idx[0,:]] #sorted cands
        data = pt[sorted_scores_idx[0,:]] #sorted data (slices)

        #Start of non-maximum suppressing
        cen_vol = np.zeros(np.shape(vol)) #centroid accumulating vol 
        dilNMS = cen_vol
        allNMS_idx = [] #for maximum indexes
        for j in range(len(cand)):
            if dilNMS[int(cand[j,0]),int(cand[j,1]),int(cand[j,2])] == 0:
                cen_vol[int(cand[j,0]),int(cand[j,1]),int(cand[j,2])] = 1
                allNMS_idx.append(j)
                #Dilation of centroid
                for ii in range(r):
                    for jj in range(r):
                       for kk in range(r):
                           if int(cand[j,0])+ii >= tz or int(cand[j,1])+jj >= tx or int(cand[j,2])+kk >= ty:
                               continue
                           if np.sqrt(ii**2 + jj**2 + kk**2) <= r:
                               dilNMS[int(cand[j,0])+ii,int(cand[j,1])+jj,int(cand[j,2])+kk] = 1
                               dilNMS[int(cand[j,0])+ii,int(cand[j,1])+jj,int(cand[j,2])-kk] = 1
                               dilNMS[int(cand[j,0])+ii,int(cand[j,1])-jj,int(cand[j,2])+kk] = 1
                               dilNMS[int(cand[j,0])+ii,int(cand[j,1])-jj,int(cand[j,2])-kk] = 1
                               dilNMS[int(cand[j,0])-ii,int(cand[j,1])+jj,int(cand[j,2])+kk] = 1
                               dilNMS[int(cand[j,0])-ii,int(cand[j,1])+jj,int(cand[j,2])-kk] = 1
                               dilNMS[int(cand[j,0])-ii,int(cand[j,1])-jj,int(cand[j,2])+kk] = 1
                               dilNMS[int(cand[j,0])-ii,int(cand[j,1])-jj,int(cand[j,2])-kk] = 1

        sorted_NMS_cand = cand[allNMS_idx]
        sorted_NMS_data = data[allNMS_idx]
        sorted_NMS_scores = sorted_scores[0, allNMS_idx]

        #Create mosaic of the 30 nodules per patient and save the image
        mosaic = np.concatenate((sorted_NMS_data[0,:,:,12],sorted_NMS_data[0,:,12,:], sorted_NMS_data[0,12,:,:]))
        if np.shape(sorted_NMS_data)[0] < nsc:
            nsc = np.shape(sorted_NMS_data)[0]
        for xx in range(nsc-1):
            X = np.concatenate((sorted_NMS_data[xx+1,:,:,12], sorted_NMS_data[xx+1,:,12,:], 
                sorted_NMS_data[xx+1,12,:,:]))
            mosaic = np.concatenate((mosaic, np.zeros([72,2]), X), axis=1)
        mosaic2 = (mosaic - np.min(mosaic))/(np.max(mosaic) - np.min(mosaic))*255
        imageio.imwrite(os.path.join(mosaic_dir, patient + '.jpg'), mosaic2.astype(np.uint8))
        nsc = 30

        #Save .npz file with sorted nodules
        filename = os.path.join(out_dir, patient)
        np.savez(filename, sorted_NMS_data, sorted_NMS_scores, sorted_NMS_cand, patient, mosaic)
        print('  subject: ' + str(i+1)  + '/' + str(len(D)) +
          ' (' + str(round(time.time() - start_time,2)) + 's)' )

