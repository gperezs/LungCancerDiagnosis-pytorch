
import time
import argparse
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
import os

from detector import Detector

import matplotlib.pyplot as plt

def run(slid_dir):
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
    parser.add_argument('--resume', default='./models/model_detector.pth',
                        help="path to model (to continue training)")
    parser.add_argument('--outname', default='./scores_detector_test',
                        help="path to scores' file")
    
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    model = Detector()
    if args.cuda:
        model.cuda()
        model.load_state_dict(torch.load(args.resume))
    else:    
        model.load_state_dict(torch.load(args.resume, map_location=torch.device('cpu')))
    
    def test_eval(epoch):
        model.eval()
        scores = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                pred = output.data.max(1)[1]
                if batch_idx % args.log_interval == 0:
                    print('    detecting nodules: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, batch_idx * len(data), len(test_loader.dataset),
                        100. * batch_idx / len(test_loader)))
                scores.extend((output.data).cpu().numpy())
        return scores

    files = os.listdir(slid_dir)
    files.sort()
    
    dist_thresh = 10
    all_scores = []
    all_labels = []
    all_images = []
    all_patients = []
    all_cands = []
    idPat = 0
    tot_time = time.time()
    for f in range(len(files)):
        slices = np.load(os.path.join(slid_dir, files[f]))
        slices = np.transpose(slices, (0, 1, 3, 2))
       
        #for i in range(100):
        #    plt.subplot(10,10,i+1)
        #    plt.imshow(slices[11850+i,12,:,:])
        #plt.show()
        #aa = aaaa
        
        slices = np.expand_dims(slices, axis=1)
        print('\n  subject %d/%d loaded'%(f+1, len(files)))
        labels = np.zeros(len(slices))
        vdata = torch.from_numpy(slices).float()
        vlabel = torch.from_numpy(labels).int()
        testv = data_utils.TensorDataset(vdata, vlabel)
        test_loader = data_utils.DataLoader(testv, batch_size=args.batch_size, shuffle=False)
    
        scores = test_eval(f+1)
    
        all_scores.extend(scores)
        pat_name = str(files[f][:-4])
        all_patients.extend([pat_name]) #patients names
        all_labels.extend(labels) #labels
        all_images.extend((idPat+1)*np.ones(len(scores))) #patient index
        idPat += 1
    
    np.savez(args.outname, all_patients, all_scores, all_labels, all_images)

