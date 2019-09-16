#coding:utf-8
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model import Model
from data import getTrainGenerator
from edge_hold_loss import EdgeHoldLoss
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lr_scheduler(epoch,base_lr):
    drop = 0.5
    epoch_drop = epochs / 8.
    lr = base_lr * math.pow(drop, math.floor((1+epoch)/epoch_drop))
    print('lr: %f'%lr)
    return lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch version of cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection')
    parser.add_argument('--train_file',default='train.txt',help='your train file', type=str)
    parser.add_argument('--test_file',default='train.txt',help='your test file', type=str)
    parser.add_argument('--model_weights',default='model/vgg16_no_top.pth',help='your model weights', type=str)
    parser.add_argument('--log_interval',default=10,help='step interval between showing logs', type=int)
    parser.add_argument('--save_interval',default=5,help='epoch interval between saving model', type=int)
    parser.add_argument('--pretrained',default=False,help='whether load pretrained weights')
    '''
    the form of 'train_pair.txt' is 
    img_path1 gt_path1\n
    img_path2 gt_path2\n 
    '''
    args = parser.parse_args()
    model_name = args.model_weights
    train_path = args.train_file
    test_path = args.test_file
    print("train_file:", train_path)
    print("test_file:", test_path)
    print("model_weights:", model_name)

    #model config
    target_size = (256,256)
    batch_size = 5
    base_lr = 1e-2
    epochs = 50
    threshold = 0.5
    f = open(train_path, 'r')
    trainlist = f.readlines()
    f.close()
    steps_per_epoch = len(trainlist) // batch_size
    if len(trainlist) % batch_size != 0:
        steps_per_epoch += 1
    f = open(test_path, 'r')
    testlist = f.readlines()
    f.close()
    test_steps = len(testlist) // batch_size
    if len(testlist) % batch_size != 0:
        test_steps += 1

    dropout = True
    with_CA = True
    with_SA = True

    #build model
    model = Model(dropout=dropout,with_CA=with_CA,with_SA=with_SA)
    model.to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(model_name))
    loss_f = EdgeHoldLoss().to(device)

    if target_size[0] % 32 != 0 or target_size[1] % 32 != 0:
        raise ValueError('Image height and wight must be a multiple of 32')
    #data generator
    traingen = getTrainGenerator(train_path, target_size, batch_size, israndom=True)
    testgen = getTrainGenerator(test_path, target_size, batch_size, israndom=False)
    i = 0
    global_Fb = 0
    print('start training!')
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        lr = lr_scheduler(epoch,base_lr)
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
        #optimizer = torch.optim.Adam(model.parameters(),lr = lr)
        for step in range(steps_per_epoch):
            i += 1#total steps
            optimizer.zero_grad()
            imgs, masks = traingen.__next__()
            imgs = Variable(imgs.to(device))
            masks = Variable(masks.to(device),requires_grad=False)
            outputs = model(imgs)
            loss = loss_f(outputs,masks)
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                secs = time.time()-start_time
                print('TIME[%02d:%02d:%02d] EPOCH[%d/%d] STEP[%d/%d] loss: %f'%(secs//3600, secs//60%60, secs%60, epoch+1, epochs, step+1, steps_per_epoch, loss.item()))
        if (epoch+1) % args.save_interval == 0:
            print('start validating!')
            model.eval()
            TP, TN, FN, FP = 0, 0, 0, 0
            for step in range(test_steps):
                imgs, masks = testgen.__next__()
                imgs = Variable(imgs.to(device))
                masks = masks.view((-1))
                outputs = model(imgs)
                preds = nn.Sigmoid()(outputs).view((-1))
                preds = preds > threshold
                preds = preds.cpu().numpy()
                masks = masks.cpu().numpy()
                TP += ((preds == 1) & (masks == 1)).sum()
                TN += ((preds == 0) & (masks == 0)).sum()
                FN += ((preds == 0) & (masks == 1)).sum()
                FP += ((preds == 1) & (masks == 0)).sum()
                #print(TP,TN,FN,FP)
                p = TP / (TP + FP)
                r = TP / (TP + FN)
                Fb = 1.3 * r * p / (r + 0.3 * p)
                acc = (TP + TN) / (TP + TN + FP + FN)
                if (step+1) % args.log_interval == 0:
                    print('VAL STEP[%d/%d] precision: %.3f, recall: %.3f, Fb score: %.3f, acc: %.3f'%(step+1, test_steps,p,r,Fb,acc))
                    f = open('result.txt','a+')
                    f.writelines('EPOCH[%d] VAL STEP[%d/%d] precision: %.3f, recall: %.3f, Fb score: %.3f, acc: %.3f'%(epoch, step+1, test_steps,p,r,Fb,acc)+'\n')
                    f.close()
            if Fb > global_Fb:
                print('get better performance from %.3f to %.3f), saving model...'%(global_Fb,Fb))
                global_Fb = Fb
                torch.save(model.state_dict(),model_name)






