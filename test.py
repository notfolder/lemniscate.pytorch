import torch
import time
import datasets
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np
import lib

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).to(lib.get_dev())
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).to(lib.get_dev())

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            #targets = targets.cuda(async=True)
            targets = targets.to(lib.get_dev())
            batchSize = inputs.size(0)
            features = net(inputs.to(lib.get_dev()))
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).to(lib.get_dev())
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            #targets = targets.cuda(async=True)
            targets = targets.to(lib.get_dev())
            batchSize = inputs.size(0)
            features = net(inputs.to(lib.get_dev()))
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).to(lib.get_dev())
    else:
        #trainLabels = torch.LongTensor(trainloader.dataset.train_labels).to(lib.get_dev())
        trainLabels = torch.LongTensor(trainloader.dataset.targets).to(lib.get_dev())
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            #targets = targets.cuda(async=True)
            targets = targets.to(lib.get_dev())
            batchSize = inputs.size(0)
            features = net(inputs.to(lib.get_dev()))
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).to(lib.get_dev())
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).to(lib.get_dev())
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            #targets = targets.cuda(async=True)
            targets = targets.to(lib.get_dev())
            batchSize = inputs.size(0)
            features = net(inputs.to(lib.get_dev()))
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1/total

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def acc_score(label_true, label_pred):
    cm = confusion_matrix(label_true, label_pred)
    row_idx, col_idx = linear_sum_assignment(-cm + np.max(cm))
    cm2 = cm[np.ix_(row_idx, col_idx)]
    acc = np.trace(cm2)/np.sum(cm2)
    return acc

#def acc_score(label_true, label_pred):
#    dic = {}
#    for i in np.unique(label_pred):
#        dic[i] = np.argmax(np.bincount(label_true[label_pred == i]))
#    v = np.array(list(dic.values()))
#    sidx = np.searchsorted(list(dic), label_pred)
#    acc = accuracy_score(label_true, v[sidx])
#    return acc

#def acc_score(y_true, y_pred):
#    """
#    Calculate clustering accuracy. Require scikit-learn installed
#    # Arguments
#        y: true labels, numpy.array with shape `(n_samples,)`
#        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#    # Return
#        accuracy, in [0,1]
#    """
#    y_true = y_true.astype(np.int64)
#    assert y_pred.size == y_true.size
#    D = max(y_pred.max(), y_true.max()) + 1
#    w = np.zeros((D, D), dtype=np.int64)
#    for i in range(y_pred.size):
#        w[y_pred[i], y_true[i]] += 1
#    from scipy.optimize import linear_sum_assignment as linear_assignment
#    ind = linear_assignment(w.max() - w)
#    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def kmeans(net, testloader):
    kmeans= KMeans(n_clusters=10)
    #pred = kmeans.fit_predict()
    with torch.no_grad():
        features_all = []
        targets_all = []
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.to(lib.get_dev())
            batchSize = inputs.size(0)
            features = net(inputs.to(lib.get_dev()))
            features_all.append(features)
            targets_all.append(targets)
        features_all = torch.cat(features_all, 0)
        targets_all = torch.cat(targets_all, 0)
        features_all = features_all.to('cpu').detach().numpy().copy()
        targets_all = targets_all.to('cpu').detach().numpy().copy()
        pred = kmeans.fit_predict(features_all)
#        print(pred.shape)
#        print(targets_all.shape)
    return acc_score(targets_all, pred), normalized_mutual_info_score(targets_all, pred), adjusted_rand_score(targets_all, pred)
