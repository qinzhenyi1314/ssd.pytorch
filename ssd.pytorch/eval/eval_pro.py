# encoding=utf-8
################
#根据PR_statistic_multiBatch_opencv的结果生成：
#person、rider、car的
#x轴：conf
#y轴：scale（scale是gt的面积开根号）
#z轴：TP、FP、FN
#根据生成的数据画出分尺度的pr曲线、mAP与尺度的关系曲线
################
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt
import os
import io
import copy

def computIOU(A, B):
    """计算两个box的IOU

    :param box1:
    :param box2:
    :return: IOU
    """
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    if (W <= 0 or H <= 0):
        return 0
    SA = (A[2] - A[0]) * (A[3] - A[1])
    SB = (B[2] - B[0]) * (B[3] - B[1])
    cross = W * H
    iou = float(cross) / (SA + SB - cross)
    return iou

def ReadXml(xml_name, labels):
    """从xml文件中读取box信息

    :param xml_name: xml文件
    :param stLabelSet:
    :return: boxes, width, height
    """
    tree = etree.parse(xml_name) #打开xml文档
    # 得到文档元素对象
    root = tree.getroot()
    size = root.find('size')  # 找到root节点下的size节点
    width = float(size.find('width').text)  # 子节点下节点width的值
    height = float(size.find('height').text)  # 子节点下节点height的值
    gtDict = {}
    for label in labels:
        gtDict.update({label: []})
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        label = object.find('name').text
        difficult = int(object.find('difficult').text)
        if not label in labels:
            continue
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        x1 = bndbox.find('xmin').text
        y1 = bndbox.find('ymin').text
        x2 = bndbox.find('xmax').text
        y2 = bndbox.find('ymax').text
        scale = np.sqrt((float(x2) - float(x1)) * (float(y2) - float(y1)))
        gtDict[label].append([int(x1), int(y1), int(x2), int(y2), difficult, scale])
    return gtDict, width, height

#读取PR_statistic生成的结果
#数据结构是[{'filename':'xxx','person':[[]],'car':[[]],'rider':[[]]}]
def ReadResult(resultFile,labels):
    resultDictList = []
    with io.open(resultFile,encoding='gbk') as f:
        for line in f.readlines():
            resultData = line.strip().split('\t')
            resultDict = {'fileName' : resultData[0]}
            for label in labels:
                resultDict.update({label:[]})
            for i in range(0,int(resultData[1])):
                dat = resultData[i*6+2 : i*6+8]
                label = dat[5]
                if label not in labels:
                    continue
                scale = np.sqrt((float(dat[3]) - float(dat[1])) * (float(dat[4]) - float(dat[2])))
                resultDict[dat[5]].append(
                        [int(dat[1]),int(dat[2]),int(dat[3]),int(dat[4]),float(dat[0]),scale])
            resultDictList.append(resultDict)
    return resultDictList

def ScaleFilter(lstArr,scaleB,scaleU,picWidth,picHeight):
    res = []
    cof = (float(picWidth) * float(picHeight)) / ( 1920.0 * 1080.0 ) 
    scaleBtrans = float(scaleB) * cof #将1920x1080折算到实际尺度上
    scaleUtrans = float(scaleU) * cof
    for lst in lstArr:
        scale = lst[5]
        if scale >= scaleBtrans and scale < scaleUtrans:
            res.append(lst)
    return res

def ConfFilter(lstArr,confTh):
    res = []
    for lst in lstArr:
        conf = lst[4]
        if conf >= confTh:
            res.append(lst)
    return res

#读取制作画图所需的数据格式
def CreatData(dataRoot,annaFile, resultFile,saveDataName,
              iouTh,confThs,scaleBs,scaleUs,labels):
    saveData = {}
    for label in labels:
        saveData.update({label:np.zeros((len(confThs),len(scaleBs),3),dtype=np.int32)})
    saveData.update({'h_confThs':confThs})
    saveData.update({'h_scaleBs':scaleBs})
    saveData.update({'h_scaleUs':scaleUs})
    saveData.update({'h_TPNF':['TP','FP','FN']})
    saveData.update({'h_labels':labels})
    
    resultList = ReadResult(resultFile,labels)
    
    count = 0
    for resultDict in resultList:
        xmlName = dataRoot + resultDict['fileName'].replace('JPEGImages','Annotations') + '.xml'
        gtDict, width, height = ReadXml(xmlName, labels)  # 所有的ground truth boxes
        if count % 100 == 0:
            print (count)
            print (xmlName)
        count = count + 1
        for label in labels:
            for conf_i in range(0,len(confThs)):
                for scale_j in range(0,len(scaleBs)):
                    confTh = confThs[conf_i]
                    scaleB = scaleBs[scale_j]
                    scaleU = scaleUs[scale_j]
                    
                    resultEachLabel = resultDict[label]
                    gtEachLabel = gtDict[label]
                    
                    resultEachLabel = ConfFilter(resultEachLabel,confTh) #过滤conf
                    resultEachLabel = ScaleFilter(resultEachLabel,scaleB,scaleU,width,height) #过滤scale
                    gtEachLabel = ScaleFilter(gtEachLabel,scaleB,scaleU,width,height)
                    
                    matchArr = np.zeros((len(resultEachLabel),len(gtEachLabel)),dtype = np.int32)
                    for i in range(0,len(resultEachLabel)):
                        for j in range(0,len(gtEachLabel)):
                            if computIOU(resultEachLabel[i][0:4], gtEachLabel[j][0:4]) >= iouTh:
                                matchArr[i][j] = 1
                    FN = np.sum(np.sum(matchArr,axis=0) == 0) #漏检
                    TP = len(gtEachLabel) - FN #正检
                    FP = np.sum(np.sum(matchArr,axis=1) == 0) #误检
                    
                    saveData[label][conf_i][scale_j][0] = saveData[label][conf_i][scale_j][0] + TP
                    saveData[label][conf_i][scale_j][1] = saveData[label][conf_i][scale_j][1] + FP
                    saveData[label][conf_i][scale_j][2] = saveData[label][conf_i][scale_j][2] + FN
        
    np.save(saveDataName, saveData)


def CalPRmAP(saveData, label):
    PRmAPDict = {}
    confThs = saveData['h_confThs']
    scaleBs = saveData['h_scaleBs']
    scaleUs = saveData['h_scaleUs']
    labels = saveData['h_labels']
    PRmAPDict.update({'h_scaleBs':scaleBs})
    PRmAPDict.update({'h_scaleUs':scaleUs})
    PRmAPDict.update({'h_confThs':confThs})
    for label_k in range(0,len(labels)):
        label = labels[label_k]
        mAPs = []
        PRs = []
        for scale_j in range(0,len(scaleUs)):
            precisions = []
            recalls = []
            for conf_i in range(0,len(confThs)):
                TP = float(saveData[label][conf_i][scale_j][0])
                FP = float(saveData[label][conf_i][scale_j][1])
                FN = float(saveData[label][conf_i][scale_j][2])
                if TP + FP == 0 or TP + FN == 0:
                    continue
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                precisions.append(precision)
                recalls.append(recall)
            PRs.append([recalls,precisions]) #每个scale的recall、precision对儿
            recalls2 = copy.deepcopy(recalls)
            precisions2 = copy.deepcopy(precisions)
            tmp = list(zip(recalls2,precisions2))
            tmp.sort()
            recalls2[:],precisions2[:] = zip(*tmp)
            recalls_befs2 = [0.0] + recalls2[0:-1]
            recall_diffs2 = list(map(lambda x: x[0]-x[1], zip(recalls2, recalls_befs2)))
            mAP = sum(list(map(lambda x: x[0]*x[1], zip(recall_diffs2, precisions2))))
            mAPs.append(mAP)
        PRmAPDict.update({'PRs_' + label: PRs})
        PRmAPDict.update({'mAP_' + label: mAPs})
    return PRmAPDict

def DrawPR(PRmAPDict,labels,label_prefix,label_colors,scale_lines):
    confThs = PRmAPDict['h_confThs']
    scaleBs = PRmAPDict['h_scaleBs']
    scaleUs = PRmAPDict['h_scaleUs']
    for label_k in range(0,len(labels)):
        label = labels[label_k]
        dictKey = 'PRs_' + label
        for scale_j in range(0,len(scaleUs)):
            recall = PRmAPDict[dictKey][scale_j][0]
            precision = PRmAPDict[dictKey][scale_j][1]
            label_plt = label_prefix + ' ' + label + ' scale: ' + str(scaleBs[scale_j]) + ' - ' + str(scaleUs[scale_j])
            axes.plot(recall, precision,
                      scale_lines[scale_j], lw=2, color=label_colors[label_k], label = label_plt )
            plt.plot(recall, precision, 'o', color=label_colors[label_k], markersize=4)
            for conf_i in range(0,len(confThs)):
                plt.annotate(confThs[conf_i], fontsize = 6, 
                             xy = (recall[conf_i],precision[conf_i]),
                             xytext = (recall[conf_i],precision[conf_i]))

def DrawScaleMAP(PRmAPDict,labels,label_prefix,colors,point):
    scaleBs = PRmAPDict['h_scaleBs']
    scaleUs = PRmAPDict['h_scaleUs']
    xTicks = list(map(lambda x: str(x[0]) + ' - ' + str(x[1]), zip(scaleBs, scaleUs)))
    for label_k in range(0,len(labels)):
        label = labels[label_k]
        dictKey = 'mAP_' + label
        plt.plot(xTicks, PRmAPDict[dictKey], point, color=colors[label_k], markersize=10, label = label_prefix + ' ' + label) 

 
#%%定义参数
dataRoot = r'../data/VOCdevkit/VOC2007/'
iouTh = 0.5
confThs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
#scaleBs = [  0, 50,100,150,200,300]  #scaleBs指的是下限
#scaleUs = [ 50,100,150,200,300,400]  #scaleBs指的是上限
scaleBs = [ 0]  #scaleBs指的是下限
scaleUs = [ 2000000]  #scaleBs指的是上限
labels = ['person']
annaFile = r'ImageSets/Main/test.txt'
resultFile = r'eval.txt'
saveDataName = r'eval.npy'
#%%生成数据

CreatData(dataRoot,annaFile,resultFile,saveDataName,iouTh,confThs, scaleBs,scaleUs,labels)


#%%读取数据
#saveData640 = np.load(r'E:\detection\testImg\sqFix12_640x480.npy').item()
#PRmAPDict640 = CalPRmAP(saveData640, '640x480')

saveData = np.load(r'eval.npy', allow_pickle=True).item()
PRmAPDict = CalPRmAP(saveData, '300x300')

#%%作PR曲线
fig, axes = plt.subplots(nrows=1, figsize=(10, 10))

#label_colors = ['royalblue', 'red','seagreen']
#scale_lines = [ '-', '--', ':', '-.', '--', '--']
#DrawPR(PRmAPDict640,labels,'640x480',label_colors,scale_lines)

label_colors = ['blue','magenta', 'lime']
scale_lines = [ '-', '--', ':', '-.', '--', '--']
DrawPR(PRmAPDict,labels,'300x300',label_colors,scale_lines)

plt.legend(loc="lower left")
plt.yticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
plt.xticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
plt.plot([0.3, 1], [0.3, 1], '--', color=(0.6, 0.6, 0.6))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.grid()
fig.tight_layout()
plt.show()

#%%作mAP曲线
fig, axes = plt.subplots(nrows=1, figsize=(8, 12))

#colors = ['royalblue', 'red','seagreen']
#DrawScaleMAP(PRmAPDict640,['person','rider','car'],'640x480',colors,'o')

colors = ['blue','magenta', 'lime']
DrawScaleMAP(PRmAPDict,['person'],'300x300',colors,'*')

plt.yticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
plt.xlabel('Scale')
plt.ylabel('mAP')
plt.title('mAP-Scale')
plt.legend(loc="lower right")
plt.grid()
plt.show()