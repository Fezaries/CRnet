import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from net import CooperationModule, RelationModule
from utils import Logger, evaluate
import sys
import time




sys.stdout = Logger('log/'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'.log')


with open('APY_train.py') as f:
    contents = f.read()
    print(contents)
f.close()


###############################################################################
K=3
BATCH_SIZE = 32  # Not recommended to change this
EPISODE = 250000
TEST_EPISODE = 1000
LEARNING_RATE = 1e-5
GPU = 0

###############################################################################
# step 1: init dataset
print("init dataset")

dataroot = './data'
dataset = 'APY'
image_embedding = 'res101'
class_embedding = 'att_splits'

matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
feature = matcontent['features'].T
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
# numpy array index starts from 0, matlab starts from 1
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

attribute = matcontent['original_att'].T * 100 # Scaled to [0,100] to keep consistent with AWA so that similar learning strategies can be adopted.

x = feature[trainval_loc] # train_features
train_label_ori = label[trainval_loc].astype(int)  # train_label
att = attribute[train_label_ori] # train attributes

att = torch.tensor(att).float().cuda()

x_test = feature[test_unseen_loc]  # test_feature
test_label = label[test_unseen_loc].astype(int) # test_label
x_test_seen = feature[test_seen_loc]  #test_seen_feature
test_label_seen = label[test_seen_loc].astype(int) # test_seen_label
test_id = np.unique(test_label)   # test_id
att_pro = attribute[test_id]      # test_attribute

# train set
train_features=torch.from_numpy(x)
print(train_features.shape)

train_label=torch.from_numpy(train_label_ori).unsqueeze(1)
print(train_label.shape)

# attributes
all_attributes=np.array(attribute)
print(all_attributes.shape)

attributes = torch.from_numpy(attribute)
# test set

test_features=torch.from_numpy(x_test)
print(test_features.shape)

test_label=torch.from_numpy(test_label).unsqueeze(1)
print(test_label.shape)

testclasses_id = np.array(test_id)
print(testclasses_id.shape)

test_attributes = torch.from_numpy(att_pro).float()
print(test_attributes.shape)

test_seen_features = torch.from_numpy(x_test_seen)
print(test_seen_features.shape)

test_seen_label = torch.from_numpy(test_label_seen)

train_data = TensorDataset(train_features,train_label)
###########################################################################

# initial algorithm
print("init networks")
att = np.unique(att,axis=0)
kmeans = KMeans(n_clusters=K, random_state=1337).fit(att) ### field number : K=3
att = kmeans.cluster_centers_
att = torch.tensor(att).float().cuda()


CM = CooperationModule(att)
RM = RelationModule()

CM.cuda(GPU)
RM.cuda(GPU)

CM_optim = torch.optim.Adam(CM.parameters(),lr=LEARNING_RATE*5, weight_decay=1e-5)
RM_optim = torch.optim.Adam(RM.parameters(),lr=LEARNING_RATE, weight_decay=1e-4)

print("training...")
last_H = 0.0

for episode in range(EPISODE):

    train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

    batch_features,batch_labels = train_loader.__iter__().next()

    sample_labels = []
    for label in batch_labels.numpy():
        if label not in sample_labels:
            sample_labels.append(label)

    sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
    class_num = sample_attributes.shape[0]

    batch_features = tensor(batch_features).cuda(GPU).float()  # 32*2048
    sample_features = CM(tensor(sample_attributes).cuda(GPU))  # x*312


    sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_SIZE,1,1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
    batch_features_ext = torch.transpose(batch_features_ext,0,1)

    relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
    relations = RM(relation_pairs).view(-1,class_num)

    # re-build batch_labels according to sample_labels
    sample_labels = np.array(sample_labels)
    re_batch_labels = []
    for label in batch_labels.numpy():
        index = np.argwhere(sample_labels==label)
        re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)

    # loss
    mse = nn.MSELoss().cuda(GPU)
    one_hot_labels = tensor(torch.zeros(BATCH_SIZE, class_num).scatter_(1, re_batch_labels.view(-1,1), 1)).cuda(GPU)
    loss = mse(relations,one_hot_labels)

    # update
    CM.zero_grad()
    RM.zero_grad()

    loss.backward()

    CM_optim.step()
    RM_optim.step()

    if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss %.4f" % loss.item())

    if (episode+1)%2000 == 0:
        # test
        print("Testing...")


        gzsl_unseen_accuracy = evaluate(test_features,test_label,np.arange(32),attributes, CM, RM, GPU)
        gzsl_seen_accuracy = evaluate(test_seen_features,test_seen_label,np.arange(32),attributes, CM, RM, GPU)

        H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

        print('seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))

        if H > last_H:


            # save networks
            torch.save(CM,"./models/APY_CM.pkl")
            torch.save(RM,"./models/APY_RM.pkl")

            print("save networks for episode:",episode)

            last_H = H



