import torch
from torch.utils.data import TensorDataset
import numpy as np
import scipy.io as sio
from utils import evaluate





###############################################################################

GPU = 0

###############################################################################
# step 1: init dataset
print("init dataset")

dataroot = './data'
dataset = 'AwA2'
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

attribute = matcontent['original_att'].T

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



CM = torch.load('./models/AwA2_CM.pkl')
RM = torch.load('./models/AwA2_RM.pkl')

CM.cuda(GPU)
RM.cuda(GPU)

CM.eval()
RM.eval()

print("Testing...")


gzsl_unseen_accuracy = evaluate(test_features,test_label,np.arange(200),attributes, CM, RM, GPU)
gzsl_seen_accuracy = evaluate(test_seen_features,test_seen_label,np.arange(200),attributes, CM, RM, GPU)

H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

print('seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))




