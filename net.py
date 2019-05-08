import torch.nn as nn
import torch.nn.functional as F
import torch

class RelationModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationModule, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class ExpertModule(nn.Module):

    def __init__(self, field_center):
        super(ExpertModule, self).__init__()
        self.field_center = field_center
        input_size = field_center.shape[0]
        self.fc = nn.Linear(input_size, 2048)

    def forward(self, semantic_vec):
        input_offsets = semantic_vec-self.field_center
        response = F.relu(self.fc(input_offsets))

        return response

class CooperationModule(nn.Module):

    def __init__(self, field_centers):
        super(CooperationModule, self).__init__()
        self.individuals = nn.ModuleList([])
        for i in range(field_centers.shape[0]):
            self.individuals.append(ExpertModule(field_centers[i]))


    def forward(self, semantic_vec):
        responses = [indiv(semantic_vec) for indiv in self.individuals]
        feature_anchor = sum(responses)


        return feature_anchor