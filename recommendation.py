import ast
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def getEmbedding(graph_list, finalList):
    graphFeature = []
    for i in range(len(finalList)):
      nodeFeature = [0]*(len(finalList))
      graphFeature.append(nodeFeature)
    for name in graph_list:
      if name in finalList:
        index = finalList.index(name)
        graphFeature[index][index] = 1
    graphTensor = torch.tensor(graphFeature)
    return graphTensor

def normalized_adj_matrix(graph1_name_list, final_list, graph1_adj_list):
    list_index = []
    for name in graph1_name_list:
      if name in final_list:
        index = final_list.index(name)
        list_index.append(index)
    list_index.sort()

    graph1_normalized_list=[]
    for i in range(len(final_list)):
      nodeFeature = [0]*(len(final_list))
      graph1_normalized_list.append(nodeFeature)
    for i in range(len(graph1_name_list)):
      x = list_index[i]
      for j in range(len(graph1_name_list)):
        y = list_index[j]
        if(graph1_adj_list[i][j]==1):
          graph1_normalized_list[x][y] = 1
    tensor = torch.Tensor(graph1_normalized_list)
    return tensor

def getCost(name1, name2, adj_matrix1, adj_matrix2):
      set1 = set(name1)
      set2 = set(name2)

      difference = list(set1.symmetric_difference(set2))
      cost = 10*(len(difference))

      for i in range(len(adj_matrix1)):
        for j in range(len(adj_matrix1)):
          if adj_matrix1[i][j] != adj_matrix2[i][j]:
            cost += 1

      return cost

def normalize(A):
    A = torch.eye(A.shape[0]) + A
    d = A.sum(1)
    D = torch.diag(torch.pow(d, -0.5))
    return torch.matmul(torch.matmul(D, A), D)

def expendDimension(X):
    tensor = torch.zeros(50,50)
    zero_list = tensor.numpy().tolist()
    X_list = X.numpy().tolist()
    index = len(X_list[0])
    for i in range(index):
      for j in range(index):
        zero_list[i][j]+= X_list[i][j]
    return torch.tensor(zero_list)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in, bias=False)

    def forward(self, X, A):
        X_copy = X.clone()
        X = self.fc1(X.mm(A))
        X = X_copy + X
        X = F.relu(X)
        X_copy = X.clone()
        X = self.fc2(X.mm(A))
        X = X_copy + X
        X = F.relu(X)
        return X

def euclidean_distance(x, y):
    return torch.sqrt( torch.sum(torch.pow(x-y, 2) ) )


def process_data(dataset_path, user_path, output_path):
    with open(dataset_path) as f:
        data = json.load(f)
    with open(user_path) as f:
        user_data = json.load(f)

    all_activities = data["all_activities"][0]

    activity_index = {act: idx for idx, act in enumerate(all_activities)}
    matrix_size = len(all_activities)

    def adjust_matrix(original_matrix, original_activities,matrix_size,activity_index):
        adjusted = np.zeros((matrix_size, matrix_size), dtype=int)
        for i, act_i in enumerate(original_activities):
            for j, act_j in enumerate(original_activities):
                new_i = activity_index[act_i]
                new_j = activity_index[act_j]
                adjusted[new_i, new_j] = original_matrix[i][j]
        return adjusted

    user_entry = next(iter(user_data.values()))
    cleaned_str = user_entry["adjacency_matrix"].strip('"')
    matrix = ast.literal_eval(cleaned_str)
    original_matrix = np.array(matrix)
    user_matrix = adjust_matrix(original_matrix, user_entry["activities"],matrix_size,activity_index)

    adjusted_data = []
    distances = []
    gcn = GCN(50, 50)
    for entry in data["dataset"]:
        file_path = entry["file"]
        cleaned_str = entry["adjacency_matrix"].strip('"')
        matrix = ast.literal_eval(cleaned_str)
        original_matrix = np.array(matrix)
        adj_matrix = adjust_matrix(original_matrix, entry["activities"],matrix_size,activity_index)

        nameList = all_activities

        user_embedding = getEmbedding(user_entry["activities"], nameList)
        database_embedding = getEmbedding(entry["activities"], nameList)
        user_embedding_amend = expendDimension(user_embedding)
        bpmn_embedding_amend = expendDimension(database_embedding)

        user_normalize = normalized_adj_matrix(user_entry["activities"], nameList, user_matrix)
        user_Laplace = normalize(user_normalize)
        user_Laplace_amend = expendDimension(user_Laplace)

        database_normalize = normalized_adj_matrix(entry["activities"], nameList, adj_matrix)
        database_Laplace = normalize(database_normalize)
        database_Laplace_amend = expendDimension(database_Laplace)

        f1 = gcn(user_embedding_amend, user_Laplace_amend)
        f2 = gcn(bpmn_embedding_amend, database_Laplace_amend)

        distance = euclidean_distance(f1, f2)

        dataset_entry = {
            "file": file_path,
            "adjacency_matrix": str(adj_matrix.tolist()),
        }
        adjusted_data.append(dataset_entry)

        distances.append((distance, file_path, adj_matrix))
        if "baseline" in file_path:
            print(file_path)
            print(adj_matrix)

    with open(output_path, "w") as f:
        json.dump(adjusted_data, f, indent=2)

    top = sorted(distances, key=lambda x: x[0])[:10]

    print("\nTop 10 Closest Matches:")
    print("=" * 60)
    for idx, (dist, path, matrix) in enumerate(top, 1):
        print(f"#{idx} Distance: {dist:.4f}")
        print(f"File: {path}")
        print("-" * 60)

    return top
