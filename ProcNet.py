import json
import os
import re
import numpy as np
import networkx as nx
import copy
from xml.dom.minidom import parse, Document
import torch
import torch.nn as nn
import torch.nn.functional as F
# from numba.core.event import start_event

from LLMs import LLMs_graph
from similarity import Alignment
from recommendation import process_data


def get_text(node):
    rc = []
    for node in node.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return "".join(rc).strip()


def bpmn_to_graph(root_dom: Document, prefix: str = "bpmn:"):
    start_event = root_dom.getElementsByTagName(f"{prefix}startEvent")
    end_event = root_dom.getElementsByTagName(f"{prefix}endEvent")
    task_nodes = root_dom.getElementsByTagName(f"{prefix}task")
    seq_flows = root_dom.getElementsByTagName(f"{prefix}sequenceFlow")
    par_gateways = root_dom.getElementsByTagName(f"{prefix}parallelGateway")
    exc_gateways = root_dom.getElementsByTagName(f"{prefix}exclusiveGateway")
    inc_gateways = root_dom.getElementsByTagName(f"{prefix}inclusiveGateway")

    for s in start_event:
        if not s.hasAttribute("name") or not s.getAttribute("name"):
            s.setAttribute("name", "start event")

    for e in end_event:
        if not e.hasAttribute("name") or not e.getAttribute("name"):
            e.setAttribute("name", "end event")


    all_tasks = task_nodes+start_event+end_event

    name_to_nodes = {}
    for task in all_tasks:
        task_name = task.getAttribute("name").replace("\n", " ")
        if task_name not in name_to_nodes:
            name_to_nodes[task_name] = []
        name_to_nodes[task_name].append(task)

    sorted_task_nodes = []
    sorted_names = sorted(name_to_nodes.keys(), key=str.lower)
    for name in sorted_names:
        sorted_task_nodes.extend(name_to_nodes[name])


    node_list =  sorted_task_nodes + par_gateways + exc_gateways +inc_gateways

    seq_flow_ids = []
    par_gateway_ids = []
    exc_gateway_ids = []
    inc_gateway_ids = []

    G = nx.DiGraph()
    for node in node_list + seq_flows:
        node_type = node.tagName
        node_id = node.getAttribute("id")
        node_name = node.getAttribute("name")

        if node_type == f"{prefix}sequenceFlow":
            seq_flow_ids.append(node_id)
        elif node_type == f"{prefix}parallelGateway":
            par_gateway_ids.append(node_id)
        elif node_type == f"{prefix}exclusiveGateway":
            exc_gateway_ids.append(node_id)
        elif node_type == f"{prefix}inclusiveGateway":
            inc_gateway_ids.append(node_id)


        if node_type in [f"{prefix}task", f"{prefix}startEvent", f"{prefix}endEvent"]:
            node_name = node.getAttribute("name").replace("\n", " ")
            if node_name in name_to_nodes:
                merged_node_id = name_to_nodes[node_name][0].getAttribute("id")
                G.add_node(merged_node_id, type=node_type, name=node_name)
            else:
                G.add_node(node_id, type=node_type, name=node_name)
        else:
            G.add_node(node_id, type=node_type, name=node_name)

    for node in node_list:
        node_id = node.getAttribute("id")
        node_name = node.getAttribute("name").replace("\n", " ")

        if node_name in name_to_nodes:
            node_id = name_to_nodes[node_name][0].getAttribute("id")

        for child in node.childNodes:

            if child.nodeType == child.TEXT_NODE:
                continue
            if child.tagName not in [f"{prefix}incoming", f"{prefix}outgoing",
                "extensionElements", f"{prefix}dataOutputAssociation",
                f"{prefix}standardLoopCharacteristics", f"{prefix}multiInstanceLoopCharacteristics",
                 f"{prefix}property", f"{prefix}dataInputAssociation"]:
                print("Invalid tag name:", child.tagName)
            child_id = get_text(child)

            if child.tagName == f"{prefix}incoming":
                G.add_edge(child_id, node_id)
            elif child.tagName == f"{prefix}outgoing":
                G.add_edge(node_id, child_id)


    for node_id in seq_flow_ids + exc_gateway_ids:
        for predecessor in G.predecessors(node_id):
            for successor in G.successors(node_id):
                G.add_edge(predecessor, successor)

        G.remove_node(node_id)

    for node_id in par_gateway_ids + inc_gateway_ids:
        for predecessor in G.predecessors(node_id):
            for successor in G.successors(node_id):
                G.add_edge(predecessor, successor, is_par=True)

        G.remove_node(node_id)
    H = copy.deepcopy(G)

    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G, orientation="original")
        G.remove_edge(*cycle[-1][:2])

    for (u, v), lca in nx.all_pairs_lowest_common_ancestor(G):
        if u == lca or v == lca:
            continue

        ok_u = False
        for path in nx.all_simple_paths(G, lca, u):
            if len(path) > 0:
                attrs = G.get_edge_data(path[0], path[1])
                if "is_par" in attrs and attrs["is_par"]:
                    ok_u = True

        ok_v = False
        for path in nx.all_simple_paths(G, lca, v):
            if len(path) > 0:
                attrs = G.get_edge_data(path[0], path[1])
                if "is_par" in attrs and attrs["is_par"]:
                    ok_v = True

        if ok_u and ok_v:
            H.add_edge(u, v)
            H.add_edge(v, u)

    return H


def find_concurrent_relations(relations,sorted_names):
    activity_to_place = []
    place_to_activity = []

    for rel in relations:
        if rel[0] not in sorted_names:
            place_to_activity.append(rel)
        elif rel[0]in sorted_names:
            activity_to_place.append(rel)

    concurrent_starts = {}
    for rel in activity_to_place:
        activity = rel[0]
        if activity not in concurrent_starts:
            concurrent_starts[activity] = []
        concurrent_starts[activity].append(rel[1])

    concurrent_starts = {k: v for k, v in concurrent_starts.items() if len(v) > 1}

    concurrent_ends = {}
    for rel in place_to_activity:
        activity = rel[1]
        if activity not in concurrent_ends:
            concurrent_ends[activity] = []
        concurrent_ends[activity].append(rel[0])

    concurrent_ends = {k: v for k, v in concurrent_ends.items() if len(v) > 1}

    seen_relations = set()
    concurrent_relations = []

    for start_activity, start_places in concurrent_starts.items():
        relation_key = (start_activity,)
        if relation_key not in seen_relations:
            seen_relations.add(relation_key)

            branches = []
            for place in start_places:
                branch_activities = []
                current_place = place

                while True:
                    found_end = False
                    for rel in place_to_activity:
                        if rel[0] == current_place and rel[1] in concurrent_ends:
                            found_end = True
                            break

                    if found_end:
                        break

                    next_activity = None
                    for rel in place_to_activity:
                        if rel[0] == current_place:
                            next_activity = rel[1]
                            break

                    if not next_activity:
                        break

                    branch_activities.append(next_activity)

                    next_place = None
                    for rel in activity_to_place:
                        if rel[0] == next_activity:
                            next_place = rel[1]
                            break

                    if not next_place:
                        break

                    current_place = next_place

                if branch_activities:
                    branches.append(branch_activities)

            if branches:
                end_activity = None
                last_activities = [branch[-1] for branch in branches if branch]

                for activity in last_activities:
                    for rel in activity_to_place:
                        if rel[0] == activity:
                            for p_rel in place_to_activity:
                                if p_rel[0] == rel[1] and p_rel[1] in concurrent_ends:
                                    end_activity = p_rel[1]
                                    break
                            if end_activity:
                                break
                    if end_activity:
                        break

                if end_activity:
                    concurrent_relations.append({
                        'start': start_activity,
                        'end': end_activity,
                        'branches': branches
                    })

    return concurrent_relations

def petri_to_graph(root_dom: Document):

    task_nodes = root_dom.getElementsByTagName("transition")
    seq_flows = root_dom.getElementsByTagName("arc")

    all_tasks = task_nodes

    name_to_nodes = {}
    for task in all_tasks:
        task_name = task.getAttribute("id").replace("\n", " ")
        if task_name not in name_to_nodes:
            name_to_nodes[task_name] = []
        name_to_nodes[task_name].append(task)

    sorted_task_nodes = []
    sorted_names = sorted(name_to_nodes.keys(), key=str.lower)
    for name in sorted_names:
        sorted_task_nodes.extend(name_to_nodes[name])

    seq_flow_list = []
    edge_list=[]
    H = nx.DiGraph()
    for seq in seq_flows:
        source = seq.getAttribute("source")
        target= seq.getAttribute("target")
        seq_flow_list.append([source,target])
        H.add_edge(source,target)

    G = nx.DiGraph()
    for node in sorted_names:
        G.add_node(node)
    for arc_i in seq_flow_list:
        if arc_i[0] in sorted_names:
            for arc_j in seq_flow_list:
                if arc_i[1]==arc_j[0]:
                    G.add_edge(arc_i[0],arc_j[1])
                    edge_list.append([arc_i[0],arc_j[1]])
                else:
                    continue
        else:
            continue

    concurrent_relations = find_concurrent_relations(seq_flow_list,sorted_names)
    for rel in concurrent_relations:
        for i in range(len(rel['branches'])):
            for j in range(i + 1, len(rel['branches'])):
                for act1 in rel['branches'][i]:
                    for act2 in rel['branches'][j]:
                        if [act1, act2] not in edge_list:
                            edge_list.append([act1, act2])
                            G.add_edge(act1, act2)
                        if [act2, act1] not in edge_list:
                            edge_list.append([act2, act1])
                            G.add_edge(act2, act1)

    flag=False
    for node in sorted_names:
        if node=="T0":
            flag=True
        else:continue

    if flag:
        for arc_i in edge_list:
            if arc_i[0]=="T0":
                for arc_j in edge_list:
                    if arc_j[1]=="T0":
                        G.add_edge(arc_j[0],arc_i[1])
                        edge_list.append([arc_j[0],arc_i[1]])
                    else:continue
            else:continue
        for arc in edge_list:
            if arc[0] =="T0" or arc[1]=="T0":
                G.remove_edge(arc[0],arc[1])
        G.remove_node("T0")
    return G

def sortedNameList_Petri(root_dom: Document):
    task = root_dom.getElementsByTagName("transition")
    sorted_name_list = []
    for element in task:
        name = element.getAttribute('id').replace("\n", " ")
        sorted_name_list.append(name)
    sorted_name_list.remove("T0")
    sorted_name = sorted(sorted_name_list, key= str.lower)
    return sorted_name

def sortedNameList_BPMN(root_dom: Document, prefix: str = "bpmn:"):

    start_event = root_dom.getElementsByTagName(f"{prefix}startEvent")
    end_event = root_dom.getElementsByTagName(f"{prefix}endEvent")
    task = root_dom.getElementsByTagName(f"{prefix}task")
    task_nodes = start_event+end_event+task
    sorted_name_list = []
    for element in task_nodes:
        name = element.getAttribute('name').replace("\n", " ")
        sorted_name_list.append(name)
    sorted_name = sorted(set(sorted_name_list), key= str.lower)
    return sorted_name

def twoGraphList(graph1_list,graph2_list):
    finalList = graph1_list+graph2_list
    finalList = list(set(finalList))
    finalList = sorted(finalList, key= str.lower)
    return finalList

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


def database_process_files(root_dir="Dataset", output_file="results.json"):

    all_activities = set()
    results = {
        "dataset": [],
        "all_activities": []
    }

    bpmn_root = os.path.join(root_dir, "Dataset_BPMN")
    for dirpath, _, filenames in os.walk(bpmn_root):
        for filename in filenames:
            if filename.endswith(".bpmn"):
                file_path = os.path.join(dirpath, filename)
                try:
                    root = parse(file_path)
                    G = bpmn_to_graph(root,"bpmn:")
                    adj_matrix = np.asarray(nx.adjacency_matrix(G).todense())
                    activities = sortedNameList_BPMN(root,"bpmn:")

                    dataset_entry = {
                        "file": file_path,
                        "adjacency_matrix": str(np.asarray(adj_matrix).tolist()),
                        "activities": activities
                    }

                    results["dataset"].append(dataset_entry)
                    for activity in activities:
                        all_activities.add(activity)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")


    petri_root = os.path.join(root_dir, "Dataset_Petri")
    for dirpath, _, filenames in os.walk(petri_root):
        for filename in filenames:
            if filename.endswith(".xml"):
                file_path = os.path.join(dirpath, filename)
                try:

                    root = parse(file_path)
                    G = petri_to_graph(root)
                    adj_matrix = np.asarray(nx.adjacency_matrix(G).todense())
                    activities = sortedNameList_Petri(root)

                    dataset_entry = {
                        "file": file_path,
                        "adjacency_matrix": str(np.asarray(adj_matrix).tolist()),
                        "activities": activities
                    }
                    results["dataset"].append(dataset_entry)
                    for activity in activities:
                        all_activities.add(activity)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    results["all_activities"].append(sorted(list(all_activities)))
    json_str = json.dumps(results, indent=2)

    with open(output_file, "w") as f:
        f.write(json_str)
    print(f"Saved to {output_file}")

def user_process_files(user_process_activities_align, user_process_adj_align):
    user_results = {}
    output_file = "user_input.json"
    user_results["user"] = {
        "adjacency_matrix": str(np.asarray(user_process_adj_align).tolist()),
        "activities": user_process_activities_align
    }
    json_str = json.dumps(user_results, indent=2)
    with open(output_file, "w") as f:
        f.write(json_str)
    print(f"Saved to {output_file}")


def main():
    database_process_files()

    with open("results.json") as f:
        dataset = json.load(f)
    database_process_activities = dataset["all_activities"][0]
    process_description_text = "First, the system will receive the loan application and then check whether the application form is complete. If the form is incomplete, the application is returned to the applicant. Once the updated application is received, the process restarts. If the form is complete, the process proceeds with two parallel checks: one involves reviewing the credit history and assessing the loan risk, while the other involves conducting a property appraisal. Once both tasks are completed, the system evaluates the applicant's eligibility for the loan. If the applicant does not meet the loan criteria, the application is rejected. If the applicant is eligible, an acceptance pack is prepared, and the system checks whether a home insurance quote is required. If an insurance quote is needed, a home insurance quote is sent; otherwise, the acceptance pack is sent directly. Subsequently, the system verifies the repayment agreement. If the applicant disagrees with the terms, the application is canceled. If the applicant agrees, the application is approved."


    user_process_activities, user_process_adj = LLMs_graph(process_description_text)

    user_process_activities_align, user_process_adj_align = Alignment(user_process_activities, database_process_activities ,user_process_adj)
    user_process_files(user_process_activities_align, user_process_adj_align)

    process_data(dataset_path="results.json", user_path="user_input.json", output_path="adjusted.json")


if __name__ == '__main__':
    main()