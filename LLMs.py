import numpy as np
from openai import OpenAI
import json
from branch_structure_checking import parse_input,find_precise_branches
from relationship_transformation import process_workflow

def get_response(messages):
    client = OpenAI(
        api_key="api_key",
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )
    return response

def read_prompts_from_file(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = f.read().split('====')
            return {
                'first_prompt': prompts[0].strip(),
                'second_prompt': prompts[1].strip()
            }
    else:
        raise ValueError("Unsupported file format")

def process_prompt_with_text(prompt_template, placeholder, replacement_text):

    if isinstance(prompt_template, list):
        template_str = '\n'.join(prompt_template)
    else:
        template_str = str(prompt_template)

    return template_str.replace(placeholder, replacement_text)


def LLMs_graph(process_description_text):
    file_path = 'prompts.json'
    prompts = read_prompts_from_file(file_path)
    messages = [{'role': 'system','content': 'You are a process analysis expert, and your main task is to analyze process description texts and extract the activities and their sequential relationships.'}]


    first_prompt = process_prompt_with_text(
        prompts['first_prompt'],
        '{process description text}',
        process_description_text
    )
    messages.append({'role': 'user', 'content': first_prompt})
    assistant_output1 = get_response(messages).choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_output1})

    first_2_prompt = process_prompt_with_text(
        prompts['first_2_prompt'],
        '{all activities}',
        assistant_output1
    )
    messages.append({'role': 'user', 'content': first_2_prompt})
    assistant_output1_2 = get_response(messages).choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_output1_2})


    second_prompt = process_prompt_with_text(
        prompts['second_prompt'],
        '{process description text}',
        process_description_text
    )
    second_prompt = process_prompt_with_text(
        second_prompt,
        '{all activities}',
        assistant_output1_2
    )
    messages.append({'role': 'user', 'content': second_prompt})
    assistant_output2 = get_response(messages).choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_output2})


    second_2_prompt = process_prompt_with_text(
        prompts['second_2_prompt'],
        '{all activities}',
        assistant_output1_2
    )
    second_2_prompt = process_prompt_with_text(
        second_2_prompt,
        '{order relationships}',
        assistant_output2
    )
    messages.append({'role': 'user', 'content': second_2_prompt})
    assistant_output2_2 = get_response(messages).choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_output2_2})



    raw_edges = [line.strip() for line in assistant_output2_2.split('\n') if line.strip()]
    parsed_edges = parse_input(raw_edges)

    detected_branches = find_precise_branches(parsed_edges)


    output_str = ""
    for i, branch in enumerate(detected_branches, 1):
        branch_str = ""
        branch_str += f"fragment{i}:\n"
        for edge in branch:
            branch_str += f"[{edge[0]}, {edge[1]}]\n"
        output_str += branch_str

    third_prompt = process_prompt_with_text(
        prompts['third_prompt'],
        '{process description text}',
        process_description_text
    )

    third_prompt = process_prompt_with_text(
        third_prompt,
        '{all activities}',
        assistant_output1_2
    )

    third_prompt = process_prompt_with_text(
        third_prompt,
        '{order relationships}',
        assistant_output2_2
    )

    third_prompt = process_prompt_with_text(
        third_prompt,
        '{order relationship fragments}',
        output_str
    )

    messages.append({'role': 'user', 'content': third_prompt})
    assistant_output3 = get_response(messages).choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_output3})

    output = process_workflow(assistant_output3)

    final_output = '\n'.join([f'[{", ".join(pair)}]' for pair in output])
    print("add edges: ",final_output)


    activities, relationships = preprocess_activities_and_relationships(assistant_output1_2, assistant_output2_2+"\n"+final_output)
    sorted_activities, adjacency_matrix = create_adjacency_matrix(activities, relationships)

    print("Sorted Activities:")
    for i, activity in enumerate(sorted_activities):
        print(f"{i}: {activity}")

    print("\nAdjacency Matrix:")
    print("    " + " ".join(f"{i:2}" for i in range(len(sorted_activities))))
    for i, row in enumerate(adjacency_matrix):
        print(f"{i:2}: " + " ".join(f"{val:2}" for val in row))

    return  sorted_activities,np.array(adjacency_matrix)



def preprocess_activities_and_relationships(activities_str, relationships_str):
    activities = [line.strip() for line in activities_str.split('\n') if line.strip()]

    relationships = []
    for line in relationships_str.split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            parts = line[1:-1].split(',')
            if len(parts) == 2:
                relationships.append([parts[0].strip(), parts[1].strip()])

    return activities, relationships


def create_adjacency_matrix(activities, relationships):
    sorted_activities = sorted(activities)
    activity_index = {activity: idx for idx, activity in enumerate(sorted_activities)}

    n = len(sorted_activities)
    adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for src, dest in relationships:
        src_idx = activity_index[src]
        dest_idx = activity_index[dest]
        adjacency_matrix[src_idx][dest_idx] = 1

    return sorted_activities, adjacency_matrix


