# Generate a multiple-choice answer dataset about coding and programming knowledge using an LLM

import os
import sys
import re
import json
import random
from vllm import LLM, SamplingParams
from torch import bfloat16

output_file_jsonl = "../datasets/ComSciQA.jsonl"
output_file_hf = "../datasets/ComSciQA.hf"
num_questions = 30000 # number of MC questions to generate
model_name = "meta-llama/Llama-3.3-70B-Instruct"   
# model_name = "meta-llama/Llama-3.1-8B-Instruct"   
batch_size = 1000

# List of different computer science topics
with open('cs_topics.txt', 'r') as file:
    cs_topics = file.read().splitlines()
print('Computer Science topics found:', len(cs_topics))

# Check if MC questions already exist
if os.path.exists(output_file_jsonl):
    with open(output_file_jsonl, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print(f"Number of existing questions: {line_count}")
    generated_questions = line_count
else:
    generated_questions = 0

prompt_template = """You are a question generator for a multiple-choice test on Computer Science and Programming. 
Generate ten unique and nuanced questions on the topic: "{topic}". 
Questions should vary in difficulty and depth of knowledge, and every question should have four answer choices. 

**Format output as a Python list of dictionaries where every dictionary is like the example below:**
{{
    "question": "What is the time complexity of quicksort in the average case?",
    "choices": {{
        "label": ["A", "B", "C", "D"],
        "text": ["O(n)", "O(n log n)", "O(n^2)", "O(log n)"]
    }},
    "answerKey": "B"
}}
"""

def extract_json_from_response(response_text):
    try:
        response_text = response_text.strip()
        parsed_json = json.loads(response_text)
        return parsed_json
    except json.JSONDecodeError:
        # match = re.search(r"{.*?}", response_text, re.DOTALL)
        match = re.search(r'(\[\s*\{.*?\}\s*\])', response_text, re.DOTALL)
        if match:
            extracted_json = match.group(1)
            try:
                return json.loads(extracted_json)
            except:
                return None
        else:
            return None
    except Exception as e:
        return None
        
    
# Load the LLM
model = LLM(model_name, tensor_parallel_size=4, dtype=bfloat16)  # Choose any suitable model
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)

f_out = open(output_file_jsonl, 'a')
while generated_questions < num_questions:
    prompts = [prompt_template.format(topic=random.choice(cs_topics)) for _ in range(batch_size)]

    # Try to perform batch inference and parse outputs as JSON
    try:
        outputs = model.generate(prompts, sampling_params)

        # Extract JSON from responses
        for output in outputs:
            output = output.outputs[0].text.strip()
            question_data = extract_json_from_response(output)
            if question_data:
                generated_questions += 10
                entry = {"all_questions": question_data, "source": "ComSciQA"}
                f_out.write(json.dumps(entry) + "\n")
    except:
        continue
    
    print('Total Generated Questions:', generated_questions)
    
f_out.close()








