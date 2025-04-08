import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def get_embedding(embed_model, question, device):
    """Get the embedding of a question."""
    embedding = embed_model.encode(question, convert_to_tensor=True, device=device)
    return embedding.cpu().numpy().tolist()

def prepare_input(tokenizer, prompts, device):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)
    return input_tokens

def apply_prompt_template(question, choices):
    choiceA = choices[0]
    choiceB = choices[1]
    choiceC = choices[2]
    choiceD = choices[3]

    # Five shot prompt
    prompt_template = """
The following are multiple choice questions (with answers) about a variety of topics. 

Question: Which organelle is responsible for producing energy in a cell?
A. nucleus
B. mitochondrian
C. ribosome
D. golgi apparatus
Answer: B

Question: If 3x-5=16, what is the value of x?
A. 2
B. 5
C. 7
D. 11
Answer: C

Question: What type of energy is stored in an object due to its position?
A. chemical energy
B. kinetic energy
C. thermal energy
D. potential energy
Answer: D

Question: Which sentence is written in passive voice?
A. The lesson was explained clearly by the teacher.
B. The teacher explained the lesson clearly.
C. The students asked many questions.
D. The teacher enjoys explaining difficult topics.
Answer: A

Question: The Renaissance was a period of cultural rebirth that began in which country?
A. France
B. England
C. Italy
D. Germany
Answer: C

Question: {question}
A. {choiceA}
B. {choiceB}
C. {choiceC}
D. {choiceD}
Answer: """

    return prompt_template.format(question=question, choiceA=choiceA, choiceB=choiceB, choiceC=choiceC, choiceD=choiceD)

def normalize_response(response):
    answer_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'a': 1, 'b': 2, 'c': 3, 'd': 4}

    if response in answer_map:    # Convert letters (upper/lower) to numbers
        return answer_map[response]
    elif response.isdigit() and int(response) in range(5):   # Keep valid numbers
        return int(response)
    else:
        return None
    

def main():
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    BATCH = 64
    model_name1 = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name2 = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    model_name3 = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME).to(DEVICE)

    print("Loading models...")
    model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(DEVICE)
    model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(DEVICE)
    model3 = AutoModelForCausalLM.from_pretrained(model_name3).to(DEVICE)

    print("Loading tokenizers...")
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1, padding_side='left', device=DEVICE)
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2, padding_side='left', device=DEVICE)
    tokenizer3 = AutoTokenizer.from_pretrained(model_name3, padding_side='left', device=DEVICE)

    ### Training dataset ###
    train = load_dataset("arrow", data_files="/ascldap/users/swtsai/BERT-CoE/BERT-CoE-v2/datasets/CombineQA.hf/data-00000-of-00001.arrow")["train"]
    train_data = []

    for example in tqdm(train, desc="Processing training questions: "):
        question = example["question"]
        subject = example["subject"]
        subject_mapping = {"ARC-Challenge": "ARC",
                            "ARC-Easy": "ARC",
                            "OpenBook-Additional": "OpenBook",
                            "OpenBook-Main": "OpenBook",
                          }
        source = subject_mapping.get(example["subject"], example["subject"])
        choices = example["choices"]
        if len(choices) < 4:
            continue
        answer = example["answer"]+1   # Training data answers are 0-3 but LLM outputs are usually 1-4

        # Get BERT embedding
        embedding = get_embedding(embed_model, question, DEVICE)

        train_data.append({"question": question, "subject": subject, "source": source, "embedding": embedding, "choices": choices, "answer": answer})

    # Convert to DataFrame
    train_df = pd.DataFrame(train_data)
    train_df.to_csv("train_metadata.csv",index=False)

    train_df = pd.read_csv("train_metadata.csv",index_col=False)

    # Grab all questions and perform batch inference
    train_questions = train_df["question"]
    train_choices = train_df["choices"]
    train_qwen_answers = []
    train_mathqwen_answers = []
    train_codeqwen_answers = []
    for i in tqdm(range(0, len(train_questions), BATCH), desc="Batch inference: "):
        batch_questions = train_questions[i:i+BATCH]
        batch_choices = train_choices[i:i+BATCH]
        batch_prompts = [apply_prompt_template(q, c) for q, c in zip(batch_questions, batch_choices)]

        # Batch tokenize
        inputs1 = prepare_input(tokenizer1, batch_prompts, DEVICE)
        inputs2 = prepare_input(tokenizer2, batch_prompts, DEVICE)
        inputs3 = prepare_input(tokenizer3, batch_prompts, DEVICE)

        # Batch inference
        outputs1 = model1.generate(**inputs1, max_new_tokens=1, pad_token_id=tokenizer1.pad_token_id)
        outputs2 = model2.generate(**inputs2, max_new_tokens=1, pad_token_id=tokenizer2.pad_token_id)
        outputs3 = model3.generate(**inputs3, max_new_tokens=1, pad_token_id=tokenizer3.pad_token_id)

        # Batch decode
        answers1 = [tokenizer1.decode(seq[-1], skip_special_tokens=True).replace(" ", "") for seq in outputs1]
        answers2 = [tokenizer2.decode(seq[-1], skip_special_tokens=True).replace(" ", "") for seq in outputs2]
        answers3 = [tokenizer3.decode(seq[-1], skip_special_tokens=True).replace(" ", "") for seq in outputs3]
        
        train_qwen_answers.extend(answers1)
        train_mathqwen_answers.extend(answers2)
        train_codeqwen_answers.extend(answers3)
    
    train_df["Qwen"] = train_qwen_answers
    train_df["MathQwen"] = train_mathqwen_answers
    train_df["CodeQwen"] = train_codeqwen_answers

    # LLM outputs are usually strings A-D or 1-4, but MMLU has integer answers 0-3
    for col in ['Qwen', 'MathQwen', 'CodeQwen']:
        train_df[f'{col}_normalized'] = train_df[col].apply(normalize_response)
        train_df[f'{col}_correct'] = (train_df[f'{col}_normalized'] == (train_df['answer'])).astype(int)
    
    train_df['label'] = train_df[['Qwen_correct', 'MathQwen_correct', 'CodeQwen_correct']].values.tolist()
    train_df.to_csv("train_metadata.csv",index=False)
        
if __name__ == "__main__":
    main()