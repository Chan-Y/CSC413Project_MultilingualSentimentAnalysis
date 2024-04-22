import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, concatenate_datasets
import random, os, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

local_dir = '/h/u7/c0/01/yuxiaoq4/24CSC413/project'

def label_to_number(example):
    ''' Convert 'positive' to 1, 'negative' to 0. '''
    example['sentiment'] = 1 if example['sentiment'] == 'positive' else 0
    return example


def load_data(data_dir, num_samples):
    eng_data = load_dataset("csv", data_files=f"{local_dir}/{data_dir}/Eng_IMDB_Dataset.csv")
    fre_data = load_dataset("csv", data_files=f"{local_dir}/{data_dir}/Fre_test.csv")

    # Random sampling num_samples number data
    eng_ind = sorted(random.sample(range(0, len(eng_data['train'])), num_samples))
    fre_ind = sorted(random.sample(range(0, len(fre_data['train'])), num_samples))
    eng_data = eng_data['train'].select(eng_ind)
    fre_data = fre_data['train'].select(fre_ind)

    # Apply the conversion to each example in the dataset
    eng_data = eng_data.map(label_to_number)

    # Rename dataset column and only keep useful columns
    fre_data = fre_data.rename_column('polarity', 'sentiment')
    fre_data = fre_data.select_columns(['review', 'sentiment']) 
    # print(eng_data, fre_data, eng_data[0])

    input_data = DatasetDict({
        'data': concatenate_datasets([eng_data, fre_data])
    })

    # print(input_data['data'][0])
    return input_data

def save_to_file(output_dir, file_name, content):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)
    print(f"File saved successfully at {file_path}")


def main(args):

    max_input_length = 350
    num_samples = args.sample_len
    model_dir = args.model_dir

    model_name = str(model_dir).rsplit("/")[1]
    print(f'Model Name: {model_name}')

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir) 

    input_data = load_data(args.input_dir, num_samples)

    results = []
    for data in input_data['data']:
        input_text = tokenizer([data['review']], max_length=max_input_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**input_text)
            predictions = outputs.logits
        prob = torch.nn.functional.softmax(predictions, dim=-1)
        predicted_class = prob.argmax(dim=-1).item()
        results.append({'review': data['review'], 'actual': data['sentiment'], 'predict': predicted_class})

    save_to_file(args.output_dir, f'{model_name}_{num_samples}_inf.txt', results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-dir',
        help="Input directory of data",
        type=Path,
        default=Path('./data')
    )
    parser.add_argument(
        '--model-dir',
        help="Input directory of model",
        type=Path,
        default=Path('./models/t5-small_moviereview_analysis/checkpoint-2000')
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory',
        type=Path,
        default=Path("./output")
    )
    parser.add_argument(
        '-l', '--sample-len',
        help='Number of samples',
        type=int,
        default=20
    )
    args = parser.parse_args()
    main(args)