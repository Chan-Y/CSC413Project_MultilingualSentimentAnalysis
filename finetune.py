import argparse
from pathlib import Path
from datasets import load_metric, load_dataset, DatasetDict, concatenate_datasets
import random, os, json
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding, AutoModelForSequenceClassification
import torch
import numpy as np

local_dir = 'TOBECHANGED'


def label_to_number(example):
    ''' Convert 'positive' to 1, 'negative' to 0. '''
    example['sentiment'] = 1 if example['sentiment'] == 'positive' else 0
    return example

def add_EoS(text):
    ''' Add End-of-Sentence sign "\n" at the end of each sentence. '''
    reviews = nltk.sent_tokenize(text.strip())
    reviews_cleaned = [s for sent in reviews for s in sent.split("\n")]
    review_cleaned = "\n".join(reviews_cleaned)
    return review_cleaned


def tokenize_text(text, max_input_length, tokenizer):
    '''
    Tokenizes a given text using the provided tokenizer.

    Parameters:
    - text (dataset): The sample dataset in which the text is to be tokenized.
    - max_input_length (int): The max length of input text to be accepted. 
    - tokenizer (object): The tokenizer object.
    
    Returns:
    - tokenized_text.
    '''
    
    texts_cleaned = [add_EoS(text) for text in text["review"]]
    inputs = [text for text in texts_cleaned]

    print(f'Processing {len(inputs)} data ... ...')

    tokenized_text = tokenizer(inputs, 
                               truncation=True, 
                               max_length=max_input_length, 
                               padding="max_length")

    tokenized_text['labels'] = text['sentiment']
    return tokenized_text

def load_data(data_dir, count_per_lang):
    '''
    Loads text data and label from csv files in the specified directory.
    Parameters:
    - data_dir (str): The directory containing raw csv files.
    - count_per_lang (int): number of data per language

    Returns:
    - eng_data, fre_data (dataset): Two dataset containing review and corresponding sentiment.
        - 'review': List of text bodies extracted from the csv files.
        - 'sentiment': List of labels (converted to numerical ints) associated with each text entry.
    '''
    # load two datasets
    eng_data = load_dataset("csv", data_files=f"{local_dir}/{data_dir}/Eng_IMDB_Dataset.csv")
    fre_data = load_dataset("csv", data_files=f"{local_dir}/{data_dir}/Fre_train.csv")
    # print(f"After loading: ENG {eng_data} FRE {fre_data}\n\n")
    
    # Random sampling count_per_lang number data
    eng_ind = sorted(random.sample(range(0, len(eng_data['train'])), count_per_lang))
    fre_ind = sorted(random.sample(range(0, len(fre_data['train'])), count_per_lang))
    eng_data = eng_data['train'].select(eng_ind)
    fre_data = fre_data['train'].select(fre_ind)
    # print(f"After sampling: ENG {eng_data} FRE {fre_data}\n\n")
    
    # Apply the conversion to each example in the dataset
    eng_data = eng_data.map(label_to_number)

    # Rename dataset column and only keep useful columns
    fre_data = fre_data.rename_column('polarity', 'sentiment')
    fre_data = fre_data.select_columns(['review', 'sentiment'])

    # print(f"Before return load_data: ENG {eng_data} FRE {fre_data}\n\n")
    return eng_data, fre_data
    
def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    return model

def compute_metrics(eval_pred):
    '''    
    Compute performance metrics based on binary classification assumptions, calculating
    accuracy, precision, recall, and f1-score from the model's predictions.

    Parameter:
        - eval_pred (tuple): A tuple containing a numpy array of the model logits, 
        predictions, and a numpy array of the true labels, labels.

    Returns:
        - dict: A dictionary containing the rounded values of accuracy, precision, recall, f1.

    '''
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")

    predictions, labels = eval_pred
    logits = predictions[0]
    predictions = np.argmax(logits, axis=1)

    accuracy = np.mean(predictions == labels)
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels)["f1"]

    return {
        "accuracy": round(accuracy, 6), 
        "precision": round(precision, 6), 
        "recall": round(recall, 6), 
        "f1": round(f1, 6)
    }


def train_model(model_name, tokenizer, tokenized_train, tokenized_validation, 
                data_collator, training_args):
    """
    Trains the model.

    Parameters:
    - model (object): The model to be trained.
    - tokenizer (object): The tokenizer object.
    - tokenized_dataset (object): Tokenized dataset.
    - data_collator (object): Data collator with padding.
    - training_args (object): Training arguments for the Trainer.
    """

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f'************************************\n\tTraining ...\n************************************')
    trainer.train()
    return trainer.evaluate(), trainer


def main(args):
    '''Main function'''

    count_per_lang = args.max
    max_input_length = 350

    ### Load data and split into train, validation, test following 80%/10%/10%
    eng_data, fre_data = load_data(args.input_dir, count_per_lang)
    # Split off the test + validation dataset (20%)
    test_val_split_eng = eng_data.train_test_split(test_size=0.2)
    # Split the test + validation data equally into test and validation datasets
    final_splits_eng = test_val_split_eng['test'].train_test_split(test_size=0.5)
    # Gather splits into a single DatasetDict
    eng_data = DatasetDict({
        'train': test_val_split_eng['train'],  # 80% of the original dataset
        'validation': final_splits_eng['train'],  # 10% of the original dataset
        'test': final_splits_eng['test']  # 10% of the original dataset
    })
    # Implement same way on French dataset
    test_val_split_fre = fre_data.train_test_split(test_size=0.2)
    final_splits_fre = test_val_split_fre['test'].train_test_split(test_size=0.5)
    fre_data = DatasetDict({
        'train': test_val_split_fre['train'],
        'validation': final_splits_fre['train'],
        'test': final_splits_fre['test']
    })
    # print(eng_data, fre_data)

    ### Check if CUDA is available
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU instead.")


    global model_name
    model_name = args.model_name
    print(f'\nUSING PRETRAINED MODEL: {model_name}\n')

    # Tokenize both datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f'{local_dir}/models')
    tokenized_eng_dataset = eng_data.map(
        lambda review: tokenize_text(review, max_input_length, tokenizer), 
        batched=True
    )
    tokenized_fre_dataset = fre_data.map(
        lambda review: tokenize_text(review, max_input_length, tokenizer), 
        batched=True
    )
    # print(tokenized_eng_dataset, tokenized_eng_dataset['train'][0])
    
    ### Combine two language sets
    combined_train = concatenate_datasets(
        [tokenized_eng_dataset['train'], tokenized_fre_dataset['train']]).select_columns(['input_ids', 'labels'])
    combined_validation = concatenate_datasets(
        [tokenized_eng_dataset['validation'], tokenized_fre_dataset['validation']]).select_columns(['input_ids', 'labels'])
    

    data_collator = DataCollatorWithPadding(
        tokenizer, padding="max_length", max_length=max_input_length
    )
    
    batch_size = 8
    model_dir = f'{local_dir}/models/{model_name}_moviereview_analysis'
    training_args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=False,
        fp16=True,
        load_best_model_at_end=True,
        # report_to="tensorboard"
    )

    evaluation_result, trainer = train_model(
        model_name,
        tokenizer, 
        combined_train, 
        combined_validation, 
        data_collator, 
        training_args
    )

    save_to_file(args.output_dir, f'{model_name}_{args.max}_evaluation_result.out', evaluation_result)


    # Predict test datasets when script has --test-set 
    if args.test_set:
        print(f'************************************\n\tTesting ...\n************************************')
        eng_test_result = trainer.predict(test_dataset=tokenized_eng_dataset['test'].select_columns(['input_ids', 'labels']))
        fre_test_result = trainer.predict(test_dataset=tokenized_fre_dataset['test'].select_columns(['input_ids', 'labels']))
        save_to_file(
            args.output_dir, 
            f'{model_name}_{args.max}_test_result.out', 
            {'English':eng_test_result.metrics, 'French': fre_test_result.metrics}
        )


def save_to_file(output_dir, file_name, content):
    '''
    Save content into a txt file in ouput_dir directory. 

    Parameters:
    - output_dir (str): The directory where the file will be saved.
    - file_name (str): The name of the file.
    - content (dict): The text content to be saved in the file.
    '''
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    
    # Open the file in write mode and save the content
    with open(file_path, 'w') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)
    print('---------------------------------------')
    print(f"File saved successfully at {file_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-dir',
        help="Input directory of data",
        type=Path,
        default=Path('./data')
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory',
        type=Path,
        default=Path("./eval_output")
    )
    parser.add_argument(
        '-m', '--model-name',
        help='Model name',
        type=str,
        default='t5-small'
    )
    parser.add_argument(
        '--max',
        help='The maximum number of reviews to be processed in each language',
        type=int,
        default=5000
    )
    parser.add_argument(
        '--test-set', 
        action="store_true",
        help="A flag whether evaluate on the test dataset."
    )

    args = parser.parse_args()
    main(args)