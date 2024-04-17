from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset, DatasetDict

def load_and_split_dataset(csv_file_path, test_size=0.2, val_size=0.1):
    # Load the dataset from the CSV file
    dataset = load_dataset('csv', data_files=csv_file_path)
    # Split the dataset
    train_val_split = dataset['train'].train_test_split(test_size=(test_size + val_size))
    val_test_split = train_val_split['test'].train_test_split(test_size=0.5)
    # Organize splits
    splits = DatasetDict({
        'train': train_val_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })
    return splits

def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return tokenizer, model

def evaluate_model(model, tokenizer, test_dataset, device, max_length):
    model.eval()
    test_reviews = test_dataset['review']  # Ensure this column exists
    inputs = tokenizer(test_reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits
    probabilities = torch.nn.functional.softmax(predictions, dim=-1)
    predicted_class_ids = probabilities.argmax(dim=-1)
    return probabilities, predicted_class_ids

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = './t5basesmallmoviereviewanalysis/checkpoint1000'
    csv_file_path = './Eng_IMDB_Dataset.csv'
    max_input_length = 256  # Adjust as per your tokenizer settings during training

    # Load and split the dataset
    data_splits = load_and_split_dataset(csv_file_path, test_size=0.2, val_size=0.1)
    test_data = data_splits['test']
    tokenizer, model = load_model(model_dir, device)

    # Evaluate the model on test data
    probabilities, predicted_class_ids = evaluate_model(model, tokenizer, test_data, device, max_input_length)

    for idx, text in enumerate(test_data['review']):
        print(f"Review: {text}")
        print(f"1 for positive, 0 for negative: {predicted_class_ids[idx]}")
        print(f"Probabilities: {probabilities[idx]}")

if __name__ == "__main__":
    main()
