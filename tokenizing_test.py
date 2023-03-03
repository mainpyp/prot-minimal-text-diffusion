import sys
from transformers import AutoTokenizer
import pandas as pd

def main(data_path: str):
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("Reading Data to Pandas df")
    df = pd.read_csv(data_path, sep="\t", header=None)
    
    print(df.info())
    print("Converting data to list")
    text = df[0].apply(lambda x: x.strip()).tolist()
    
    print(sys.getsizeof(text)/8/1_000_000)
    print("encoding whole dataset")
    encoded = tokenizer(text_target=text, max_length=512, padding=True, truncation=True)
    print("encoding done")
    print(f"{sys.getsizeof(encoded)}")


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=512, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    data_path = "data/prot_total/prot_total.txt"
    main(data_path)
    # from datasets import load_dataset
    #
    # data_path = "data/prot_minimal/prot_minimal.txt"
    #
    # raw_datasets = load_dataset("csv", data_files=data_path)
    # print(raw_datasets)
    # processed_datasets = raw_datasets.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=args.preprocessing_num_workers,token
    #     remove_columns=column_names,
    #     load_from_cache_file=not args.overwrite_cache,
    #     desc="Running tokenizer on dataset",
    # )
