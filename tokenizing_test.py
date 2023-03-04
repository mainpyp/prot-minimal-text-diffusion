import datasets
import sys
from transformers import AutoTokenizer
import pandas as pd

def main(data_path: str):
    print("import matplotlib")
    import matplotlib.pyplot as plt
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("Reading Data to Pandas df")
    df = pd.read_csv(data_path, sep="\t", header=None)

    print(f"df head:\n {df.head()}\n")
    print(f"df tail:\n {df.tail()}\n")


    df[0].apply(lambda x: len(x)).hist()
    plt.savefig("test_plot.png")

    return
    
    print(df.info())
    print("Converting data to list")
    text = df[0].apply(lambda x: x.strip()).tolist()

    for i, t in enumerate(text):
        if i % 100_000 == 0:
            print(f"At {i}, {len(text) - i} to go.")
        tokenizer(t)
    
    print(sys.getsizeof(text)/8/1_000_000)
    print("encoding whole dataset")
    #encoded = tokenizer(text=text, max_length=512, padding=True, truncation=True)
    print("encoding done")
    #print(f"{sys.getsizeof(encoded)}")


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=512, padding=True, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=512, padding=True, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and False:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    data_path = "data/prot_total/prot_total.txt"
    #data_path = "data/prot_minimal/prot_minimal.txt"
    main(data_path)



    # paths = ["/mnt/home/mheinzinger/deepppi1tb/ProSST5/martins_set/data_mixed/train.csv",
    #          "/mnt/home/mheinzinger/deepppi1tb/ProSST5/martins_set/data_mixed/test.csv",
    #          "/mnt/home/mheinzinger/deepppi1tb/ProSST5/martins_set/data_mixed/val.csv"]
    #
    # print("saved paths")
    #
    #
    #
    #
    #
    # print("load datasets as raw")
    # raw_datasets = datasets.load_dataset("csv", data_files=paths)
    #
    # print("process datasets")
    # print(type(raw_datasets))
    # print(len(raw_datasets))
    # processed_datasets = raw_datasets.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=1,
    #     remove_columns=[],
    #     load_from_cache_file=True,
    #     desc="Running tokenizer on dataset",
    # )
    #
    # print(raw_datasets)



    




