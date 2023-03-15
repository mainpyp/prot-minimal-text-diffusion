import datasets
import sys
from transformers import AutoTokenizer
import pandas as pd


def main_old(data_path: str):
    print("import matplotlib")
    import matplotlib.pyplot as plt
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("Reading Data to Pandas df")
    df = pd.read_csv(data_path, sep="\t", header=None)

    print(f"df head:\n {df.head()}\n")
    print(f"df tail:\n {df.tail()}\n")


    # df[0].apply(lambda x: len(x)).hist()
    # plt.savefig("test_plot.png")
    
    print(df.info())
    print("Converting data to list")
    text = df[0].apply(lambda x: x.strip()).tolist()

    for i, t in enumerate(text):

        for x in range(10):
            print(t[x], flush=True, end="|")
        break

        print(type(t))
        if i % 100_000 == 0:
            print(f"At {i}, {len(text) - i} to go.")
        tokenizer(t)
    
    print(sys.getsizeof(text)/8/1_000_000)
    print("encoding whole dataset")
    #encoded = tokenizer(text=text, max_length=512, padding=True, truncation=True)
    print("encoding done")
    #print(f"{sys.getsizeof(encoded)}")


def main(data_paths: dict):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    from datasets import load_dataset

    print("Loading datasets")
    dataset = load_dataset("text", data_files={"train": data_paths["train"],
                                               "test": data_paths["test"]})

    dataset = load_dataset("text", data_files=data_paths["test"])
    print(f"Dataset: {dataset['train']['text']}")

    print("Creating Helper Function")
    def tokenization(example):
        return tokenizer(example["text"], max_length=512, padding=True, truncation=True)

    print("Start Batched Tokenization")
    # https://huggingface.co/docs/datasets/v2.10.0/en/package_reference/main_classes#datasets.Dataset.map
    d = dataset.map(tokenization, batched=True,
                    num_proc=1,
                    remove_columns=[],
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset")

    print(f"Dataset: \n{len(d['train']['input_ids'])}")


if __name__ == "__main__":
    data_path = "data/prot_total/prot_total.txt"
    #data_path = "data/prot_minimal/prot_minimal.txt"

    paths = {"test": "../prot-minimal-text-diffusion/data/prot_minimal-test.txt",
             "train": "../prot-minimal-text-diffusion/data/prot_minimal-train.txt"}

    # paths = {"test": "../prot-minimal-text-diffusion/data/prot_total-test.txt",
    #          "train": "../prot-minimal-text-diffusion/data/prot_total-train.txt"}

    main(paths)

    # paths = ["/mnt/home/mheinzinger/deepppi1tb/ProSST5/martins_set/data_mixed/train.csv",
    #          "/mnt/home/mheinzinger/deepppi1tb/ProSST5/martins_set/data_mixed/test.csv",
    #          "/mnt/home/mheinzinger/deepppi1tb/ProSST5/martins_set/data_mixed/val.csv"]




    




