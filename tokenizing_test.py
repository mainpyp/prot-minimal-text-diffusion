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


    encoded = [tokenizer(t) for t in text]


if __name__ == "__main__":
    data_path = "data/prot_total/prot_total.txt"
    main(data_path)
