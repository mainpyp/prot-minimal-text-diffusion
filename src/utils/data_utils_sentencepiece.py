import logging
import torch
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial

logging.basicConfig(level=logging.INFO)

# BAD: this should not be global
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def get_dataloader(tokenizer, data_path, batch_size, max_seq_len):
    dataset = TextDataset(tokenizer=tokenizer, data_path=data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle=True,
        num_workers=1,
        collate_fn=partial(TextDataset.collate_pad, cutoff=max_seq_len),
    )

    while True:
        for batch in dataloader:
            yield batch


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_path: str,
            has_labels: bool = False
            ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.input_ids = None
        self.read_data()
        if has_labels:
            self.read_labels()

    def read_data(self):
        logging.info("Reading data from {}".format(self.data_path))

        dataset = load_dataset("text", data_files=self.data_path)

        logging.info("Creating helper function")

        def tokenization(example):
            return self.tokenizer(example["text"],
                                  max_length=512,
                                  padding=True,
                                  truncation=True)

        if hasattr(self.tokenizer, 'encode_batch'):  # relict of old days
            print("encode_batch")
            self.text = dataset["text"].apply(lambda x: x.strip()).tolist()
            encoded_input = self.tokenizer.encode_batch(self.text)
            self.input_ids = [x.ids for x in encoded_input]

        else:
            logging.info("Start Batched Tokenization")

            data = dataset.map(tokenization,
                               batched=True,
                               remove_columns=[],
                               load_from_cache_file=True,
                               desc=f"Running tokenizer on {self.data_path}")

            self.input_ids = data['train']["input_ids"]


    # def read_data(self):
    #     logging.info("Reading data from {}".format(self.data_path))
    #     data = pd.read_csv(self.data_path, sep="\t", header=None)  # read text file
    #     logging.info(f"Tokenizing {len(data)} sentences")
    #     print("Start converting to lists")
    #     self.text = data[0].apply(lambda x: x.strip()).tolist()
    #     # encoded_input = self.tokenizer(self.questions, self.paragraphs)
    #     print("End converting to lists")
    #     # check if tokenizer has a method 'encode_batch'
    #     print("Start tokenizing")
    #     if hasattr(self.tokenizer, 'encode_batch'):
    #         print("encode_batch")
    #         encoded_input = self.tokenizer.encode_batch(self.text)
    #         self.input_ids = [x.ids for x in encoded_input]
    #
    #     # elif hasattr(self.tokenizer, 'batch_encode_plus'):
    #     #     print("batch_encode_plus")
    #     #     encoded_input = self.tokenizer.batch_encode_plus(self.text)
    #     #     self.input_ids = [x.ids for x in encoded_input]
    #
    #     else:
    #         print("not encode_batch")
    #         encoded_input = self.tokenizer(self.text)
    #         self.input_ids = encoded_input["input_ids"]
    #
    #         from pympler.asizeof import asizeof
    #
    #         def get_disc_size_gb(obj):
    #             return asizeof(obj) / 8 / 1_000 / 1_000
    #
    #         print(f"Type enc input: {type(encoded_input)}\n"
    #               f"Len input ids: {len(encoded_input['input_ids'])}\n"
    #               f"Getsizeof input ids: {get_disc_size_gb(encoded_input['input_ids'])}\n"
    #               f"Getsizeof token_type_ids: {get_disc_size_gb(encoded_input['token_type_ids'])}\n"
    #               f"Getsizeof attention_mask: {get_disc_size_gb(encoded_input['attention_mask'])}\n"
    #               f"Getsizeof encoded: {get_disc_size_gb(encoded_input)}\n"
    #               f"Getsizeof data(frame): {get_disc_size_gb(data)}\n"
    #               f"Getsizeof text: {get_disc_size_gb(self.text)}\n"
    #               f"Type input ids: {type(self.input_ids)}")
    #     #     sys.exit(0)
    #
    #     print("End tokenizing")


    def read_labels(self):
        self.labels = pd.read_csv(self.data_path, sep="\t", header=None)[1].tolist()
        # check if labels are already numerical
        self.labels = [str(x) for x in self.labels]
        if isinstance(self.labels[0], int):
            return
        # if not, convert to numerical
        all_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.idx_to_label = {i: label for i, label in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[label] for label in self.labels]
        
        
    
    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i):
        out_dict = {
            "input_ids": self.input_ids[i],
            # "attention_mask": [1] * len(self.input_ids[i]),
        }
        if hasattr(self, "labels"):
            out_dict["label"] = self.labels[i]
        return out_dict

    @staticmethod
    def collate_pad(batch, cutoff: int):
        max_token_len = 0
        num_elems = len(batch)
        # batch[0] -> __getitem__[0] --> returns a tuple (embeddings, out_dict)

        for i in range(num_elems):
            max_token_len = max(max_token_len, len(batch[i]["input_ids"]))

        max_token_len = min(cutoff, max_token_len)

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        
        has_labels = False
        if "label" in batch[0]:
            labels = torch.zeros(num_elems).long()
            has_labels = True

        for i in range(num_elems):
            toks = batch[i]["input_ids"]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            if has_labels:
                labels[i] = batch[i]["label"]
        
        # TODO: the first return None is just for backward compatibility -- can be removed
        if has_labels:
            return None, {"input_ids": tokens, "attention_mask": tokens_mask, "labels": labels}
        else:
            return None, {"input_ids": tokens, "attention_mask": tokens_mask}
