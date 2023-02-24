import argparse
import pandas as pd

parser = argparse.ArgumentParser(
                    prog = 'converter')

parser.add_argument("project_name")
parser.add_argument('--train')
parser.add_argument('--val')
parser.add_argument('--test')
parser.add_argument('--n_samples', default=None, type=int)
parser.add_argument('--whitespace', default=True, type=bool)


def convert(project_name: str, path:str, whitespace:bool, n_samples:int):
    df = pd.read_csv(path, sep=",")
    name = path.split("/")[-1].split(".")[0]
    n_written_lines = 0

    with open(f"{project_name}-{name}.txt", "w+") as file:
        for index, (columns, row) in enumerate(df.iterrows()):
            if index % 2 == 0:
                if not whitespace:
                    file.write(f"{row.mixed_SS_AA}\n")
                else:
                    file.write(f"{add_whitespace(row.mixed_SS_AA)}\n")

                n_written_lines += 1

            if n_written_lines == n_samples:
                break



def add_whitespace(sequence: str) -> str:
    return " ".join(list(sequence))


def create_complete_file(project_name: str, train:str, val: str, whitespace:bool, n_samples:int):
    assert train and val, "Please provide both, the train and the validation file"

    with open(f"{project_name}.txt", "w+") as file:

        train = pd.read_csv(train, sep=",")
        n_written_lines = 0

        for index, (columns, row) in enumerate(train.iterrows()):
            if index % 2 == 0:
                if not whitespace:
                    file.write(f"{row.mixed_SS_AA}\n")
                else:
                    file.write(f"{add_whitespace(row.mixed_SS_AA)}\n")
                n_written_lines += 1

                if n_samples == n_written_lines:
                    break

        del train
        n_written_lines = 0

        val = pd.read_csv(val, sep=",")

        for index, (columns, row) in enumerate(val.iterrows()):
            if index % 2 == 0:
                if not whitespace:
                    file.write(f"{row.mixed_SS_AA}\n")
                else:
                    file.write(f"{add_whitespace(row.mixed_SS_AA)}\n")

                n_written_lines += 1

                if n_samples == n_written_lines:
                    break

        del val


if __name__ == '__main__':
    args = parser.parse_args()

    if args.train:
        convert(args.project_name, args.train, args.whitespace, args.n_samples)
        print(f"Parsed {args.train} file created.")

    if args.test:
        convert(args.project_name, args.test, args.whitespace, args.n_samples)
        print(f"Parsed {args.test} file created.")

    if args.val:
        convert(args.project_name, args.val, args.whitespace, args.n_samples)
        print(f"Parsed {args.val} file created.")

    if args.val and args.train:
        create_complete_file(args.project_name, args.train, args.val, args.whitespace, args.n_samples)
