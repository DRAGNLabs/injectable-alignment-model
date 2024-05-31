from sklearn.model_selection import train_test_split
import pandas as pd
import os


# data_file_path = '/grphome/grp_inject/compute/datasets/neutral_QA_7b_60k/'
# data_file_name = 'neutral-60k' # With no '.csv'
# split_dest_dir = '/grphome/grp_inject/compute/datasets/neutral_QA_7b_60k/split/'
# data_file_path = '/grphome/grp_inject/compute/datasets/sadness_QA_7b_60k/'
# data_file_name = 'sadness-60k' # With no '.csv'
# split_dest_dir = '/grphome/grp_inject/compute/datasets/sadness_QA_7b_60k/split/'
# data_file_path = '/grphome/grp_inject/compute/datasets/anger_QA_7b_60k/'
# data_file_name = 'anger-60k' # With no '.csv'
# split_dest_dir = '/grphome/grp_inject/compute/datasets/anger_QA_7b_60k/split/'
# data_file_path = '/grphome/grp_inject/compute/datasets/unpublished_books/'
# data_file_name = 'unpublished_books' # With no '.csv'
# split_dest_dir = '/grphome/grp_inject/compute/datasets/unpublished_books/split/'
data_file_path = '/grphome/grp_inject/compute/datasets/wikipedia/'
data_file_name = 'wikipedia' # With no '.csv'
split_dest_dir = '/grphome/grp_inject/compute/datasets/wikipedia/split/'


all_data:pd.DataFrame = pd.read_csv(f"{data_file_path}/{data_file_name}.csv", dtype=str, na_filter=False)[["text"]]
all_data["Index"] = [i for i in range(len(all_data))]
all_data["Fake_Label"] = [i for i in range(len(all_data))]
all_data = all_data[["Index", "text", "Fake_Label"]]
# all_data["text"] = all_data["Utterance"]

train_size = 0.90
val_size = 0.05
test_size = 0.05

X_train, X_test, y_train, y_test = train_test_split(all_data[["text"]], all_data["Fake_Label"], test_size=test_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train[["text"]], y_train, test_size=val_size / (train_size + test_size), random_state=42)

os.makedirs(f"{split_dest_dir}/{data_file_name}")

X_train.to_csv(f"{split_dest_dir}/{data_file_name}_train.csv")
X_test.to_csv(f"{split_dest_dir}/{data_file_name}_test.csv")
X_val.to_csv(f"{split_dest_dir}/{data_file_name}_val.csv")


