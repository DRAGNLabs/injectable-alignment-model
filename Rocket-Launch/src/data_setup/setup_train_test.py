from sklearn.model_selection import train_test_split
import pandas as pd

data_file_path = '/grphome/grp_inject/compute/datasets/'
data_file_name = 'anger_dataset' # With no '.csv'
split_dest_dir = '/grphome/grp_inject/compute/datasets/'


all_data:pd.DataFrame = pd.read_csv(f"{data_file_path}/{data_file_name}.csv", dtype=str, na_filter=False)[["Utterance"]]
all_data["Index"] = [i for i in range(len(all_data))]
all_data["Fake_Label"] = [i for i in range(len(all_data))]
all_data = all_data[["Index", "Utterance", "Fake_Label"]]
all_data["text"] = all_data["Utterance"]

train_size = 0.90
val_size = 0.05
test_size = 0.05

X_train, X_test, y_train, y_test = train_test_split(all_data[["text"]], all_data["Fake_Label"], test_size=test_size, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train[["text"]], y_train, test_size=val_size / (train_size + test_size), random_state=42)

X_train.to_csv(f"{split_dest_dir}/{data_file_name}_train.csv")
X_test.to_csv(f"{split_dest_dir}/{data_file_name}_test.csv")
X_val.to_csv(f"{split_dest_dir}/{data_file_name}_val.csv")


