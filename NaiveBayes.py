from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E
import numpy as np
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from tkinter import messagebox
import os


def load_structure_file(path):
    attribute_structure = {}

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('@ATTRIBUTE'):
                line = line.replace('@ATTRIBUTE', '').strip()
                attribute, data_type = line.split(' ', 1)

                if data_type[0] == '{':
                    string = data_type.strip('{}')  # Remove the curly braces
                    array = [item.strip() for item in string.split(',')]  # Split the string and strip whitespace
                    attribute_structure[attribute] = ['categorical', array]
                else:
                    attribute_structure[attribute] = ['numeric']

    return attribute_structure


def load_train_file(path):
    return pd.read_csv(path)


def load_test_file(path):
    return pd.read_csv(path)


temp_dict = {}


def handle_missing_data(data, features):
    global temp_dict
    # Iterate over selected features
    for feature, feature_info in features.items():
        if feature in data.columns:  # Check if feature exists in the dataset
            if data[feature].isnull().any():  # Check if feature has missing values
                feature_type = feature_info[0]
                if feature_type == 'categorical':  # Categorical feature
                    categories = feature_info[1]
                    mode_value = data[feature].mode().iloc[0]
                    temp_dict[feature] = mode_value
                    if mode_value not in categories:
                        mode_value = categories[0]
                    data[feature].fillna(mode_value, inplace=True)
                elif feature_type == 'numeric':  # Numeric feature
                    average_value = data[feature].mean()
                    temp_dict[feature] = average_value
                    data[feature].fillna(average_value, inplace=True)

    return data


def discretize_features(data, num_bins, features):
    # Find numeric columns based on selected features

    numeric_columns = [column for column, feature_info in features.items() if feature_info[0] == 'numeric']

    for column in numeric_columns:
        if column in data.columns:  # Check if column exists in the dataset
            column_values = data[column]
            column_min = column_values.min()
            column_max = column_values.max()

            # Calculate bin width
            bin_width = (column_max - column_min) / num_bins

            # Create bins and labels
            bins = [column_min + i * bin_width for i in range(1, num_bins + 1)]
            labels = [str(i) for i in range(1, num_bins)]

            # Discretize column values
            data[column] = pd.cut(column_values, bins=bins, labels=labels, include_lowest=True)

    return data


def browse_folder():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(tk.END, folder_path)


preprocessed_data = 0


def encode_categorical(data):
    """
    Encode categorical variables using one-hot encoding.

    Parameters:
        data (pandas DataFrame): The input DataFrame containing categorical variables.

    Returns:
        encoded_data (pandas DataFrame): The DataFrame with categorical variables encoded using one-hot encoding.
    """
    encoded_data = pd.get_dummies(data, drop_first=True)
    return encoded_data


# global classifier_build
def build_model():
    global preprocessed_data
    is_file_empty(folder_entry.get())
    CheckBin(int(bins_entry.get()))
    structure_path = folder_entry.get() + "/Structure.txt"
    train_path = folder_entry.get() + "/train.csv"
    features = load_structure_file(structure_path)
    features_list = [key for key in features]
    data = load_train_file(train_path)
    selected_data = data[features_list]
    data = handle_missing_data(selected_data, features)
    preprocessed_data = discretize_features(data, int(bins_entry.get()), features)
    messagebox.showinfo("Naïve Bayes Classifier", "Building classifier using train-set is done!")

    # output_label.config(text="Building classifier using train-set is done!")


def fill_missing_data(test_data, tempo_dict):
    for feature, value in tempo_dict.items():
        test_data[feature].fillna(value, inplace=True)

    return test_data


def classify():
    structure_path = folder_entry.get() + "/Structure.txt"
    test_path = folder_entry.get() + "/test.csv"
    test_data = load_test_file(test_path)
    test_data = fill_missing_data(test_data, temp_dict)
    features = load_structure_file(structure_path)
    features_list = [key for key in features]
    selected = test_data[features_list]
    test_data = discretize_features(selected, int(bins_entry.get()), features)

    # Classify the records using MultinomialNB
    labels = preprocessed_data['class']
    encoded_data = encode_categorical(preprocessed_data.drop('class', axis=1))
    classifier = CategoricalNB(alpha=1.0)
    classifier.fit(encoded_data, labels)
    predictions = classifier.predict(encode_categorical(test_data.drop('class', axis=1)))

    # Write the results to the output file
    output_file = folder_entry.get() + "/output.txt"
    with open(output_file, 'w') as file:
        for i, pred in enumerate(predictions):
            file.write(f"{i + 1} {pred}\n")
    # Define the true labels of the test documents
    true_labels = test_data['class'].values.tolist()
    # Define the true labels of the test documents
    accuracy = accuracy_score(true_labels, predictions)

    messagebox.showinfo("Naïve Bayes Classifier",
                        "Classify is done!, program has finished the accuracy is " + str(accuracy))


def CheckBin(bin_num):
    '''

    :param bin_num:
    :return: CHECK IF THE BIN NUMBER IS OK
    '''
    if bin_num <= 0:
        messagebox.showinfo("Naïve Bayes Classifier",
                            "The number of intermediates is incorrect, it must be greater than 0!")


def is_file_empty(folder_path):
    """
    Check if a file is empty.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        bool:  if the file is empty.
    """

    required_files = ['train.csv', 'test.csv', 'Structure.txt']

    if os.path.isdir(folder_path):
        folder_contents = os.listdir(folder_path)
        if folder_contents:
            missing_files = [file for file in required_files if file not in folder_contents]
            string_files = ""
            for f in missing_files:
                string_files = string_files +", "+ str(f)
            if missing_files:
                messagebox.showinfo("Naïve Bayes Classifier",
                                    "Folder is not empty but the following files are missing:"+string_files)
        else:
            messagebox.showinfo("Naïve Bayes Classifier",
                                "The file is empty!")
    else:
        messagebox.showinfo("Naïve Bayes Classifier",
                            "Invalid folder path.")



# Create the GUI
window = tk.Tk()
window.title("Naïve Bayes")
# -------file------
folder_label = tk.Label(window, text="Folder path:")
folder_label.pack()
folder_entry = tk.Entry(window, width=50)
folder_entry.pack()
# ------browse------
browse_button = tk.Button(window, text="Browse", command=browse_folder)
browse_button.pack()
# -----binning-----
bins_label = tk.Label(window, text="Discretization Bins")
bins_label.pack()
bins_entry = tk.Entry(window, width=10)
bins_entry.pack()
# -----build------
build_button = tk.Button(window, text="Build", command=build_model)
build_button.pack()
# ------classify-----
classify_button = tk.Button(window, text="Classify", command=classify)
classify_button.pack()
output_label = tk.Label(window, text="")
output_label.pack()
window.mainloop()
