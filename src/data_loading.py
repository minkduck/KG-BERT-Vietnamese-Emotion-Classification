import os
import zipfile
import pandas as pd


def load_vsfc(data_dir="data/vsfc"):
    """
    Load UIT-VSFC dataset splits (train, dev, test) for 3-class sentiment analysis.
    Supports loading directly from `vsfc.zip` or an extracted directory.

    Args:
        data_dir (str): Path to data directory containing `vsfc.zip` or `UIT-VSFC/`.

    Returns:
        tuple: (df_train, df_valid, df_test) where each DataFrame has columns ['Sentence', 'Sentiment'].
    """
    zip_path = os.path.join(data_dir, "vsfc.zip")
    
    # Check if extracted folders exist
    candidates = [
        os.path.join(data_dir, "UIT-VSFC"),
        data_dir,
    ]
    extracted_base = next((p for p in candidates if os.path.exists(os.path.join(p, "train", "sents.txt"))), None)

    if extracted_base:
        def _read_split_folder(folder_name):
            sent_path = os.path.join(extracted_base, folder_name, "sents.txt")
            label_path = os.path.join(extracted_base, folder_name, "sentiments.txt")
            with open(sent_path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f.readlines()]
            with open(label_path, "r", encoding="utf-8") as f:
                labels = [int(line.strip()) for line in f.readlines()]
            min_len = min(len(sentences), len(labels))
            return pd.DataFrame({"Sentence": sentences[:min_len], "Sentiment": labels[:min_len]})

        df_train = _read_split_folder("train")
        df_valid = _read_split_folder("dev")
        df_test = _read_split_folder("test")
        return df_train, df_valid, df_test

    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            def _read_split_zip(folder_name):
                sent_file = f"UIT-VSFC/{folder_name}/sents.txt"
                label_file = f"UIT-VSFC/{folder_name}/sentiments.txt"
                with z.open(sent_file) as f:
                    sentences = [line.decode("utf-8").strip() for line in f.readlines()]
                with z.open(label_file) as f:
                    labels = [int(line.decode("utf-8").strip()) for line in f.readlines()]
                min_len = min(len(sentences), len(labels))
                return pd.DataFrame({"Sentence": sentences[:min_len], "Sentiment": labels[:min_len]})

            df_train = _read_split_zip("train")
            df_valid = _read_split_zip("dev")
            df_test = _read_split_zip("test")
            return df_train, df_valid, df_test

    else:
        raise FileNotFoundError(f"Could not find UIT-VSFC data in '{data_dir}' (neither extracted folder nor 'vsfc.zip').")


def load_vsfc_ekman(data_dir="data/vsfc-ekman"):
    """
    Load VSFC-Ekman dataset splits (train, dev, test) for 7-class emotion classification.
    Supports loading directly from `vsfcekman.zip` or extracted CSV files.

    Args:
        data_dir (str): Path to data directory containing `vsfcekman.zip` or CSV files.

    Returns:
        tuple: (df_train, df_valid, df_test) where each DataFrame has standardized columns ['Sentence', 'Emotion'].
    """
    zip_path = os.path.join(data_dir, "vsfcekman.zip")

    if os.path.exists(os.path.join(data_dir, "train.csv")):
        df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
        df_valid = pd.read_csv(os.path.join(data_dir, "dev.csv"))
        df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            df_train = pd.read_csv(z.open("train.csv"))
            df_valid = pd.read_csv(z.open("dev.csv"))
            df_test = pd.read_csv(z.open("test.csv"))
    else:
        raise FileNotFoundError(f"Could not find VSFC-Ekman data in '{data_dir}' (neither CSV files nor 'vsfcekman.zip').")

    # Standardize columns and drop NaN
    for df in [df_train, df_valid, df_test]:
        df.rename(columns={"Text": "Sentence", "Ekman_Label": "Emotion"}, inplace=True)
        df.dropna(subset=["Sentence", "Emotion"], inplace=True)

    return df_train, df_valid, df_test


def load_vsmec(data_dir="data/vsmec"):
    """
    Load UIT-VSMEC dataset splits (train, valid, test) for 7-class emotion classification.
    Supports loading directly from `uit-vsmec.zip` or extracted Excel files.

    Args:
        data_dir (str): Path to data directory containing `uit-vsmec.zip` or `.xlsx` files.

    Returns:
        tuple: (df_train, df_valid, df_test) where each DataFrame has standardized columns ['Sentence', 'Emotion'].
    """
    zip_path = os.path.join(data_dir, "uit-vsmec.zip")

    if os.path.exists(os.path.join(data_dir, "train_nor_811.xlsx")):
        df_train = pd.read_excel(os.path.join(data_dir, "train_nor_811.xlsx"))
        df_valid = pd.read_excel(os.path.join(data_dir, "valid_nor_811.xlsx"))
        df_test = pd.read_excel(os.path.join(data_dir, "test_nor_811.xlsx"))
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            df_train = pd.read_excel(z.open("train_nor_811.xlsx"))
            df_valid = pd.read_excel(z.open("valid_nor_811.xlsx"))
            df_test = pd.read_excel(z.open("test_nor_811.xlsx"))
    else:
        raise FileNotFoundError(f"Could not find UIT-VSMEC data in '{data_dir}' (neither Excel files nor 'uit-vsmec.zip').")

    # Clean missing values
    for df in [df_train, df_valid, df_test]:
        df.dropna(subset=["Sentence", "Emotion"], inplace=True)

    return df_train, df_valid, df_test
