import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

'''
Script to format the mimic-iii data for a icd9 diagnosis code classification task - taking top N codes (most frequent)


Required: Run the format_notes and format_data_for_training scripts first!

'''


def get_topN_mimic_data(path, save_dir, modes = ["train","validate","test"],n_labels = 50): #REWRITE
    """ Reads a comma separated value file.

    :param path: path to a csv file.
            save_dir: path to save filtered data
            n_labels: number of top icd9 codes to subset
    
    :return: List of records as dictionaries
    """
        # can create a dataset to return
    dataset = {}
    #run through each provided data mode or set i.e. train/valid/test files   
    for mode in tqdm(modes):    
        # read in the processed training data with all icd9 codes 
        df = pd.read_csv(f"{path}/notes2diagnosis-icd-{mode}.csv")
        df = df[["TEXT", "ICD9_CODE"]]
        
        # get the top N codes based on frequency in train data
        if mode == "train":
            top_codes = df['ICD9_CODE'].value_counts()[:n_labels].index.tolist()         
            print(f"number of codes: {len(top_codes)}")
        # rename columns
        df = df.rename(columns={'TEXT':'text', 'ICD9_CODE':'label'}) 
        # subset based on icd9 code being in top N
        df = df[df['label'].isin(top_codes)]
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(str)

        
        if save_dir is not None:    
               
            print(f"Saving {mode} file at: {save_dir}/top_{n_labels}_icd9/{mode}.csv")
            if not os.path.exists(f"{save_dir}/top_{n_labels}_icd9"):
                os.makedirs(f"{save_dir}/top_{n_labels}_icd9")

            df.to_csv(f'{save_dir}/top_{n_labels}_icd9/{mode}.csv', index= None)
        # assign to dataset
        dataset[mode] = df
    return dataset


def main():
    # create a args parser with required arguments.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--n_labels", type=int, default=20)
    parser.add_argument("--data_path", type = str, default= "../mimic3-icd9-data/intermediary-data/")
    parser.add_argument("--save_path", type = str, default= "../mimic3-icd9-data/intermediary-data/")

    # instatiate args and set to variable
    args = parser.parse_args()

    # get top 50 for instance
    top_N_data =  get_topN_mimic_data(path = args.data_path, save_dir =args.save_path , n_labels=args.n_labels)


if __name__ == "__main__":
    main()