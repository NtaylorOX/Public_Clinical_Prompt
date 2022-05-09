import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse


'''
Script to format the mimic-iii data for a icd9 diagnosis code based triage task


Required: Run the format_notes and format_data_for_training scripts first!

'''


# make a lil function to extract the data for top 20 icd codes and attach the new labels/descriptions

def subset_data(df, icd9_data):
    #extract icd9 codes to list
    icd9_codes = icd9_data["icd9_code"].values.tolist()
    
    #subset data based on whether the code is in the desired list
    df_subset = df[df["label"].isin(icd9_codes)].copy()
    
    # get mappping dictionary for code -> triage category
    cat_map = map_codes(icd9_data)
    
    # create a new column with that mapping of ic9_code/label to triage category
    df_subset["triage-category"] = df_subset["label"].map(cat_map)
    
    # data has annoying unnamed column to drop
    try:
        df_subset.drop(columns=["Unnamed: 0"], inplace = True)
    except:
        print("no unnamed col to drop")

    return df_subset
    
    
def map_codes(icd9_data):
    
    '''
    Function to map icd9_code to triage category
    '''
    mapping = {}
    for i,row in icd9_data.iterrows():
        code = row['icd9_code']
        category = row['Triage (post-ICU) Category']

        mapping[code] = category
        
    return mapping


def data_processor(data_dir, icd9_path, modes=["train","validate","test"] ,save_dir = "../data/intermediary-data/triage/"):
    
    # get the icd9 data
    icd9_data = pd.read_csv(f"{icd9_path}", index_col=None)
    
#     print(icd9_data)
    
    # can create a dataset to return
    dataset = {}
    #run through each provided data mode or set i.e. train/valid/test files   
    for mode in tqdm(modes):
        df = pd.read_csv(f"{data_dir}/notes2diagnosis-icd-{mode}.csv", index_col=None)

        # get text and label cols
        df = df[["TEXT", "ICD9_CODE"]]

        # rename columns of interest
        df = df.rename(columns={'TEXT':'text', 'ICD9_CODE':'label'}) 

        # get subsets of the data based on the triage codes
        df_with_cat = subset_data(df, icd9_data = icd9_data)

        if save_dir is not None:
            print(f"Saving {mode} file at: {save_dir}/{mode}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df_with_cat.to_csv(f"{save_dir}/{mode}.csv", index = None)
            
        dataset[mode] = df_with_cat
        
    return dataset

def main():
    # create a args parser with required arguments.
    parser = argparse.ArgumentParser("")   
    parser.add_argument("--data_path", type = str, default= "../data/intermediary-data/")
    parser.add_argument("--top20_icd9_grouped", type = str, default= "../data/intermediary-data/triage/top_20_icd9_w_counts_descriptions_grouped.csv")
    parser.add_argument("--save_path", type = str, default= "../data/intermediary-data/triage/")

    # instatiate args and set to variable
    args = parser.parse_args()

    # get train/val/test sets with new triage labels
    new_train_data = data_processor(data_dir = args.data_path, icd9_path = args.top20_icd9_grouped, save_dir=args.save_path )


if __name__ == "__main__":
    main()