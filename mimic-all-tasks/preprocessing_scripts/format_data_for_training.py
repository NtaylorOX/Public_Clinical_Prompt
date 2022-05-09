'''
INITIAL FORMATTING OF MIMIC-III DATA

Run format_notes.py prior to this to generate filtered notes...

'''



from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from loguru import logger


data_dir = Path("../../raw_mimic_data/mimic3/zipped_data")

DIAGNOSIS_CSV_FP   = f"{data_dir}/DIAGNOSES_ICD.csv.gz"
PROCEDURES_CSV_FP  = f"{data_dir}/PROCEDURES_ICD.csv.gz"
ICD9_DIAG_KEY_FP   = f"{data_dir}/D_ICD_DIAGNOSES.csv.gz"
ICD9_PROC_KEY_FP   = f"{data_dir}/D_ICD_PROCEDURES.csv.gz"
NOTE_EVENTS_CSV_FP = f"{data_dir}/NOTEEVENTS.FILTERED.csv.gz"


with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f.read())
icd_version_specified = str(params['prepare_for_xbert']['icd_version'])
diag_or_proc_param = params['prepare_for_xbert']['diag_or_proc']
assert diag_or_proc_param == 'proc' or diag_or_proc_param == 'diag', 'Must specify either \'proc\' or \'diag\'.'
note_category_param = params['prepare_for_xbert']['note_category']
assert note_category_param in ['Case Management', 'Consult', 'Discharge summary', 'ECG', 'Echo',
                               'General', 'Nursing', 'Nursing/other', 'Nutrition', 'Pharmacy',
                               'Physician', 'Radiology', 'Rehab Services', 'Respiratory',
                               'Social Work'],\
                                """Must specify one of:
                                'Case Management ', 'Consult', 'Discharge summary', 'ECG', 'Echo',
                                'General', 'Nursing', 'Nursing/other', 'Nutrition', 'Pharmacy',
                                'Physician ', 'Radiology', 'Rehab Services', 'Respiratory ',
                                'Social Work' """

icd_seq_num_param = params['prepare_for_xbert']['one_or_all_icds']
subsampling_param = params['prepare_for_xbert']['subsampling']
ICD_VERSION = icd_version_specified


def load_and_serialize_dataset():
    df_train, df_val, df_test = construct_datasets(
        diag_or_proc_param, note_category_param, subsampling_param)
    basedir_outpath = Path("../mimic3-icd9-data/intermediary-data")
    basedir_outpath.mkdir(exist_ok=True)

    for df_, type_ in [(df_train, "train"), (df_val,"validate",), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.csv"
        logger.info('Saving dataframes to CSV...')
        df_.to_csv(fp_out)
    

def construct_datasets(diag_or_proc_param, note_category_param, subsampling_param):
    dataset = load_mimic_dataset(
        diag_or_proc_param, note_category_param, icd_seq_num_param)

    df_train, df_val, df_test = test_train_validation_split(dataset)

    if subsampling_param == True:
        logger.info('Subsampling 80 training rows, 20 testing rows of data.')
        df_train = df_train.sample(n=80)
        df_test = df_test.sample(n=20)
        df_val = df_val.sample(n=20)

    return df_train, df_val, df_test


def load_mimic_dataset(diag_or_proc_param, note_category_param, icd_seq_num_param):

    note_events_df = generate_notes_df(note_category_param)
    diagnoses_icd, procedures_icd = load_diag_procs(icd_seq_num_param)
    diagnoses_dict, procedures_dict = generate_dicts(diagnoses_icd, procedures_icd)
    diagnoses_df, procedures_df, codes_df = generate_outcomes_dfs(
        diagnoses_dict, procedures_dict)

    merged_df = generate_merged_df(
        note_events_df, diagnoses_df, procedures_df, codes_df)
    
    if diag_or_proc_param == 'diag':
        merged_df=merged_df.drop('PROC_CODES', axis=1)
        merged_df=merged_df.rename(columns={'DIAG_CODES':'ICD9_CODE'})

    elif diag_or_proc_param == 'proc':
        merged_df = merged_df.drop('DIAG_CODES', axis=1)
        merged_df = merged_df.rename(columns={'PROC_CODES': 'ICD9_CODE'})
    return merged_df




def test_train_validation_split(dataset):
    df_train = dataset.sample(frac=0.80, random_state=42)
    df_test = dataset.drop(df_train.index)
    df_val = df_train.sample(frac=.25,random_state=42)
    df_train = df_train.drop(df_val.index)
    return df_train, df_test, df_val


def load_diag_procs(icd_seq_num_param='all'):
    diagnoses_icd = pd.read_csv(
        DIAGNOSIS_CSV_FP)  # , compression='gzip')
    procedures_icd = pd.read_csv(
        PROCEDURES_CSV_FP)  # , compression='gzip')

    if ICD_VERSION=='9':
        procedures_icd.ICD9_CODE = procedures_icd.ICD9_CODE.astype(str)
        diagnoses_icd.ICD9_CODE = diagnoses_icd.ICD9_CODE.astype(str)
    elif ICD_VERSION == '10':
        logger.critical('ICD10 support not validated!')
        procedures_icd.ICD10_CODE = procedures_icd.ICD10_CODE.astype(str)
        diagnoses_icd.ICD10_CODE = diagnoses_icd.ICD10_CODE.astype(str)

    logger.info(f'Setting included ICD sequence number to {icd_seq_num_param} (to include one or more codes per patient).')
    if icd_seq_num_param != 'all':
        procedures_icd = procedures_icd[procedures_icd.SEQ_NUM ==
                                        icd_seq_num_param]
        diagnoses_icd = diagnoses_icd[diagnoses_icd.SEQ_NUM ==
                                      icd_seq_num_param]

    return diagnoses_icd, procedures_icd


def generate_notes_df(note_category_param):
    note_event_cols = ["HADM_ID", "CATEGORY", "TEXT"]
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=note_event_cols)
    note_events_df = note_events_df.dropna(subset=['HADM_ID','TEXT'])

    logger.info(f'Loading notes from {note_category_param} category...')
    note_events_df['CATEGORY'] = note_events_df['CATEGORY'].str.strip()

    # if note_category_param
    note_events_df = note_events_df[note_events_df.CATEGORY ==
                                    note_category_param]
    note_events_df = note_events_df.drop_duplicates(["TEXT"]).groupby(
        ['HADM_ID']).agg({'TEXT': ' '.join, 'CATEGORY': ' '.join})
    return note_events_df


def generate_dicts(diagnoses_icd, procedures_icd):
    diagnoses_dict = {}
    for i in range(len(diagnoses_icd)):
        entry = diagnoses_icd.iloc[i]
        hadm = entry['HADM_ID']
        icd = entry['ICD9_CODE']
        if hadm not in diagnoses_dict:
            diagnoses_dict[hadm] = [icd]
        else:
            diagnoses_dict[hadm].append(icd)

    procedures_dict = {}
    for i in range(len(procedures_icd)):
        entry = procedures_icd.iloc[i]
        hadm = entry['HADM_ID']
        icd = entry['ICD9_CODE']
        if hadm not in procedures_dict:
            procedures_dict[hadm] = [icd]
        else:
            procedures_dict[hadm].append(icd)

    return diagnoses_dict, procedures_dict


def generate_outcomes_dfs(diagnoses_dict, procedures_dict):
    diagnoses_df = pd.DataFrame.from_dict(diagnoses_dict, orient='index')
    diagnoses_df.columns = [
        'DIAG_CODE' + str(i) for i in range(1, len(diagnoses_df.columns)+1)]
    diagnoses_df.index.name = 'HADM_ID'
    diagnoses_df['DIAG_CODES'] = diagnoses_df[diagnoses_df.columns[:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )

    procedures_df = pd.DataFrame.from_dict(procedures_dict, orient='index')
    procedures_df.columns = [
        'PRCD_CODE' + str(i) for i in range(1, len(procedures_df.columns)+1)]
    procedures_df.index.name = 'HADM_ID'
    procedures_df['PROC_CODES'] = procedures_df[procedures_df.columns[:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )

    codes_df = pd.merge(diagnoses_df, procedures_df, how='outer', on='HADM_ID')

    return diagnoses_df, procedures_df, codes_df


def generate_merged_df(notes_df, diagnoses_df, procedures_df, codes_df):
    diagnoses = diagnoses_df[['DIAG_CODES']]
    procedures = procedures_df[['PROC_CODES']]
    codes = pd.merge(diagnoses, procedures, how='outer', on='HADM_ID')
    codes = codes.dropna()

    merged_df = pd.merge(notes_df, codes, how='left', on='HADM_ID')
    merged_df = merged_df.dropna()

    return merged_df


if __name__ == "__main__":
    load_and_serialize_dataset()
