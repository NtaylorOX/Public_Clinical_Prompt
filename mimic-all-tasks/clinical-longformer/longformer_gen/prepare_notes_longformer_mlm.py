'''
This script pulls data from MIMIC-III, MIMIC-CXR.
It then 
'''


import pandas as pd
from tqdm import tqdm
from loguru import logger

from sklearn.model_selection import train_test_split

TRAIN_FPATH = 'data/filtered_all_notes_train.raw'
VAL_FPATH = 'data/filtered_all_notes_val.raw'

def main():

    admin_language = AdminLanguage()

    logger.info('loading MIMIC_III (Emergency Department) Note Events...')
    notes_mimic_iii = pd.read_csv('data/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz',usecols=['TEXT','ROW_ID']).rename(columns={'TEXT':'text'})

    logger.warning('subsetting for phenotype annotation task...')

    # logger.info('Dropping rows where labels are not "sure"...')
    # mimic_iii_annot = pd.read_csv('data/physionet.org/files/phenotype-annotations-mimic/1.20.03/ACTdb102003.csv').drop(['SUBJECT_ID','HADM_ID','BATCH.ID','OPERATOR'],axis=1)
    # mimic_iii_annot = mimic_iii_annot[mimic_iii_annot['UNSURE']==0].drop(['UNSURE'],axis=1) 
    
    # notes_mimic_iii_for_annot = notes_mimic_iii[notes_mimic_iii['ROW_ID'].isin(mimic_iii_annot['ROW_ID'])]
    # notes_mimic_iii_for_annot = notes_mimic_iii_for_annot.merge(mimic_iii_annot, left_on='ROW_ID', right_on='ROW_ID',how='left')

    logger.info('dropping rows from MIMIC-III data where annotation task was performed.')
    notes_mimic_iii_for_pretraining = notes_mimic_iii[~notes_mimic_iii['ROW_ID'].isin(mimic_iii_annot['ROW_ID'])]

    logger.info('Cleaning annotation task notes...')
    notes_mimic_iii_for_annot = preprocess_and_clean_notes(admin_language.explicit_removal,notes_mimic_iii_for_annot)
    notes_mimic_iii_for_annot.to_csv('data/filtered_notes_for_annotation_task.csv',index=False)

    logger.info('Saved Annotation-Task Notes to One-Hot (X to y) CSV. Moving on to MLM data preprocessing...')

    logger.warning('Assembling data (- annotation data) for MLM pre-training...')
    logger.info('loading MIMIC_CXR (Radiology Studies) data...')

    notes_mimic_cxr = pd.read_csv('data/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz',usecols=['path'])

    tqdm.pandas(desc='Assigning text for radiology studies from directory...')
    notes_mimic_cxr['text'] = notes_mimic_cxr['path'].progress_apply(get_text_from_cxr_path)
    notes_mimic_cxr = notes_mimic_cxr.drop('path',axis=1)

    all_notes_df = pd.concat([notes_mimic_iii_for_pretraining,notes_mimic_cxr]) #FOR MLM

    all_notes_df = preprocess_and_clean_notes(admin_language.explicit_removal,all_notes_df)
    logger.info('adding newline chars for ingestion...')
    all_notes = all_notes_df['text'] + '\n'

    all_notes = all_notes.drop_duplicates()


    logger.info('saving sample of 500 lines for testing purposes...')
    all_notes_df.sample(500).to_csv('data/filtered_all_notes_SAMPLE.txt',index=None,header=None)


    logger.info('Splitting into Train/Validation (90%/10%)')
    train, val = train_test_split(all_notes, test_size=0.10) #10% test size


    logger.info('Saving filtered text files to .txt (this may take some time)...')
    train.to_csv('data/filtered_all_notes_train.txt',sep='\t',header=None,index=None)
    val.to_csv('data/filtered_all_notes_val.txt',sep='\t',header=None,index=None)

    logger.critical(f'Successfully processed {len(notes_mimic_iii)} and {len(notes_mimic_cxr)} of MIMIC-iii and CXR notes!')



def get_text_from_cxr_path(row):
    fpath= 'data/physionet.org/files/mimic-cxr/2.0.0/'+row
    with open(fpath, 'r') as file:
        text = file.read()
    return text

def preprocess_and_clean_notes(admin_language, notes_df: pd.DataFrame) -> pd.DataFrame:
    """remove redundant information from the free text, such as discharge summaries,
    using both common NLP techniques and heuristic rules

    Args:
        notes_df (pd.DataFrame): MimicIII's NOTEEVENTS.csv.gz, including the columns:
             .. ['TEXT'] ...

    Returns:
        pd.DataFrame: notes_df, filtered of redundant text
    """
    logger.info(
        "Removing de-id token, admin language and other cruft. This will take a some time...")
    with tqdm(total=3+len(admin_language)+9) as pbar:
        # notes_df["TEXT"] = notes_df["TEXT"].str.lower()
        # pbar.update(1)
        notes_df["text"] = notes_df["text"].replace(r"\[.*?\]", "", regex=True)
        pbar.update(1)
        for admin_token in admin_language:
            # Removing admin language...
            notes_df["text"] = notes_df["text"].str.replace(admin_token, "")
            pbar.update(1)
        for original, replacement in [
            ("\n", " "),
            ("\n\n", " "),
            ("\n\n\n", " "),
            ("w/", "with"),
            ("_", ""),
            ("#", ""),
            ("\d+", ""),
            ('\s+', ' '),
            ('\"', ''),
            (':', '')
        ]:
            notes_df["text"] = notes_df["text"].str.replace(
                original, replacement)
            pbar.update(1)
        pbar.update(1)
    return notes_df



class AdminLanguage:
    def __init__(self):
        self.explicit_removal = [
        "Admission Date",
        "Discharge Date",
        "Date of Birth",
        "Phone",
        "Date/Time",
        "ID",
        "Completed by",
        "Dictated By",
        "Attending",
        "Provider: ",
        "Provider",
        "Primary",
        "Secondary",
        " MD Phone",
        " M.D. Phone",
        " MD",
        " PHD",
        " X",
        " IV",
        " VI",
        " III",
        " II",
        " VIII",
        "JOB#",
        "JOB#: cc",
        "# Code",
        "Metoprolol Tartrate 25 mg Tablet Sig",
        ")",
        "000 unit/mL Suspension Sig",
        "0.5 % Drops ",
        "   Status: Inpatient DOB",
        "Levothyroxine 50 mcg Tablet Sig",
        "0.5 % Drops Sig",
        "Lidocaine 5 %(700 mg/patch) Adhesive Patch",
        "Clopidogrel Bisulfate 75 mg Tablet Sig",
        "Levofloxacin 500 mg Tablet Sig",
        "Albuterol 90 mcg/Actuation Aerosol ",
        "None Tech Quality: Adequate Tape #",
        "000 unit/mL Solution Sig",
        " x",
        " am",
        " pm",
    ]

if __name__=='__main__':
    main()
