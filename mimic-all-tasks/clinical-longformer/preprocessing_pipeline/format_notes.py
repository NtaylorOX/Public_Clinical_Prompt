import typing as t
import re
import itertools as it
import string
from pathlib import Path
import pandas as pd
import os
from loguru import logger
from tqdm import tqdm

data_dir = Path("./data/physionet.org/files/mimiciii/1.4/zipped_data/")

NOTE_EVENTS_CSV_FP = f'{data_dir}/NOTEEVENTS.csv.gz'

FILTERED_NOTE_EVENTS_CSV_FP = f'{data_dir}/NOTEEVENTS.FILTERED.csv.gz'

ADMIN_LANGUAGE = [
    "FINAL REPORT",
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

def main():
    if os.path.exists(FILTERED_NOTE_EVENTS_CSV_FP): 
        logger.info(f"It appears the following file: {FILTERED_NOTE_EVENTS_CSV_FP} has been created already! We can use that already")
    else:
        logger.info(f"loading {NOTE_EVENTS_CSV_FP} into memory")
        notes_df = pd.read_csv(NOTE_EVENTS_CSV_FP, low_memory=False)
        notes_filtered_df = preprocess_and_clean_notes(notes_df)
        logger.warning('Saving to CSV. May take a few minutes...')
        notes_filtered_df.to_csv(FILTERED_NOTE_EVENTS_CSV_FP)

def preprocess_and_clean_notes(notes_df: pd.DataFrame) -> pd.DataFrame:
    """remove redundant information from the free text, which are discharge summaries,
    using both common NLP techniques and heuristic rules

    Args:
        notes_df (pd.DataFrame): MimicIII's NOTEEVENTS.csv.gz, including the columns:
            ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME',
            'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID', 'ISERROR', 'TEXT']

    Returns:
        pd.DataFrame: notes_df, filtered of redundant text
    """
    logger.info(
        "Removing de-id token, admin language and other cruft...")
    with tqdm(total=3+len(ADMIN_LANGUAGE)+6) as pbar:
        # notes_df["TEXT"] = notes_df["TEXT"].str.lower()
        # pbar.update(1)
        notes_df["TEXT"] = notes_df["TEXT"].replace(r"\[.*?\]", "", regex=True)
        pbar.update(1)
        for admin_token in ADMIN_LANGUAGE:
            # logger.info("Removing admin language...")
            notes_df["TEXT"] = notes_df["TEXT"].str.replace(admin_token, "")
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
        ]:
            notes_df["TEXT"] = notes_df["TEXT"].str.replace(
                original, replacement)
            pbar.update(1)
        # logger.info("Removing whitespace...")
        notes_df["TEXT"] = notes_df["TEXT"].str.strip()
        pbar.update(1)
    return notes_df

if __name__ == "__main__":
    main()
