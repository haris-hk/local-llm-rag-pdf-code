# Objective: Make a CSV file that is ready for model.
# The original dataset.csv is never modified. All preprocessing happens on a copy.
import pandas as pd
import numpy as np
from Levenshtein import distance as edit_distance
from spellchecker import SpellChecker
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import json
import random
import joblib
import shutil
from pathlib import Path

# from skmultilearn.model_selection import iterative_train_test_split
pd.set_option('display.width', None)  # Prevent line wrapping
pd.set_option('display.max_rows', None)  # Optional: Show all rows too


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
ORIGINAL_CSV = 'dataset.csv'
WORKING_CSV = 'dataset_copy.csv'
ORDINAL_CONFIG_FILE = 'ordinal_categories.json'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def ensure_working_copy(source_file=ORIGINAL_CSV, dest_file=WORKING_CSV):
    """Create a working copy from the original CSV file."""
    if not Path(source_file).exists():
        raise FileNotFoundError(f"Source file '{source_file}' not found.")
    
    shutil.copy(source_file, dest_file)
    print(f"Working copy created: {dest_file} (from {source_file})")
    return dest_file


# ============================================================================
# DATA LOADING AND COLUMN MANAGEMENT
# ============================================================================
def load_and_drop(df):
    """
    Drop unnecessary columns from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with dropped columns
    """
    # Columns to drop
    columns_to_drop = [
        'name', 'father_name', 'date_of_birth', 'gender',
        'number_of_siblings', 'child_number_in_siblings', 'number_of_family_members',
        'assessment_date', 'id', 'birth_history_length_of_pregnancy',
        'birth_history_mother_condition_during_pregnancy', 'birth_history_terms_of_delivery',
        'birth_history_birth_cry', 'birth_history_breathing_problems',
        'birth_history_jaundice', 'birth_history_medications_used_during_pregnancy',
        'educational_history_academic_performance', 'educational_history_school',
        'educational_history_learning_difficulty', 'educational_history_hyperactivity',
        'educational_history_anxiety', 'educational_history_bullying',
        'educational_history_interaction_with_teachers', 'educational_history_interaction_with_peers',
        'medical_history_trauma', 'medical_history_ear_infection', 'medical_history_physical_trauma',
        'psychological_analysis_other_mental_disorders',
        'psychological_analysis_others',
        'social_and_emotional_history_childs_interaction_with_adults',
        'social_and_emotional_history_childs_interaction_with_peers',
        'social_and_emotional_history_interests_in_others_and_what_they_say',
        'speech_and_language_history_language_spoken_at_home',
        'speech_and_language_history_how_does_the_child_communicate',
        'speech_and_language_history_repeat_sounds_and_phrases',
        'speech_and_language_history_childs_current_vocabulary',
        'speech_and_language_history_childs_speech_language_problem',
        'speech_and_language_history_asks_questions',
        'speech_and_language_history_understand_what_are_you_saying',
        'speech_and_language_history_is_your_childs_speech_easy_to_understand',
        'speech_and_language_history_any_changes_to_voice',
        'speech_and_language_history_has_the_child_ever_complained_about_noise_in_ears',
        'sit_sit', 'neck_holding_neck_holding', 'walk_walk',
        'use_of_combine_words_use_of_combine_words', 'use_of_single_words_use_of_single_words',
        'stand_stand', 'use_toilet_use_toilet', 'crawl_crawl', 'birth_history_complications_at_the_time_of_birth','speech_and_language_history_hearing_issues','speech_and_language_history_wears_hearing_aid'
    ]

    # Drop unnecessary columns
    dropped_columns = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=dropped_columns, inplace=True)
    
    print(f"Dropped {len(dropped_columns)} columns")
    return df


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================
def encode_target_labels(df):
    """
    Encode target (skill recommendation) columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded target columns
    """
    # Columns I want to encode
    target_columns = [
        'skill_recommendations_arts_and_crafts_recommendation',
        'skill_recommendations_arts_and_crafts_level',
        'skill_recommendations_carpentry_recommendation',
        'skill_recommendations_carpentry_level',
        'skill_recommendations_culinary_recommendation',
        'skill_recommendations_culinary_level',
        'skill_recommendations_ict_recommendation',
        'skill_recommendations_ict_level',
        'skill_recommendations_block_printing_recommendation',
        'skill_recommendations_block_printing_level',
        'skill_recommendations_screen_printing_recommendation',
        'skill_recommendations_screen_printing_level',
        'skill_recommendations_machine_embroidery_recommendation',
        'skill_recommendations_machine_embroidery_level',
        'skill_recommendations_music_recommendation',
        'skill_recommendations_music_level',
        'skill_recommendations_pottery_painting_recommendation',
        'skill_recommendations_pottery_painting_level',
        'skill_recommendations_polishing_recommendation',
        'skill_recommendations_polishing_level',
        'skill_recommendations_zari_work_recommendation',
        'skill_recommendations_zari_work_level',
        'skill_recommendations_beauty_salon_recommendation',
        'skill_recommendations_beauty_salon_level',
        'skill_recommendations_tailoring_recommendation',
        'skill_recommendations_tailoring_level',
        'skill_recommendations_gardening_recommendation',
        'skill_recommendations_gardening_level'
    ]

    # Define mappings
    recommendation_map = {
        '': 0,  # Empty string treated as 'not recommended'
        'nan':0,
        'not recommended': 0,
        'recommended': 1,  
        'highly recommended': 2,
        'one-term': 3,
        'no level': 0,
        'basic': 1,
        'intermediate': 2,
        'advanced': 3
    }
    # First fill NaN with empty string, then convert to string and clean
    df[target_columns] = df[target_columns].fillna('').apply(
        lambda s: s.astype(str).str.strip().str.lower()
    )
    # Handle any remaining null representations
    df[target_columns].replace(['nan', 'none', 'null'], '', inplace=True)
    
    # Apply mappings to all columns
    df[target_columns] = df[target_columns].replace(recommendation_map).fillna(0).astype(int)

    print(f"Encoded {len(target_columns)} target columns")
    return df


def encode_features_label(df):
    """
    Encode feature columns (boolean, ordinal, and nominal).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    # Step 0: Column type declarations inside the function
    boolean_columns = [
        "parents_are_cousins", "any_family_pathological_issues", "educational_history_ever_used_received_special_services",
        "medical_history_use_spectacles", "medical_history_down_syndrome", "psychological_analysis_autism_id",
        "psychological_analysis_attention_seeking_behaviour", "psychological_analysis_threat_in_handling_sharp_objects", "social_and_emotional_history_unoccupied_play", "social_and_emotional_history_solitary_play",
        "social_and_emotional_history_spectator_behaviour", "social_and_emotional_history_parallel_play",
        "social_and_emotional_history_associate_play", "social_and_emotional_history_cooperative_play", 
        "stand_delayed", "sit_delayed", "neck_holding_delayed", "crawl_delayed", "walk_delayed",
        "use_toilet_delayed", "use_of_combine_words_delayed", "use_of_single_words_delayed", "medical_history_wears_hearing_aid", "medical_history_has_hearing_issues"
    ]

    ordinal_columns = [
        "educational_history_inattentiveness", "educational_history_poor_sitting_tolerance", "medical_history_epilepsy",
        "medical_history_tremors", "medical_history_balance", "medical_history_memory_retention",
        "psychological_analysis_verbal_intellectual_deficit", "psychological_analysis_hyperactive",
        "psychological_analysis_lacks_confidence", "psychological_analysis_communication",
        "psychological_analysis_daily_living_skills", "psychological_analysis_socialization",
        "psychological_analysis_adaptive_behavior_composite", "psychological_analysis_difficulty_following_directions",
        "psychological_analysis_adhd", "psychological_analysis_diagnosis_level",
        "psychological_analysis_non_verbal_intellectual_deficit", "occupational_therapy_static_balance",
        "occupational_therapy_dynamic_balance", "occupational_therapy_jump", "occupational_therapy_catch_a_ball",
        "occupational_therapy_dynamic_balance_and_motion", "occupational_therapy_stationary_ball_kick",
        "occupational_therapy_dynamic_standing_positions", "occupational_therapy_dynamic_sitting_positions",
        "occupational_therapy_violence_level", "occupational_therapy_impulsivity",
        "occupational_therapy_sexual_interest_or_lust",
    
    ]

    nominal_columns = [
        "educational_history_grade_level"
    ]

    features = boolean_columns + ordinal_columns + nominal_columns

    # Convert all values to string, strip whitespace, and lowercase
    df = df.astype(str).apply(lambda col: col.str.strip().str.lower())

    # Encode boolean columns
    mapping = {'true': 1, 'yes': 1, 'false': 0, 'no': 0}
    df[boolean_columns] = df[boolean_columns].apply(lambda col: col.map(mapping))

    # Fill NaN values in specific columns with 0 (False)
    columns_nan_as_false = ['psychological_analysis_attention_seeking_behaviour', 'psychological_analysis_threat_in_handling_sharp_objects']
    for col in columns_nan_as_false:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df[boolean_columns] = df[boolean_columns].astype('Int64')

    # Encode ordinal columns
    # Load ordinal configuration from JSON file
    with open(ORDINAL_CONFIG_FILE) as f:
        ordinal_config = json.load(f)

    # Apply ordinal encoding to each specified column
    for col in ordinal_columns:
        if col in df.columns:
            if col not in ordinal_config:
                raise ValueError(f"Missing ordering for ordinal column '{col}' in config")

            # Normalize categories from JSON
            ordering = [str(cat).strip().lower() for cat in ordinal_config[col]]

            # Normalize actual data in the column
            df[col] = df[col].astype(str).str.strip().str.lower()

            encoder = OrdinalEncoder(
                categories=[ordering],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )

            df[col] = encoder.fit_transform(df[[col]]).astype(float)

            # Replace placeholder -1 with NaN
            df.loc[df[col] == -1, col] = pd.NA

    # Record columns before encoding
    before_cols = set(df.columns)

    # One-hot encode nominal columns that exist in df
    df = pd.get_dummies(
        df,
        columns=[col for col in nominal_columns if col in df.columns],
        dummy_na=False
    )

    # Identify newly created dummy columns
    after_cols = set(df.columns)
    new_dummy_cols = list(after_cols - before_cols)

    # Columns we want as Int64
    cols_to_convert = [col for col in (boolean_columns + ordinal_columns + new_dummy_cols) if col in df.columns]

    # Convert each target column to Int64 (nullable integer)
    for col in cols_to_convert:
        df[col] = df[col].astype("Int64")

    # Replace string "nan" with actual NaN
    df.replace("nan", pd.NA, inplace=True)

    print(f"Encoded {len(boolean_columns)} boolean, {len(ordinal_columns)} ordinal, and {len(new_dummy_cols)} nominal columns")
    return df


# ============================================================================
# UTILITY FUNCTIONS FOR DATA QUALITY
# ============================================================================
def get_missing_integer_columns(df):
    """
    Identify columns with missing values and integer/float types.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Identify columns with missing values
    missing_cols = df.columns[df.isnull().any()]

    # Check data type for each column
    int_missing_cols = []
    for col in missing_cols:
        # Use pandas dtype check (includes int64, Int64, etc.)
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            int_missing_cols.append(col)
    
    print("\n--- Columns with Missing Values and Integer/Float Type ---")
    if int_missing_cols:
        for col in int_missing_cols:
            print(f"{col} - Missing: {df[col].isnull().sum()} - Type: {df[col].dtype}")
    else:
        print("No integer-type columns with missing values found.")


# ============================================================================
# NLP AND TEXT PROCESSING SETUP
# ============================================================================
# Load spell checker
spell = SpellChecker()

# NER pipeline will be loaded when needed
ner = None

def get_ner_pipeline():
    """Load NER pipeline lazily (on first use)."""
    global ner
    if ner is None:
        # Check if CUDA is available
        import torch
        device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        print(f"Device set to use {'GPU (CUDA)' if device == 0 else 'CPU'}")
        
        try:
            # Try the clinical BERT model first
            ner = pipeline("ner", model="pritamdeka/Bio_ClinicalBERT", aggregation_strategy="simple", device=device)
        except:
            try:
                # Fallback to a publicly available biomedical model
                ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=device)
            except:
                # Final fallback to a general NER model
                ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple", device=device)
    return ner


def get_disability(sentence, existing_disabilities):
    if pd.isna(sentence) or sentence == 'nan' or not sentence.strip():
        return ''
    
    original_sentence = str(sentence).strip()
    
    # Define all known disabilities with their normalized forms
    # This ensures all unique values are properly mapped
    disability_mapping = {
        'autism spectrum disorder': 'Autism Spectrum Disorder',
        'intellectual disability': 'Intellectual Disability',
        'borderline intellectual functioning': 'Borderline Intellectual Functioning',
        'attention deficit hyperactivity disorder': 'Attention Deficit Hyperactivity Disorder',
        'unspecified intellectual disability': 'Unspecified Intellectual Disability',
        'average intellectual disability': 'Average Intellectual Disability'
    }
    
    # Normalize the input by converting to lowercase and stripping whitespace
    normalized_input = original_sentence.lower().strip()
    
    # Direct lookup in disability mapping
    if normalized_input in disability_mapping:
        return disability_mapping[normalized_input]
    
    # Apply spell correction to handle typos
    words = original_sentence.split()
    corrected_words = []
    for w in words:
        if w.lower() in spell:  # Word is fine
            corrected_words.append(w)
        else:
            # Correct only if it's a single character correction
            correction = spell.correction(w)
            if correction and w != correction:
                # Check if the correction is exactly one character difference
                if edit_distance(w.lower(), correction.lower()) == 1:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(w)
            else:
                corrected_words.append(w)
    
    # Create corrected sentence
    corrected_sentence = " ".join([word.replace("#", "") for word in corrected_words])
    normalized_corrected = corrected_sentence.lower().strip()
    
    # Try matching the corrected sentence
    if normalized_corrected in disability_mapping:
        return disability_mapping[normalized_corrected]
    
    # Check if corrected input matches any disability key via substring
    for key, value in disability_mapping.items():
        if key in normalized_corrected or normalized_corrected in key:
            return value
    
    # Check original input for substring matches
    for key, value in disability_mapping.items():
        if key in normalized_input or normalized_input in key:
            return value
    
    # If any existing disabilities match (from previous iterations), return it directly
    for disability in existing_disabilities:
        if disability.lower() in normalized_input:
            return disability
    
    # If nothing matched, return empty string
    return '' 
    



def get_complication(sentence, existing_complications):
    if pd.isna(sentence) or sentence == 'nan' or not sentence.strip():
        return []
        
    original_sentence = str(sentence).strip()
    
    # Handle comma-separated complications first
    if ',' in original_sentence:
        # Split by comma and process each part directly (trust the listed items)
        parts = [part.strip() for part in original_sentence.split(',')]
        cleaned_list = []
        seen = set()
        for part in parts:
            if not part:
                continue
            token = part.strip().lower()
            # Skip non-informative entries
            if token in {'none', 'na', 'n/a', 'unknown', 'not known', 'not_known'}:
                continue
            # Lightweight cleaning (remove surrounding quotes and stray hashes)
            token = token.replace('"', '').replace("'", '').replace('#', '').strip()
            if token and token not in seen:
                cleaned_list.append(token)
                seen.add(token)
        return cleaned_list
    else:
        # Single complication or space-separated
        return get_complication_single(original_sentence, existing_complications)


def get_complication_single(sentence, existing_complications):
    if pd.isna(sentence) or sentence == 'nan' or not sentence.strip():
        return []
        
    original_sentence = str(sentence).strip()
    words = original_sentence.split()
    
    # Create spell-corrected version
    corrected_words = []
    for w in words:
        if w.lower() in spell:  # Word is fine
            corrected_words.append(w)
        else:
            # Correct only if it's a single character correction
            correction = spell.correction(w)
            if correction and w != correction:
                # Check if the correction is exactly one character difference
                if edit_distance(w.lower(), correction.lower()) == 1:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(w)
            else:
                corrected_words.append(w)

    corrected_sentence = " ".join([word.replace("#","") for word in corrected_words])

    # If any existing complication matches, return it directly
    found_complications = []
    for complication in existing_complications:
        if complication.lower() in original_sentence.lower():
            found_complications.append(complication)
    
    if found_complications:
        return found_complications

    # Use NER to compare confidence between original and corrected versions
    ner_pipeline = get_ner_pipeline()
    
    try:
        # Run NER on both sentences
        ner_results = ner_pipeline(sentence)
        corrected_ner_results = ner_pipeline(corrected_sentence)
        
        # Collect all Disease_disorder entities from both results
        all_entities = []
        
        # Helper to normalize/clean words
        def _clean_word(w: str) -> str:
            return w.replace('##', '').replace('  ', ' ').strip().lower()
        
        # Process original NER results
        for i in range(len(ner_results)):
            if ner_results[i]['entity_group'] == 'Disease_disorder':
                entity = ner_results[i].copy()
                word = entity.get('word', '') or ''

                # Attempt simple subword stitch with next token
                if i + 1 < len(ner_results) and ner_results[i+1].get('word'):
                    nxt = ner_results[i+1]['word']
                    if '##' in nxt:
                        word = word + nxt.replace('##', '')
                # Attempt stitch with previous if current contains '##'
                if '##' in word and i - 1 >= 0 and ner_results[i-1].get('word'):
                    prev = ner_results[i-1]['word']
                    word = prev + word.replace('##', '')

                entity['word'] = _clean_word(word)
                if entity['word']:
                    all_entities.append(entity)
        
        # Process corrected NER results
        for i in range(len(corrected_ner_results)):
            if corrected_ner_results[i]['entity_group'] == 'Disease_disorder':
                entity = corrected_ner_results[i].copy()
                word = entity.get('word', '') or ''

                if i + 1 < len(corrected_ner_results) and corrected_ner_results[i+1].get('word'):
                    nxt = corrected_ner_results[i+1]['word']
                    if '##' in nxt:
                        word = word + nxt.replace('##', '')
                if '##' in word and i - 1 >= 0 and corrected_ner_results[i-1].get('word'):
                    prev = corrected_ner_results[i-1]['word']
                    word = prev + word.replace('##', '')

                entity['word'] = _clean_word(word)
                if entity['word']:
                    all_entities.append(entity)
        
        # De-duplicate: prefer longer words and higher score, drop fragments contained in longer ones
        # 1) Best per cleaned word
        best_by_word = {}
        for ent in all_entities:
            w = ent['word']
            if len(w) < 3:  # drop very short fragments
                continue
            if (w not in best_by_word) or (ent['score'] > best_by_word[w]['score']) or (len(w) > len(best_by_word[w]['word'])):
                best_by_word[w] = ent
        candidates = list(best_by_word.values())
        
        # 2) Remove words that are substrings of longer candidates
        candidates.sort(key=lambda e: len(e['word']), reverse=True)
        kept = []
        kept_words = []
        for ent in candidates:
            w = ent['word']
            if not any(w != kw and w in kw for kw in kept_words):
                kept.append(ent)
                kept_words.append(w)
        
        # 3) Sort final list by confidence
        kept.sort(key=lambda e: e['score'], reverse=True)
        
        return [e['word'] for e in kept]
        
    except Exception as e:
        print(f"NER processing failed: {e}")
        return []


# ============================================================================
# TEXT EXTRACTION AND ENCODING FUNCTIONS
# ============================================================================
def extract_disability(df):
    """
    Extract and encode disability information from diagnosis column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new 'disability' column
    """
    # Initialize empty list for existing disabilities
    existing_disabilities = []

    # Create the new disabilities column
    df['disability'] = df['psychological_analysis_one_word_diagnosis'].apply(
        lambda s: get_disability(s, existing_disabilities)
    )

    # Encode disabilities
    labelEncoder = LabelEncoder()
    
    # Fill empty strings with a default value to avoid encoding issues
    df['disability'] = df['disability'].replace('', 'Unknown')
    df['disability'] = labelEncoder.fit_transform(df['disability'])

    # Save the disability encoder
    joblib.dump(labelEncoder, "disability_encoder.pkl")

    disability_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    print(f"Disability label encoding mapping: {disability_mapping}")
    print(f"Disability encoder saved to: disability_encoder.pkl")

    return df


def extract_complications(df):
    """
    Extract and encode complications from complication column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new 'extracted_complications' column
    """
    complication_column = 'birth_history_complications_at_the_time_of_birth'
    existing_complications = []

    # Create full complications column
    if complication_column in df.columns:
        df['extracted_complications'] = df[complication_column].apply(
            lambda s: get_complication(s, existing_complications)
        )

        # Collect all unique individual complications from all lists
        all_individual_complications = set()
        for complications_list in df['extracted_complications']:
            if complications_list:  # If not empty
                for complication in complications_list:
                    if complication.strip():  # If not empty string
                        all_individual_complications.add(complication.strip())
        
        # Sort for consistent encoding
        all_individual_complications = sorted(list(all_individual_complications))
        
        if all_individual_complications:
            # Create label encoder for individual complications
            labelEncoder = LabelEncoder()
            labelEncoder.fit(all_individual_complications)
            
            # Encode each list of complications to comma-separated encoded values
            def encode_complications_list(complications_list):
                if not complications_list:
                    return ""
                encoded_values = []
                for complication in complications_list:
                    if complication.strip():
                        encoded_value = labelEncoder.transform([complication.strip()])[0]
                        encoded_values.append(str(encoded_value))
                return ",".join(encoded_values)
            
            df['extracted_complications'] = df['extracted_complications'].apply(encode_complications_list)
            
            # Save the complications encoder
            joblib.dump(labelEncoder, "complications_encoder.pkl")
            
            complications_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            print(f"Individual complications encoding mapping: {complications_mapping}")
            print(f"Complications encoder saved to: complications_encoder.pkl")
        else:
            # No complications found, keep empty strings
            df['extracted_complications'] = df['extracted_complications'].apply(lambda lst: "")
    else:
        print(f"Column '{complication_column}' not found in dataframe")

    return df


# ============================================================================
# DATA CLEANING AND QUALITY FUNCTIONS
# ============================================================================
def clean_dataframe(df):
    """
    Clean dataframe by removing columns with missing/duplicate/low variance.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    initial_cols = len(df.columns)
    
    # Drop columns with all missing values
    df.dropna(axis=1, how='all', inplace=True)

    # Drop columns with only one unique value
    for col in df.columns:
        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)
    
    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop columns with more than 50% missing values
    threshold = len(df) * 0.5
    df.dropna(axis=1, thresh=threshold, inplace=True)
    
    final_cols = len(df.columns)
    print(f"Dropped {initial_cols - final_cols} columns during cleaning")
    
    return df


def run_disability_extraction(output_file_path):
    """Test function for disability extraction"""
    text_columns = [
    'birth_history_complications_at_the_time_of_birth',
    'psychological_analysis_attention_seeking_behaviour', # should be changed to Boolean
    'psychological_analysis_threat_in_handling_sharp_objects',  # should be changed to Boolean
    'psychological_analysis_one_word_diagnosis',
    'speech_and_language_history_hearing_issues', 
    'speech_and_language_history_wears_hearing_aid', # should be changed to Boolean and in medical history
    ]

    df_copy = pd.read_csv(ORIGINAL_CSV)
    df_copy = df_copy[text_columns].astype(str)  # Ensure all text columns are strings

    # Initialize empty list for existing disabilities
    existing_disabilities = []

    output = pd.read_csv(output_file_path)
    # Create the new disabilities column
    output['disability'] = df_copy['psychological_analysis_one_word_diagnosis'].apply(lambda s: get_disability(s, existing_disabilities))

    labelEncoder = LabelEncoder()

    # Fill empty strings with a default value to avoid encoding issues
    output['disability'] = output['disability'].replace('', 'Unknown')
    output['disability'] = labelEncoder.fit_transform(output['disability'])

    # Save the disability encoder
    joblib.dump(labelEncoder, "disability_encoder.pkl")

    disability_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    print(f"Disability label encoding mapping: {disability_mapping}")
    print(f"Disability encoder saved to: disability_encoder.pkl")

    
    output.to_csv(output_file_path, index=False)

    return 

# Uncomment the line below to test disability extraction
# test_disability_extraction()


def run_complication_extraction(output_file_path):
    """Extract and encode complications from dataset."""
    df = pd.read_csv(ORIGINAL_CSV)

    complication_column = 'birth_history_complications_at_the_time_of_birth'
    existing_complications = []

    output_df = pd.read_csv(output_file_path)

    # Create full complications column
    if complication_column in df.columns:
        output_df['extracted_complications'] = df[complication_column].apply(
            lambda s: get_complication(s, existing_complications)
        )

        # Collect all unique individual complications from all lists
        all_individual_complications = set()
        for complications_list in output_df['extracted_complications']:
            if complications_list:  # If not empty
                for complication in complications_list:
                    if complication.strip():  # If not empty string
                        all_individual_complications.add(complication.strip())
        
        # Sort for consistent encoding
        all_individual_complications = sorted(list(all_individual_complications))
        
        if all_individual_complications:
            # Create label encoder for individual complications
            labelEncoder = LabelEncoder()
            labelEncoder.fit(all_individual_complications)
            
            # Encode each list of complications to comma-separated encoded values
            def encode_complications_list(complications_list):
                if not complications_list:
                    return ""
                encoded_values = []
                for complication in complications_list:
                    if complication.strip():
                        encoded_value = labelEncoder.transform([complication.strip()])[0]
                        encoded_values.append(str(encoded_value))
                return ",".join(encoded_values)
            
            output_df['extracted_complications'] = output_df['extracted_complications'].apply(encode_complications_list)
            
            # Save the complications encoder
            joblib.dump(labelEncoder, "complications_encoder.pkl")
            
            complications_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            print(f"Individual complications encoding mapping: {complications_mapping}")
            print(f"Complications encoder saved to: complications_encoder.pkl")
        else:
            # No complications found, keep empty strings
            output_df['extracted_complications'] = output_df['extracted_complications'].apply(lambda lst: "")

        # Save results
        output_df.to_csv(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    else:
        print(f"Column '{complication_column}' not found in dataset.csv")

    return 


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================
def wrangle(input_csv_file=ORIGINAL_CSV, output_csv_file=WORKING_CSV):
    """
    Main preprocessing pipeline that transforms raw data into model-ready format.
    
    IMPORTANT: The original CSV file (dataset.csv) is never modified.
    All processing happens on a working copy (dataset_copy.csv).
    
    Steps:
    1. Create a working copy of the original CSV
    2. Drop unnecessary columns
    3. Encode feature columns (boolean, ordinal, nominal)
    4. Encode target (skill recommendation) columns
    5. Extract and encode disabilities and complications
    6. Clean the final dataframe
    
    Args:
        input_csv_file (str): Path to original CSV (default: dataset.csv)
        output_csv_file (str): Path to output CSV (default: dataset_copy.csv)
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print(f"\n{'='*70}")
    print("STARTING PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    
    # Step 1: Create working copy
    print("\n[Step 1] Creating working copy...")
    ensure_working_copy(input_csv_file, output_csv_file)
    
    # Load the working copy into memory
    df = pd.read_csv(output_csv_file)
    print(f"Loaded dataframe with shape: {df.shape}")
    
    # Step 2: Drop unnecessary columns
    print("\n[Step 2] Dropping unnecessary columns...")
    df = load_and_drop(df)
    print(f"Dataframe shape after dropping: {df.shape}")
    
    # Step 3: Encode features
    print("\n[Step 3] Encoding feature columns...")
    df = encode_features_label(df)
    print(f"Dataframe shape after feature encoding: {df.shape}")
    
    # Step 4: Encode targets
    print("\n[Step 4] Encoding target columns...")
    df = encode_target_labels(df)
    print(f"Dataframe shape after target encoding: {df.shape}")
    
    # Step 5: Extract disabilities and complications
    print("\n[Step 5] Extracting disabilities...")
    df = extract_disability(df)
    print(f"Dataframe shape after disability extraction: {df.shape}")
    
    print("\n[Step 6] Extracting complications...")
    df = extract_complications(df)
    print(f"Dataframe shape after complications extraction: {df.shape}")
    
    # Step 6: Clean the final dataframe
    print("\n[Step 7] Cleaning dataframe...")
    df = clean_dataframe(df)
    print(f"Final dataframe shape: {df.shape}")
    
    # Save the processed dataframe
    df.to_csv(output_csv_file, index=False)
    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Processed data saved to: {output_csv_file}")
    print(f"{'='*70}\n")

    return df


def main():
    """Run the main preprocessing pipeline."""
    df = wrangle()
    
    print("Preview of processed data:")
    print(df.head())
    print(f"\nDataframe info:")
    print(df.info())


if __name__ == "__main__":
    main()
