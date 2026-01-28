import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error

def knn_impute_numerics_only(df, n_neighbors=3, output_path='imputed.csv'):
    """
    Scales and applies KNN imputation only on numeric columns, keeping text columns untouched.
    After imputation, inverse scales & rounds boolean & one-hot encoded columns back to 0/1.
    Leaves other numeric columns scaled.
    """
    text_columns = [
        'birth_history_complications_at_the_time_of_birth',
        'psychological_analysis_attention_seeking_behaviour',
        'psychological_analysis_threat_in_handling_sharp_objects',
        'psychological_analysis_one_word_diagnosis',
        'speech_and_language_history_hearing_issues',
        'speech_and_language_history_wears_hearing_aid',
        'disability',
        'extracted_complications',
        #data hasn't been input yet.
        'occupational_therapy_strength',
        'occupational_therapy_manipulation',
        'occupational_therapy_coordination',
        'occupational_therapy_opposition',
        'occupational_therapy_violence_level',
        'occupational_therapy_impulsivity',
        'occupational_therapy_sexual_interest_or_lust',
        'social_and_emotional_history_unoccupied_play',
        'social_and_emotional_history_solitary_play',
        'social_and_emotional_history_spectator_behaviour',
        'social_and_emotional_history_parallel_play',
        'social_and_emotional_history_associate_play',
        'social_and_emotional_history_cooperative_play',
        "psychological_analysis_kars_score" 
    ]

    df_copy = df.copy()

    existing_text_cols = [col for col in text_columns if col in df_copy.columns]
    numeric_cols = [col for col in df_copy.columns if col not in existing_text_cols]
    numeric_df = df_copy[numeric_cols]

    # Identify boolean / one-hot columns before scaling
    bool_or_onehot_cols = [
        col for col in numeric_cols
        if set(df_copy[col].dropna().unique()).issubset({0, 1})
    ]

    # Handle text column DataFrame
    text_df = df_copy[existing_text_cols] if existing_text_cols else pd.DataFrame(index=df_copy.index)

    # Replace pandas NA with numpy nan
    numeric_df = numeric_df.replace({pd.NA: np.nan})

    # Scale numeric data
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(numeric_df)
  
    # scaled_df.to_csv('scaled.csv', index=False)

    # Apply KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled = imputer.fit_transform(scaled_numeric)
    imputed_numeric_df = pd.DataFrame(imputed_scaled, columns=numeric_cols, index=df_copy.index)

    # Inverse scale only the boolean / one-hot columns before rounding
    if bool_or_onehot_cols:
        # Extract only those columns from imputed_scaled and inverse scale them
        inverse_scaled_bool = scaler.inverse_transform(
            imputed_numeric_df
        )[:, [numeric_cols.index(c) for c in bool_or_onehot_cols]]

        # Assign back and round to 0/1
        for i, col in enumerate(bool_or_onehot_cols):
            imputed_numeric_df[col] = np.round(inverse_scaled_bool[:, i]).astype(int)

    # Merge imputed numeric data with text columns
    final_df = pd.concat([imputed_numeric_df, text_df], axis=1)

    final_df.to_csv(output_path, index=False)
    print(f"Imputed data saved to {output_path}")

    return final_df


def test_knn_imputer_accuracy(df, n_neighbors=3, missing_fraction=0.1):
    """
    Tests KNN imputation accuracy:
    - Binary columns → accuracy
    - Continuous/ordinal columns → MAE + normalized MAE
    """

    # Define text columns to exclude
    text_columns = [
        'birth_history_complications_at_the_time_of_birth',
        'psychological_analysis_attention_seeking_behaviour',
        'psychological_analysis_threat_in_handling_sharp_objects',
        'psychological_analysis_one_word_diagnosis',
        'speech_and_language_history_hearing_issues',
        'speech_and_language_history_wears_hearing_aid',
        'disability',
        'extracted_complications',
        #data hasn't been input yet.
        'occupational_therapy_strength',
        'occupational_therapy_manipulation',
        'occupational_therapy_coordination',
        'occupational_therapy_opposition',
        'occupational_therapy_violence_level',
        'occupational_therapy_impulsivity',
        'occupational_therapy_sexual_interest_or_lust',
        'social_and_emotional_history_unoccupied_play',
        'social_and_emotional_history_solitary_play',
        'social_and_emotional_history_spectator_behaviour',
        'social_and_emotional_history_parallel_play',
        'social_and_emotional_history_associate_play',
        'social_and_emotional_history_cooperative_play',
        "psychological_analysis_kars_score" 
    ]

    # Remove text columns
    numeric_cols = [col for col in df.columns if col not in text_columns]
    numeric_df = df[numeric_cols].copy()

    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    # Keep only rows with no missing values
    complete_rows = numeric_df.dropna()
    print(f"Found {complete_rows.shape[0]} complete rows for testing.")
    if complete_rows.empty:
        print("⚠️ No complete rows available to test imputation.")
        return None

    # Copy original complete data
    original_df = complete_rows.copy()

    # Randomly mask a fraction of values
    mask = np.random.rand(*original_df.shape) < missing_fraction
    df_with_missing = original_df.mask(mask)

    # Scale before KNN
    scaler = StandardScaler()
    scaled_missing = scaler.fit_transform(df_with_missing)

    # Apply KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled = imputer.fit_transform(scaled_missing)

    # Inverse scaling
    imputed_df = pd.DataFrame(
        scaler.inverse_transform(imputed_scaled),
        columns=numeric_cols,
        index=original_df.index
    )

    # Evaluation
    results = {}
    binary_accs = []
    continuous_maes = []
    continuous_nmaes = []

    for idx, col in enumerate(numeric_cols):
        col_mask = mask[:, idx]  # where we intentionally hid data
        if col_mask.sum() == 0:
            continue  # skip if no masked values for this column

        original_values = pd.to_numeric(original_df.loc[col_mask, col], errors='coerce')
        imputed_values = pd.to_numeric(imputed_df.loc[col_mask, col], errors='coerce')

        # Check if binary/one-hot
        unique_vals = set(original_df[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            acc = (imputed_values.round().astype(int) == original_values).mean()
            results[col] = {"type": "binary", "accuracy": acc}
            binary_accs.append(acc)
        else:
            mae = mean_absolute_error(original_values, imputed_values)
            col_range = original_df[col].max() - original_df[col].min()
            nmae = mae / col_range if col_range != 0 else 0
            results[col] = {"type": "continuous", "mae": mae, "nmae": nmae}
            continuous_maes.append(mae)
            continuous_nmaes.append(nmae)

    # Summary
    print("\n=== KNN Imputer Accuracy Test ===")
    if binary_accs:
        print(f"Overall Binary Accuracy: {np.mean(binary_accs):.4f}")
    if continuous_maes:
        print(f"Average Continuous MAE: {np.mean(continuous_maes):.4f}")
        print(f"Average Continuous nMAE: {np.mean(continuous_nmaes)*100:.2f}%")

    for col, metrics in results.items():
        if metrics["type"] == "binary":
            print(f"  {col}: Binary Accuracy = {metrics['accuracy']:.4f}")
        else:
            print(f"  {col}: MAE = {metrics['mae']:.4f}, nMAE = {metrics['nmae']*100:.2f}%")

    return results


def main():

    df = pd.read_csv("dataset_copy.csv")  # Change filename to your CSV

    # Call the imputer
    knn_impute_numerics_only(df)

if __name__ == "__main__":
    main()