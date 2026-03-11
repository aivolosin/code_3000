import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = [c for c in anon_df.columns if c != 'anon_id']

    merged = anon_df.merge(aux_df, on=quasi_identifiers, how='left')

    counts = merged.groupby('anon_id')['name'].count().reset_index()
    unique_ids = counts[counts['name'] == 1]['anon_id']

    unique_matches = merged[merged['anon_id'].isin(unique_ids)][['anon_id', 'name']]

    return unique_matches


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    
    if len(anon_df) == 0:
        return 0.0
    
    return len(matches_df) / len(anon_df)
