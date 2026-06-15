from .cleaning import normalize_column_names


def apply_schema_mapping(df, mapping=None, required=None, normalize=True):
    """Map source columns into a standard schema.

    Parameters
    ----------
    df:
        Input DataFrame.
    mapping:
        Dictionary where keys are source columns and values are standard column names.
        Matching is case-insensitive after snake_case normalization.
    required:
        Optional list of standard column names that must exist after mapping.
    normalize:
        Whether to normalize input column names before applying the mapping.
    """
    working = normalize_column_names(df) if normalize else df.copy()
    if mapping:
        normalized_mapping = {}
        for source, target in mapping.items():
            source_key = normalize_column_names(working[[source]]).columns[0] if source in working.columns else source
            normalized_mapping[source_key] = target
        working = working.rename(columns=normalized_mapping)

    missing = [col for col in (required or []) if col not in working.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return working

