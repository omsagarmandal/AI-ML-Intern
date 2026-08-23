import pandas as pd

def clean_data(train_df, test_df):
    missing_percent = train_df.isnull().sum() / len(train_df) * 100
    cols_to_drop = missing_percent[missing_percent > 40].index
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if train_df[col].isnull().sum() > 0:
            train_df[col] = train_df[col].fillna(train_df[col].median())

    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if train_df[col].isnull().sum() > 0:
            train_df[col] = train_df[col].fillna(train_df[col].mode()[0])

    return train_df, test_df