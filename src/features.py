def per90(value, minutes):
    if minutes is None or minutes <= 0:
        return None
    return value * 90.0 / minutes

def add_change_features(df, names):
    df = df.copy()
    for name in names:
        pre, post = f"pre_{name}", f"post_{name}"
        if pre in df.columns and post in df.columns:
            df[f"delta_{name}"] = df[post] - df[pre]
    return df
