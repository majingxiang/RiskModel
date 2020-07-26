from prettytable import PrettyTable

def format_for_print(df):
    """transform dataframe to pretty table"""
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return table
