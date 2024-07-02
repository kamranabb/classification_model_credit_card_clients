def remove_col_label(num_cols, cat_cols,label_col):

    if label_col in num_cols:
        num_cols=num_cols.remove(label_col)
    elif label_col in cat_cols:
        cat_cols=cat_cols.remove(label_col)
    return num_cols,cat_cols  



