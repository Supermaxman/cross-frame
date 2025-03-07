import string
from typing import Optional

import pandas as pd


def create_excel(
    data: list[dict[str, str]],
    output_path: str,
    columns: Optional[list[str]] = None,
    labels: Optional[dict] = None,
):
    if labels is None:
        labels = {}
    labels = labels.copy()
    if columns is None:
        columns = list(data[0].keys())
    column_types = {}
    for label_letter, label_name in zip(string.ascii_uppercase, labels):
        columns.append(label_name)
        labels[label_name]["label_letter"] = label_letter
    # TODO support more than A-Z columns
    for column_letter, column_name in zip(string.ascii_uppercase, columns):
        if column_name == "id" or column_name.endswith("_id"):
            column_types[column_letter] = "id"
        elif column_name in labels:
            labels[column_name]["data_letter"] = column_letter
            column_types[column_letter] = "label"
        else:
            column_types[column_letter] = "text"

    df = pd.DataFrame(data=data, columns=columns)
    df = df.fillna("")
    df = df.set_index(columns[0])
    data_size = len(df)
    l_df = pd.DataFrame({l_name: l["values"] for l_name, l in labels.items()})

    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="data")
    l_df.to_excel(writer, sheet_name="labels", index=False)
    workbook = writer.book
    worksheet = writer.sheets["data"]
    for label_name, label in labels.items():
        data_letter = label["data_letter"]
        label_letter = label["label_letter"]
        label_values = label["values"]
        label_message = label["message"]
        worksheet.data_validation(
            f"{data_letter}2:{data_letter}{data_size+1}",
            {
                "validate": "list",
                "source": f"=labels!${label_letter}$2:${label_letter}${len(label_values)+1}",
                "input_message": label_message,
            },
        )
    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    for column_letter, column_type in column_types.items():
        column_range = f"{column_letter}:{column_letter}"
        # first_col, last_col, width, cell_format, options
        if column_type == "id":
            worksheet.set_column(column_range, options={"hidden": True})
        elif column_type == "text":
            worksheet.set_column(column_range, cell_format=cell_format, width=55)
        elif column_type == "label":
            worksheet.set_column(column_range, width=15)

    writer.close()
