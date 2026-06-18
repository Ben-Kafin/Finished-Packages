# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:28:40 2025

@author: Benjamin Kafin
"""

import numpy as np
from spym.io import rhksm4
import os

def dump_sm4_file_structure(input_file, output_file, max_rows=5, max_cols=5):
    """
    Uses the spym rhksm4 module to load an SM4 file and writes a text dump of its
    internal structure to output_file.
    
    The dump includes:
      - Total number of pages.
      - For each page:
          - The list of attributes and their values,
          - The shape (dimensions) of the data,
          - A snippet (first few rows and columns) of the data array.
    
    Parameters:
      input_file  (str): Path to the SM4 file.
      output_file (str): Path to the text file to write the output.
      max_rows    (int): Maximum number of rows from each data array to print.
      max_cols    (int): Maximum number of columns from each data array to print.
    """
    if not os.path.exists(input_file):
        print(f"Error: File not found at '{input_file}'.")
        return

    try:
        # Load the SM4 file using the provided rhksm4 module function.
        f = rhksm4.load(input_file)
    except Exception as e:
        print("Error loading file:", e)
        return

    out_lines = []
    out_lines.append(f"SM4 file: {input_file}")
    out_lines.append(f"Number of pages: {len(f._pages)}")
    out_lines.append("-" * 50)

    # f._pages is assumed to be a dictionary or list of pages.
    # We iterate over sorted keys if it is a dictionary.
    if isinstance(f._pages, dict):
        page_keys = sorted(f._pages.keys())
    else:
        page_keys = range(len(f._pages))

    for key in page_keys:
        page = f._pages[key]
        # Get the shape of the data array.
        data_shape = np.shape(page.data)
        out_lines.append(f"Page {key}:")
        out_lines.append(" Attributes:")
        # Iterate over attributes:
        for attr_key, attr_value in page.attrs.items():
            out_lines.append(f"   {attr_key}: {attr_value}")
        out_lines.append(f" Data shape: {data_shape}")

        # Create a snippet of the data:
        try:
            data = np.array(page.data)
            if data.ndim == 1:
                nrows = min(max_rows, data.shape[0])
                snippet = data[:nrows]
            else:
                nrows = min(max_rows, data.shape[0])
                ncols = min(max_cols, data.shape[1])
                snippet = data[:nrows, :ncols]
            out_lines.append(" Data snippet:")
            out_lines.append(str(snippet))
        except Exception as e:
            out_lines.append(" Unable to convert data to numpy array: " + str(e))
        out_lines.append("-" * 50)

    # Write the output lines to the output file.
    try:
        with open(output_file, 'w') as f_out:
            f_out.write("\n".join(out_lines))
        print(f"Dump completed. See output file: {output_file}")
    except Exception as e:
        print("Error writing dump file:", e)


if __name__ == '__main__':
    input_file = r'C:/Users/Benjamin Kafin/Downloads/NHC-iPr_Au_base4_CT_2023_12_29_10_33_32_074.sm4'  # Replace with your SM4 file path.
    output_file = r"SM4_structure_ZigZag.txt"
    dump_sm4_file_structure(input_file, output_file)