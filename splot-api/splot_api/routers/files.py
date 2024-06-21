from fastapi import APIRouter, UploadFile, File
from typing import List
import os
import tempfile

from internals import read_touchstone
from internals.conversions.db import v2db

# Generate a route for processing sparams
router = APIRouter()


# Process the sparam files
@router.post("/process_files", tags=["process_files"])
async def process_files(files: List[UploadFile] = File(...)):
    processed_data = {}
    for file in files:
        file_contents = await file.read()
        fname = file.filename
        fext = fname.split(".")[-1]

        # Create a temporary file without extension
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_contents)
            temp_file_path = temp_file.name

        # Rename file to include the snp extension
        final_temp_file_path = f"{temp_file.name}.{fext}"
        os.rename(temp_file_path, final_temp_file_path)

        # Read the s params, separate out complex and imag because you can't 
        # easily conver complex number to json, then store in combined dict
        _, s = read_touchstone(final_temp_file_path, xarray=True)
        s_real = s.real
        s_real_dict = s_real.to_dict()
        s_imag = s.imag
        s_imag_dict = s_imag.to_dict()
        s_dict = {
            'real': s_real_dict,
            'imag': s_imag_dict,
        }

        print(s)

        # Store the data in the final json
        processed_data[fname] = s_dict

        # Delete the temp file
        os.remove(final_temp_file_path)

    return processed_data

