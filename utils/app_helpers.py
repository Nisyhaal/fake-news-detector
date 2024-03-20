import os
import sys

from fastapi import HTTPException

sys.path.append(os.getcwd())
import config


def validate_authentication(credentials):
    """
    Validates the authentication credentials.

    Parameters:
    - credentials (str): The authentication bearer token.

    Raises:
    - HTTPException: If the provided credentials do not match the expected bearer token,
      raises a 401 Unauthorized HTTPException with the appropriate error detail.
    """
    if credentials != config.BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication bearer token")


def validate_description_classifier(classifier):
    """
    Validates the provided description classifier.

    Parameters:
    - classifier: The description classifier object.

    Raises:
    - HTTPException: If the provided classifier is None, raises a 500 Internal Server Error
      HTTPException with the appropriate error detail indicating a failure to load the classifier.
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Failed to load description classifier")


def validate_description(text):
    """
    Validates the input text to ensure it is a string.

    Parameters:
    - text: The input text to be validated.

    Raises:
    - HTTPException: If the input is not a string, raises a 400 Bad Request HTTPException
      with the appropriate error detail.
    """
    try:
        if not isinstance(text, str):
            raise Exception("Input must be a string")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def process_result(result, label_mapping):
    """
    Extracts the label with the highest score from the given output data
    and maps it back using the provided label mapping.

    Parameters:
    - output_data (list): Nested list of dictionaries containing 'label' and 'score'.
    - label_mapping (dict): Dictionary to map labels to corresponding values.

    Returns:
    dict: Dictionary containing the extracted label and its score, rounded to four decimal points.
    """

    flattened_output_data = [item for sublist in result for item in sublist]

    max_score_label = max(flattened_output_data, key=lambda x: x['score'])

    output = {
        "label": list(label_mapping.keys())[list(label_mapping.values()).index(int(max_score_label['label'][-1]))],
        "score": round(max_score_label['score'], 4)
    }

    return output
