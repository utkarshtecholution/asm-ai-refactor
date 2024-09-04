from ocr_models.google_ocr import cloud_vision_inference
import re

ocr_model = cloud_vision_inference

def text_ocr_process_single_image(img_path, type):
    result = ocr_model(img_path)
    text = ""
    ocr_confidence = 0
    for entry in result:
        if type == 'barcode':
            if entry['text'].startswith('('):
                text +=" " + entry['text']
            else:
                continue 
        else:
            text += " " + entry['text']
        ocr_confidence += entry['confidence']
    return text

def digit_exists_in_text(text):
    return any(char.isdigit() for char in text)

def extract_number(ocr_text, number_type):
    """
    Extract the specified number (e.g., LOT number or REF number) from the OCR text.
    The number should start after the specified type (e.g., 'LOT' or 'REF') and begin with a character between A-Z or 0-9.

    Parameters:
    ocr_text (str): The OCR text from which to extract the number.
    number_type (str): The type of number to extract ('LOT' or 'REF').

    Returns:
    str: The extracted number, or an empty string if no valid number is found.
    """
    # Define regex patterns for different number types
    patterns  = {
        'USE_BY': r'(\d{4}-\d{2}-\d{2})',
        'REF': r'\b(?:[A-Za-z]*\d+[A-Za-z]*)(?:-[A-Za-z]*\d+[A-Za-z]*)*\b',
        'LOT': r'\b(?:[A-Za-z]*\d+[A-Za-z]*)(?:-[A-Za-z]*\d+[A-Za-z]*)*\b'
    }

    # Validate number_type and get the corresponding pattern
    if number_type not in patterns:
        raise ValueError(f"Unsupported number type: {number_type}")

    if number_type == 'REF' or number_type == 'LOT':
        final_text = ""
        ocr_text = ocr_text.split(' ')
        print(f" SPLIT {ocr_text}")
        ## To be changed later
        if len(ocr_text) > 1:
            for word in ocr_text:
                if digit_exists_in_text(word):
                    word = word.replace(' ', '')
                    final_text += " " + word 

            return final_text 
    
        else:
            ocr_text = ocr_text[0]
            ocr_text = ocr_text.replace(' ', '')
            return ocr_text
        
    elif number_type == 'USE_BY':

        pattern = patterns[number_type]
        
        # Compile the pattern and search for matches
        number_pattern = re.compile(pattern, re.IGNORECASE)
        match = number_pattern.search(ocr_text)

        if match:
            # Return the extracted number
            return match.group(0)
        else:
            # Return an empty string if no valid number is found
            return ''

def extract_and_format_date(barcode_text):
    # Extract the date part between (17) and (10)
    date_match = re.search(r'\(17\)(\d{6})', barcode_text)
    
    if date_match:
        # Extract the date part
        date_str = date_match.group(1)
        # Format the date part to YYYY-MM-DD
        formatted_date = f'20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}'
        return formatted_date
    else:
        return ""

def remove_unwanted_text(type, result):
    if type == 'lot_no':
        result = result.lower()
        result = result.replace('lot', '')
        result = result.replace('number', '')
        result = result.replace('catalogue', '')
        result = result.replace('catalog', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace(':', " ")
        result = result.upper()
        # result = self.filter_raw_text(result, 'LOT')
        result = extract_number(result, 'LOT')

    elif type == 'ref_no':
        result = result.lower()
        result = result.replace('reference', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace(':', '')
        result = result.replace('number', '')
        result = result.replace('ref', '')
        result = result.replace('rep', '')
        result = result.replace('catalogue' ,'')
        result = result.replace('catalog', '')
        result = result.replace(':', " ")
        result = result.upper()
        result = extract_number(result, 'REF')
        # result = self.filter_raw_text(result, 'REF')
        
    elif type == 'use_by':
        result = extract_number(result, 'USE_BY')
    
    return result