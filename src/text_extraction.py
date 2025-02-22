from typing import Tuple, List

import layoutparser as lp
from PIL import Image
from pytesseract import pytesseract
from difflib import SequenceMatcher


layout_model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config')


def extract_text_from_image(image: Image) -> Tuple[list[str], list[str]]:
    text_blocks = []
    page_width = image.width
    layout = layout_model.detect(image)

    for block in layout:
        cropped_image = image.crop(block.block.coordinates)
        text = pytesseract.image_to_string(cropped_image).strip()
        if text:
            text_blocks.append((block.block.coordinates, text))

    # Sort blocks by vertical position
    sorted_blocks = sorted(text_blocks, key=lambda x: x[0][3])  # x[0][3] accesses y_2

    # Split into left and right columns based on page width
    left_column = []
    right_column = []

    for block in sorted_blocks:

        if block[0][0] < (page_width // 4) and block[0][2] > (page_width // 1.6):
            left_column.append(block)
            continue

        elif block[0][0] < (page_width // 1.6) and block[0][2] < (page_width // 1.6):
            left_column.append(block)
            continue

        elif block[0][0] > (page_width // 2.5) and block[0][2] > (page_width // 1.6):
            right_column.append(block)

    left_column_text = [block[1] for block in left_column]
    right_column_text = [block[1] for block in right_column if block[1] not in left_column_text]

    return left_column_text, right_column_text


def remove_duplicates(lst: List[str], sequence_matcher: bool = False) -> List[str]:
    def helper(lst):
        if len(lst) <= 1:
            return lst

        # Start from the last element
        last_idx = len(lst) - 1

        # Compare with each previous element
        for compare_idx in range(last_idx - 1, -1, -1):
            if sequence_matcher:
                similarity = SequenceMatcher(None, lst[last_idx].lower(), lst[compare_idx].lower()).ratio()
                if similarity > 0.8:
                    lst.pop(last_idx)
                    return helper(lst)
            else:
                if lst[last_idx] == lst[compare_idx]:
                    lst.pop(last_idx)
                    return helper(lst)

        # If we get here, no matches were found for the last element
        # Process the rest of the list
        if len(lst) > 1:
            last_item = lst.pop()
            lst = helper(lst)
            lst.append(last_item)

        return lst

    return helper(lst.copy())


def clean_text(lst_text: List[str]) -> str:
    text_with_no_duplicates = remove_duplicates(lst_text)
    clean_text_1 = remove_duplicates(text_with_no_duplicates, sequence_matcher=True)

    return '\n'.join(clean_text_1)


def extract_text_from_pdf_image(pdf_page_image: Image) -> List[str]:
    text_1, text_2 = extract_text_from_image(pdf_page_image)
    clean_text_1 = clean_text(text_1)
    clean_text_2 = clean_text(text_2)

    return [clean_text_1, clean_text_2]
