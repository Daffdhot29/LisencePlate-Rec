import cv2
import re
import numpy as np 
from collections import defaultdict, deque 
from ultralytics import YOLO 
import easyocr

model = YOLO("license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

plate_pattern = re.compile(r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$')

def correct_plate_format(ocr_text) : 
    mapping_num_to_alpha = {"0" : "O", "1" : "I", "5":"S", "8":"B"}
    mapping_alpha_to_num = {"O" : "0", "I" :"1" ,"S" : "5","B":"8"}

    ocr_text = ocr_text.upper().replace(" ","")

    if len(ocr_text) != 7 : 
        return ""
    
    corrected = []
    for i, ch in enumerate(ocr_text) : 
        if i < 2 or i == 4 : #alphabet Position
            if ch.isdigit() and ch in mapping_num_to_alpha :
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha() : 
                corrected.append(ch)
            else : 
                return ""
        else : # number position 
            if ch.isdigit() and ch in mapping_alpha_to_num : 
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isalpha() : 
                corrected.append(ch)
            else : 
                return "" 
    
    return "".join(corrected)


