import pytesseract
from PIL import Image


# Come back to this. Not working bc dont have tesseract installed in container. 


def text_in_image(image_path, search_text):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return search_text.lower() in extracted_text.lower()

def text_in_image(image_path, search_text):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return search_text.lower() in extracted_text.lower()

# Example
if text_in_image("/home/csutter/cron/data/NYSDOT_1d4cuonek3o/20220120/Taconic_State_Parkway_South_of_Exit_38__Southbound__NYSDOT_1d4cuonek3o_2022-01-20-14:30:54.jpg", "no live camera"):
    print("Found!")
else:
    print("Not found!")

if text_in_image("/home/csutter/cron/data/NYSDOT_1s24mzqvgkk/20220123/I_87_SB_MP_15.3_Gov._Mario_M._Cuomo_Bridge__Southbound__NYSDOT_1s24mzqvgkk_2022-01-23-00:17:53.jpg", "no live camera"):
    print("Found!")
else:
    print("Not found!")


# if text_in_image("example.png", "unavailable"):
#     print("Found!")
# else:
#     print("Not found!")

# /home/csutter/cron/data/NYSDOT_1d4cuonek3o/20220120/Taconic_State_Parkway_South_of_Exit_38__Southbound__NYSDOT_1d4cuonek3o_2022-01-20-14:30:54.jpg

# /home/csutter/cron/data/NYSDOT_1s24mzqvgkk/20220123/I_87_SB_MP_15.3_Gov._Mario_M._Cuomo_Bridge__Southbound__NYSDOT_1s24mzqvgkk_2022-01-23-00:17:53.jpg