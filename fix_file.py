# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:24:36 2022

@author: Eduardo Sanglard
"""

from bs4 import BeautifulSoup

file_path = r".\SW\Bands\1"
cleaned_file_path = r".\Cleaned Text\Bands\1.txt"

file_text = ''

with open(file_path, 'r') as fopen:
    
    file_text = fopen.read()
    
    print(file_text)
    
# Remove All HTML Content
soup = BeautifulSoup(file_text, 'html.parser')
file_text = soup.get_text()
file_text = file_text.lower()


with open(cleaned_file_path, 'w') as nFile:
    
    nFile.write(file_text)
    