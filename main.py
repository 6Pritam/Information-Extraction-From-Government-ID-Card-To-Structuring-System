import streamlit as st
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
import re
import numpy as np
import cv2
from transformers import pipeline
import easyocr
import json
import requests
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coading = load_lottieurl("https://lottie.host/300ab164-4a40-4b1e-bcdd-e5cc31f9395c/86ZJd6mwhm.json")

# NLP model for named entity recognition
nlp = pipeline("ner", grouped_entities=True)
reader = easyocr.Reader(['en'])

def main():
    st.title("Information Extraction From Government ID Card To Structuring System")
    st.lottie(lottie_coading, key="coading")
    st.write("Upload a photo or PDF of your ID card")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        if uploaded_file.type == "application/pdf":
            handle_pdf(uploaded_file)
        else:
            handle_image(uploaded_file)

def handle_image(image_file):
    progress_bar = st.progress(0)
    image = Image.open(image_file)
    progress_bar.progress(10)
    preprocessed_image = preprocess_image(image)
    progress_bar.progress(30)
    boxed_image, text, results_table, filtered_results_table = extract_text_and_draw_boxes(preprocessed_image)
    progress_bar.progress(60)
    st.image(boxed_image, caption='Processed Image with Bounding Boxes', use_column_width=True)
    st.write("All Extracted Text:")
    st.table(results_table)
    st.write("Text with Confidence Score above 0.5:")
    st.table(filtered_results_table)
    process_filtered_text(filtered_results_table)
    progress_bar.progress(100)
    st.success('Processing Complete!')

def preprocess_image(image):
    # Convert to OpenCV format
    open_cv_image = np.array(image.convert('RGB'))
    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to preserve edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # Use adaptive thresholding to binarize the image
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Resize the image to improve OCR accuracy
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(gray)

def extract_text_and_draw_boxes(image):
    open_cv_image = np.array(image.convert('RGB'))
    results = reader.readtext(open_cv_image, detail=1)
    draw = ImageDraw.Draw(image)
    text = ""
    results_table = []
    filtered_results_table = []

    for result in results:
        bbox, text_segment, confidence = result
        text += text_segment + " "
        bbox = [tuple(map(int, point)) for point in bbox]
        draw.line([bbox[0], bbox[1], bbox[2], bbox[3], bbox[0]], fill="red", width=2)
        results_table.append({"bbox": bbox, "Text": text_segment, "Confidence": confidence})
        if confidence > 0.5:
            filtered_results_table.append({"bbox": bbox, "Text": text_segment, "Confidence": confidence})

    # Filter the bounding boxes in order
    filtered_results_table = filter_bounding_boxes(filtered_results_table)

    num_boxes = len(results_table)
    st.write(f"Number of boxes: {num_boxes}")

    return image, text, results_table, filtered_results_table

def filter_bounding_boxes(results_table, min_confidence=0.5, min_size=20):
    # Filter based on confidence and size, maintaining the order
    filtered_results = []
    for result in results_table:
        bbox, text, confidence = result['bbox'], result['Text'], result['Confidence']
        bbox_width = bbox[2][0] - bbox[0][0]
        bbox_height = bbox[2][1] - bbox[0][1]
        if confidence > min_confidence and bbox_width > min_size and bbox_height > min_size:
            filtered_results.append(result)
    return filtered_results

def handle_pdf(pdf_file):
    progress_bar = st.progress(0)
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    num_pages = doc.page_count
    results_table = []
    filtered_results_table = []

    for i, page in enumerate(doc):
        text += page.get_text()
        progress_bar.progress(int(((i + 1) / num_pages) * 60))

    if text.strip() != "":
        st.write("Extracted Text from PDF:")
        st.text(text)
        process_filtered_text(results_table)
        progress_bar.progress(100)
        st.success('Processing Complete!')
    else:
        handle_image_pdf(pdf_bytes, progress_bar, results_table, filtered_results_table)

def handle_image_pdf(pdf_bytes, progress_bar, results_table, filtered_results_table):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    num_pages = doc.page_count

    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        preprocessed_image = preprocess_image(img)
        boxed_image, page_text, page_results, page_filtered_results = extract_text_and_draw_boxes(preprocessed_image)
        text += page_text
        results_table.extend(page_results)
        filtered_results_table.extend(page_filtered_results)
        st.image(boxed_image, caption=f'Processed Image with Bounding Boxes - Page {i + 1}', use_column_width=True)
        progress_bar.progress(int(((i + 1) / num_pages) * 60) + 30)

    process_filtered_text(filtered_results_table)
    st.write("All Extracted Text:")
    st.table(results_table)
    st.write("Text with Confidence Score above 0.5:")
    st.table(filtered_results_table)
    progress_bar.progress(100)
    st.success('Processing Complete!')

def process_filtered_text(filtered_results_table):
    filtered_text = " ".join([result["Text"] for result in filtered_results_table])
    st.write(filtered_text)

    # Use NLP model to extract entities
    entities = nlp(filtered_text)

    for entity in entities:
        if entity['entity_group'] == 'PER':
            st.write(f"Name: {entity['word']}")
        elif entity['entity_group'] == 'DATE':
            st.write(f"Date of Birth: {entity['word']}")
        elif entity['entity_group'] == 'ID':
            st.write(f"Card Number: {entity['word']}")
        elif entity['entity_group'] == 'GENDER':
            st.write(f"Gender: {entity['word']}")

if __name__ == "__main__":
    main()
