import docx

def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text
