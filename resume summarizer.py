from transformers import pipeline
import PyPDF2


# -------------------------------------------------------------
# 1. Load Summarization Model
# -------------------------------------------------------------
print("Loading model... ")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")


# -------------------------------------------------------------
# 2. Utility Functions
# -------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF resume file.

    Returns:
        str: Extracted text from all pages.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()


def summarize_resume(text: str) -> str:
    """
    Generates a concise 2–3 line summary from resume text.

    Args:
        text (str): Full text of the resume.

    Returns:
        str: Summarized resume (2–3 lines).
    """
    summary = summarizer(text, max_length=60, min_length=25, do_sample=False)
    return summary[0]['summary_text']


# -------------------------------------------------------------
# 3. Testing on 3 Sample Resumes
# -------------------------------------------------------------

#  Resume 1 (text)
resume1 = """
John Doe is a data analyst with 3 years of experience in Python, SQL, and Tableau.
He developed dashboards and automated reports for business intelligence.
He is seeking a data science role to apply analytical and programming skills.
"""
print("\n--- Resume 1 Summary ---")
print(summarize_resume(resume1))

#  Resume 2 (text)
resume2 = """
Sarah Smith is a software engineer with expertise in Java, Spring Boot, and REST APIs.
She has built scalable web services and led a team of 4 developers at TechNova.
Sarah aims to transition into AI/ML-focused backend engineering roles.
"""
print("\n--- Resume 2 Summary ---")
print(summarize_resume(resume2))

#  Resume 3 (PDF file OR text fallback)
# Example: pdf_path = r"C:\Users\Jash Ajmera\Desktop\resume_sample.pdf"
pdf_path = None  # change this to your PDF path if available
if pdf_path:
    resume3_text = extract_text_from_pdf(pdf_path)
else:
    resume3_text = """
    Raj Patel is an electrical engineer with 5 years of experience in automation and IoT systems.
    He has implemented sensor networks and data acquisition systems for smart manufacturing.
    He is passionate about combining engineering and data-driven decision making.
    """

print("\n--- Resume 3 Summary ---")
print(summarize_resume(resume3_text))
