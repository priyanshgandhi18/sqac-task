<h1 style="color:#1E90FF;">🌟 SQAC TASKS EXPLANATION AND WORKING</h1>
<b>Author:</b> Priyansh Gandhi  

This repository contains <b>2 SQAC tasks</b> in the domains of <b>Machine Learning</b> and <b>GenAI</b>.  

---

<h2 style="color:#FF6347;">➤ TASK 1: Mood Classification (Machine Learning)</h2>

A Python script that classifies text messages into different moods (`happy`, `sad`, `neutral`, `stressed`) using <b>Logistic Regression</b> and <b>Multinomial Naive Bayes</b>.  

<h3 style="color:#32CD32;">✅ Features</h3>
<ul>
<li>Classifies short text messages into multiple moods.</li>
<li>Uses <b>TF-IDF vectorization</b> to convert text into numerical features.</li>
<li>Trains and evaluates <b>Logistic Regression</b> and <b>Naive Bayes</b> models.</li>
<li>Prints <b>accuracy</b> and <b>detailed classification metrics</b>.</li>
</ul>

<h3 style="color:#FFD700;">🛠 Installation</h3>
<pre>
git clone https://github.com/yourusername/mood-classification.git
cd mood-classification
pip install pandas scikit-learn
</pre>

<h3 style="color:#FFD700;">▶ Usage</h3>
<pre>
python mood_classifier.py
</pre>

<h3 style="color:#8A2BE2;">⚙ How It Works</h3>
<ol>
<li><b>Dataset Creation:</b> Text messages are labeled with moods.</li>
<li><b>TF-IDF Vectorization:</b> Converts text into numerical features.</li>
<li><b>Train-Test Split:</b> 75% training, 25% testing.</li>
<li><b>Model Training:</b> Logistic Regression & Naive Bayes.</li>
<li><b>Prediction & Evaluation:</b> Accuracy & classification report.</li>
</ol>

<h3 style="color:#FF69B4;">📊 Example Output</h3>
<pre>
--- Logistic Regression Model ---
Accuracy: 0.83
Classification Report:
              precision    recall  f1-score   support
     happy       1.00      1.00      1.00        1
       sad       1.00      1.00      1.00        1
   neutral       0.50      1.00      0.67        1
   stressed      1.00      0.50      0.67        2
</pre>

---

<h2 style="color:#FF6347;">➤ TASK 2: Resume Summarizer (GenAI)</h2>

A Python script that reads resumes (PDF or text) and generates a <b>2–3 line summary</b> using a pre-trained Hugging Face model.  

<h3 style="color:#32CD32;">✅ Features</h3>
<ul>
<li>Summarizes resumes in <b>2–3 lines</b>.</li>
<li>Handles <b>text files</b> and <b>PDF resumes</b>.</li>
<li>Uses <b>BART model</b> (<i>facebook/bart-large-cnn</i>) with PyTorch backend.</li>
<li>Can be extended for multiple resumes or CSV output.</li>
</ul>

<h3 style="color:#FFD700;">🛠 Installation</h3>
<pre>
git clone https://github.com/yourusername/resume-summarizer.git
cd resume-summarizer
py -3.10 -m venv resume_env
resume_env\Scripts\activate
pip install torch transformers PyPDF2
</pre>

<h3 style="color:#FFD700;">▶ Usage</h3>

<b>Using Text Resumes:</b>
<pre>
from resume_summarizer import summarize_resume

resume_text = """
John Doe is a data analyst with 3 years of experience in Python, SQL, and Tableau.
He developed dashboards and automated reports for business intelligence.
He is seeking a data science role to apply analytical and programming skills.
"""
summary = summarize_resume(resume_text)
print(summary)
</pre>

<b>Using PDF Resumes:</b>
<pre>
from resume_summarizer import extract_text_from_pdf, summarize_resume

pdf_path = r"C:\Users\YourName\Documents\resume.pdf"
text = extract_text_from_pdf(pdf_path)
summary = summarize_resume(text)
print(summary)
</pre>

<h3 style="color:#8A2BE2;">⚙ How It Works</h3>
<ol>
<li><b>Load Model:</b> Loads BART summarization model.</li>
<li><b>Extract Text:</b> PDFs read using PyPDF2; plain text used directly.</li>
<li><b>Generate Summary:</b> Produces concise 2–3 line summary.</li>
<li><b>Print Output:</b> Summaries printed for each resume.</li>
</ol>

<h3 style="color:#FF69B4;">📊 Example Output</h3>
<pre>
--- Resume 1 Summary ---
John Doe is a data analyst with 3 years of experience in Python, SQL, and Tableau.
Created automated dashboards for business intelligence and reporting.
Skilled in applying analytical and programming skills to solve business problems.

--- Resume 2 Summary ---
Sarah Smith is a software engineer specializing in Java, Spring Boot, and REST APIs.
Built scalable web services and led a team of 4 developers.
Seeking AI/ML-focused backend engineering roles.

--- Resume 3 Summary ---
Raj Patel is an electrical engineer with 5 years of experience in automation and IoT systems.
Implemented sensor networks and data acquisition systems for smart manufacturing.
Passionate about combining engineering expertise with data-driven decision making.
</pre>

<h3 style="color:#32CD32;">Dependencies</h3>
<ul>
<li>Python 3.10+</li>
<li>torch</li>
<li>transformers</li>
<li>PyPDF2</li>
</ul>

Install via:
<pre>
pip install torch transformers PyPDF2
</pre>

---

✅ <b>Author:</b> Priyansh Gandhi  
📂 This repository demonstrates **Machine Learning** and **GenAI** tasks with real examples, ready to run.
