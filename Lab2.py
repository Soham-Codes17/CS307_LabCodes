"""
CS367 - Artificial Intelligence Lab
Lab Assignment: Plagiarism Detection using A* Search Algorithm
ADD your pdf files in the same directory as this file
add the names of your files in the last of this code.
"""

import fitz  # PyMuPDF
import re
import csv
import heapq
from nltk.tokenize import sent_tokenize
from nltk import download

# Ensure NLTK punkt tokenizer is available
download('punkt', quiet=True)


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts plain text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):
    """Cleans and tokenizes text into sentences."""
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text.lower())
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences


# -----------------------------
# LEVENSHTEIN (EDIT DISTANCE)
# -----------------------------
def levenshtein_distance(s1, s2):
    """Computes the edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # deletion
                           dp[i][j - 1] + 1,  # insertion
                           dp[i - 1][j - 1] + cost)  # substitution
    return dp[m][n]


# -----------------------------
# A* SEARCH FOR TEXT ALIGNMENT
# -----------------------------
class Node:
    def __init__(self, i, j, cost, heuristic, path):
        self.i = i
        self.j = j
        self.cost = cost
        self.heuristic = heuristic
        self.path = path

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


def heuristic_estimate(i, j, len1, len2):
    """Simple heuristic: remaining sentences to align."""
    return abs((len1 - i) - (len2 - j))


def a_star_text_alignment(sentences1, sentences2):
    """A* search to align two sets of sentences."""
    len1, len2 = len(sentences1), len(sentences2)
    if len1 == 0 or len2 == 0:
        return []  # No alignment possible

    start = Node(0, 0, 0, heuristic_estimate(0, 0, len1, len2), [])
    frontier = [start]
    visited = set()

    while frontier:
        current = heapq.heappop(frontier)
        i, j = current.i, current.j

        if (i, j) in visited:
            continue
        visited.add((i, j))

        if i == len1 and j == len2:
            return current.path

        # Align current sentences
        if i < len1 and j < len2:
            cost = levenshtein_distance(sentences1[i], sentences2[j])
            new_path = current.path + [(sentences1[i], sentences2[j], cost)]
            heapq.heappush(frontier, Node(i + 1, j + 1, current.cost + cost,
                                          heuristic_estimate(i + 1, j + 1, len1, len2),
                                          new_path))

        # Skip sentence in doc1
        if i < len1:
            heapq.heappush(frontier, Node(i + 1, j, current.cost + 1,
                                          heuristic_estimate(i + 1, j, len1, len2),
                                          current.path))
        # Skip sentence in doc2
        if j < len2:
            heapq.heappush(frontier, Node(i, j + 1, current.cost + 1,
                                          heuristic_estimate(i, j + 1, len1, len2),
                                          current.path))
    return []


# -----------------------------
# MAIN DETECTION FUNCTION
# -----------------------------
def detect_plagiarism_from_pdfs(pdf1_path, pdf2_path, output_csv="plagiarism_report.csv", threshold=10):
    print("🔍 Extracting and processing text...")
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)

    if not text1:
        print(f"⚠️ Warning: No extractable text found in {pdf1_path}.")
        return
    if not text2:
        print(f"⚠️ Warning: No extractable text found in {pdf2_path}.")
        return

    sentences1 = preprocess_text(text1)
    sentences2 = preprocess_text(text2)

    if len(sentences1) == 0 or len(sentences2) == 0:
        print("⚠️ No valid sentences extracted from one or both documents. Cannot run alignment.")
        return

    print(f"📘 Document 1: {len(sentences1)} sentences")
    print(f"📙 Document 2: {len(sentences2)} sentences")

    print("🤖 Running A* plagiarism detection...")
    alignments = a_star_text_alignment(sentences1, sentences2)

    if not alignments:
        print("⚠️ Alignment returned empty. No plagiarism detected.")
        return

    print("🧾 Generating report...")
    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sentence 1", "Sentence 2", "Edit Distance", "Plagiarism Likelihood"])

        for s1, s2, cost in alignments:
            likelihood = "High" if cost <= threshold else "Low"
            writer.writerow([s1, s2, cost, likelihood])

    print(f"✅ Report saved to {output_csv}")
    print("🎯 Detection Complete!")


# -----------------------------
# RUN EXAMPLE
# -----------------------------

#add the names of your pdf files here
if __name__ == "__main__":
    pdf1 = "Tutorial_Questions.pdf"  # Replace with your PDF file
    pdf2 = "202507_lab_manual.pdf"  # Replace with your PDF file
    detect_plagiarism_from_pdfs(pdf1, pdf2)
