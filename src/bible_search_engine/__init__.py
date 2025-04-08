import spacy
import subprocess

try:
    spacy.load("en_core_web_lg")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)