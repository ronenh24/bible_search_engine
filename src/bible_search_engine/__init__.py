import subprocess


subprocess.run(["python", "-m", "spacy", "download",
                "en_core_web_lg"], check=True)
subprocess.run(["python", "-m", "nltk.downloader",
                "wordnet"], check=True)
subprocess.run(["python", "-m", "nltk.downloader",
                "omw"], check=True)
