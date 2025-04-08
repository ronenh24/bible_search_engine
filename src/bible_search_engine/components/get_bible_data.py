# Author: Ronen Huang

import json
import os
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from typing import Literal


# Old testament books.
old_testament = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua',
                 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles',
                 '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs',
                 'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel',
                 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk',
                 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi']

# New testament books
new_testament = ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians', '2 Corinthians',
                 'Galatians', 'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
                 '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter',
                 '1 John', '2 John', '3 John', 'Jude', 'Revelation']


def get_bible_data(version: Literal["csv", "esv", "kjv", "msg", "nas", "niv", "nkjv", "nlt", "nrs"]="niv") -> None:
    """
    Produces the data files of the Bible version with one for the old testament and one for the new testament.
    """
    if os.path.isdir('bible_search_engine/bible_data'):
        if os.path.isfile('bible_search_engine/bible_data/old_testament_' + version + '.jsonl'):
            os.remove('bible_search_engine/bible_data/old_testament_' + version + '.jsonl')
        if os.path.isfile('bible_search_engine/bible_data/new_testament_' + version + '.jsonl'):
            os.remove('bible_search_engine/bible_data/new_testament_' + version + '.jsonl')
    else:
        os.mkdir('bible_search_engine/bible_data')
    if not os.path.isfile('bible_search_engine/bible_data/__init__.py'):
        init_file = open('bible_search_engine/bible_data/__init__.py', 'x', encoding='utf-8')
        init_file.close()

    web_link = "https://www.biblestudytools.com"
    if version != "niv":
        web_link += "/" + version

    chapter_id = 1
    with open('bible_search_engine/bible_data/old_testament_' + version + '.jsonl',
              'x', encoding='utf-8') as old_testament_data:
        for book in tqdm(old_testament):
            book_url = web_link + "/" + book.lower().replace(" ", "-")
            book_session = requests.session()
            book_session.mount(book_url, HTTPAdapter(max_retries=10))
            book_response = book_session.get(book_url, timeout=5)
            parsed_book = BeautifulSoup(book_response.text, "html.parser")

            chapter_num = 1
            for a_tag in parsed_book.find_all("a", href=True):
                chapter_url = a_tag["href"]
                if re.fullmatch(book_url + r"/\d{1,3}\.html", chapter_url):
                    chapter_dict = {'chapterid': chapter_id, 'chapter': book + ' ' + str(chapter_num), 'num_verses': 0,
                                    'verses': {}}
                    chapter_session = requests.session()
                    chapter_session.mount(chapter_url, HTTPAdapter(max_retries=10))
                    chapter_response = chapter_session.get(chapter_url, timeout=5)
                    parsed_chapter = BeautifulSoup(chapter_response.text, "html.parser")
                    verses = parsed_chapter.find_all("div", attrs={"data-verse-id": True})
                    for verse in verses:
                        for a in verse.find_all("a", href=True):
                            a.decompose()
                        for h3 in verse.find_all("h3"):
                            h3.decompose()
                        verse_id = verse.get("data-verse-id")
                        verse_text = " ".join([s.strip() for s in verse.strings])
                        chapter_dict["num_verses"] += 1
                        chapter_dict["verses"][verse_id] = verse_text.lstrip(verse_id).lstrip()
                    old_testament_data.write(json.dumps(chapter_dict) + "\n")
                    chapter_num += 1
                    chapter_id += 1

    with open('bible_search_engine/bible_data/new_testament_' + version + '.jsonl',
              'x', encoding='utf-8') as new_testament_data:
        for book in tqdm(new_testament):
            book_url = web_link + "/" + book.lower().replace(" ", "-")
            book_session = requests.session()
            book_session.mount(book_url, HTTPAdapter(max_retries=10))
            book_response = book_session.get(book_url, timeout=5)
            parsed_book = BeautifulSoup(book_response.text, "html.parser")

            chapter_num = 1
            for a_tag in parsed_book.find_all("a", href=True):
                chapter_url = a_tag["href"]
                if re.fullmatch(book_url + r"/\d{1,3}\.html", chapter_url):
                    chapter_dict = {'chapterid': chapter_id, 'chapter': book + ' ' + str(chapter_num), 'num_verses': 0,
                                    'verses': {}}
                    chapter_session = requests.session()
                    chapter_session.mount(chapter_url, HTTPAdapter(max_retries=10))
                    chapter_response = chapter_session.get(chapter_url, timeout=5)
                    parsed_chapter = BeautifulSoup(chapter_response.text, "html.parser")
                    verses = parsed_chapter.find_all("div", attrs={"data-verse-id": True})
                    for verse in verses:
                        for a in verse.find_all("a", href=True):
                            a.decompose()
                        for h3 in verse.find_all("h3"):
                            h3.decompose()
                        verse_id = verse.get("data-verse-id")
                        verse_text = " ".join([s.strip() for s in verse.strings])
                        chapter_dict["num_verses"] += 1
                        chapter_dict["verses"][verse_id] = verse_text.lstrip(verse_id).lstrip()
                    new_testament_data.write(json.dumps(chapter_dict) + "\n")
                    chapter_num += 1
                    chapter_id += 1
