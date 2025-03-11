# Author: Ronen H

# Produces the jsonl data files of the King James Bible with one for
# the old testament and one for the new testament.
# Each line represents a chapter with its verses.

import json
import os
import string


def get_bible_data(kjv_text_file_dir: str):
    '''
    Produces the jsonl data file of the King James Bible
    from the kjv_text_file_dir of chapters.

    Input: Directory Path to Chapters of King James Bible.
    '''

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

    # Overwrite existing data.
    if os.path.isdir('bible_data'):
        for bible_data_file in os.listdir('bible_data'):
            os.remove('bible_data/' + bible_data_file)
        os.rmdir('bible_data')
    os.mkdir('bible_data')

    chapterid = 1

    # Produce old testament data.
    with open('bible_data/old_testament.jsonl', 'x', encoding='utf-8') as old_testament_data:
        for book in old_testament:
            with open(kjv_text_file_dir + '/' + book + '.txt') as book_data:
                book_line = book_data.readline()
                while book_line.strip() != "":
                    book_line = book_data.readline()
                while book_line.strip() == "":
                    book_line = book_data.readline()
                chapter_num = 1
                is_chapter = True
                while is_chapter:
                    chapter_dict = {'chapterid': chapterid, 'chapter': book + ' ' + str(chapter_num), 'num_verses': 0,
                                    'verses': {}}
                    verse_num = 1
                    verse = book_data.readline().strip()
                    if book == 'Psalms' and verse.split()[0] != str(verse_num):
                        verse = book_data.readline().strip()
                    while verse != "" and verse.split()[0] == str(verse_num):
                        chapter_dict['verses'][str(verse_num)] = verse.lstrip(string.digits + string.whitespace + 'Â¶')
                        chapter_dict['num_verses'] += 1
                        verse_num += 1
                        verse = book_data.readline().strip()
                        if book == 'Psalms' and chapter_num == 119 and verse != "" and verse.split()[0] != str(
                                verse_num):
                            verse = book_data.readline().strip()
                    if verse != "":
                        is_chapter = False
                    else:
                        book_line = book_data.readline()
                    old_testament_data.write(json.dumps(chapter_dict) + "\n")
                    chapter_num += 1
                    chapterid += 1

    # Produce new testament data.
    with open('bible_data/new_testament.jsonl', 'x', encoding='utf-8') as new_testament_data:
        for book in new_testament:
            with open(kjv_text_file_dir + '/' + book + '.txt') as book_data:
                book_line = book_data.readline()
                while book_line.strip() != "":
                    book_line = book_data.readline()
                while book_line.strip() == "":
                    book_line = book_data.readline()
                chapter_num = 1
                is_chapter = True
                while is_chapter:
                    chapter_dict = {'chapterid': chapterid, 'chapter': book + ' ' + str(chapter_num), 'num_verses': 0,
                                    'verses': {}}
                    verse_num = 1
                    verse = book_data.readline().strip()
                    while verse != "" and verse.split()[0] == str(verse_num):
                        chapter_dict['verses'][str(verse_num)] = verse.lstrip(string.digits + string.whitespace + 'Â¶')
                        chapter_dict['num_verses'] += 1
                        verse_num += 1
                        verse = book_data.readline().strip()
                    if verse != "":
                        is_chapter = False
                    else:
                        book_data.readline()
                    new_testament_data.write(json.dumps(chapter_dict) + "\n")
                    chapter_num += 1
                    chapterid += 1

# if __name__ == "__main__":
    # get_bible_data('kjv-text-files')
