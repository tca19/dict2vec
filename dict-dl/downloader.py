#!/usr/bin/env python3
#
# Copyright (c) 2017-present, All rights reserved.
# Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
#
# This file is part of Dict2vec.
#
# Dict2vec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Dict2vec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at the root of this repository for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with Dict2vec.  If not, see <http://www.gnu.org/licenses/>.

from urllib.request import urlopen
from urllib.error import HTTPError
import re

def download_cambridge(word, pos="all"):
    URL = "http://dictionary.cambridge.org/dictionary/english/" + word

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urlopen(URL).read().decode('utf-8')

        # definitions are in a <b> tag that has the class "def"
        defs_pat = re.compile('<b class="def">(.*?)</b>', re.I|re.S)

        # need to extract definitions only if it's a certain pos type
        if pos in ["adjective", "noun", "verb"]:

            # each type entry (adj, noun or verb) is in a "entry-body__el"
            # block. A word might have many blocks (if it is both a noun and a
            # verb, it will have 2 blocks). Moreover, there are also different
            # blocks for British or American language. I can't extract blocks
            # because there is no ending regex that works for every word, so I
            # consider a block to be between the indexes of 2 consecutive
            # block_pat matches. Last block goes to the end of html string.
            block_pat = re.compile('<div class="entry-body__el ', re.I|re.S)
            idx = [m.start() for m in block_pat.finditer(html)] + [len(html)]
            span = [(idx[i], idx[i+1]) for i in range(len(idx)-1)]

            # then for each block, I only extract the definitions if it matches
            # the pos argument
            pos_pat = re.compile('class="pos".*?>(.*?)</span>', re.I|re.S)
            defs = []

            for start, end in span:
                pos_extracted = re.search(pos_pat, html[start:end]).group(1)

                if pos_extracted != pos:
                    continue

                defs += re.findall(defs_pat, html[start:end])

        # otherwise extract all definitions available
        else:
            defs = re.findall(defs_pat, html)

        # need to clean definitions of <a> and <span> tags. Use cleaner to
        # replace these tags by empty string
        cleaner = re.compile('<.+?>', re.I|re.S)
        return [ re.sub(cleaner, '', x) for x in defs ]

    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry Cambridge -", word)
        return -1

def download_dictionary(word, pos="all"):
    URL = "http://www.dictionary.com/browse/" + word

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urlopen(URL).read().decode('utf-8')

        # definitions are in a big block <div class="def-list">, but
        # only the first block contains interesting definitions.
        # Can't use </div> for regex ending because there are some <div>
        # inside definitions, so i use the next big div.
        block_p = re.compile('<div class="def-list">(.*?)<div class="tail-wrap',
                             re.I|re.S)
        block_one = re.findall(block_p, html)[0]

        # inside this block, all definitions are in <div class="def-content">.
        # Sometimes closed by a new <div>, sometimes a closing </div> so ?/
        # catches the two cases
        defs_p = re.compile('<div class="def-content">(.+?)</?div', re.I|re.S)

        # need to extract definitions only if it's a certain pos type
        if pos in ["adjective", "noun", "verb"]:

            # inside block_one, there are sub-blocks : one for each type of
            # pos (adjective/noun/verb). Extract all these sub-blocks
            section_pat = re.compile("<section(.*?)</section>", re.I|re.S)
            sections = re.findall(section_pat, block_one)

            # then for each block, I only extract the definitions if it matches
            # the pos argument
            pos_pat = re.compile('class="dbox-pg">(.*?)</span>', re.I|re.S)
            defs = []

            for section in sections:
                pos_extracted = re.search(pos_pat, section).group(1).split()[0]

                if pos_extracted != pos:
                    continue

                defs += re.findall(defs_p, section)

        # otherwise extract all definitions available
        else:
            defs = re.findall(defs_p, block_one)

        # need to clean definitions of <a> and <span> tags. Use cleaner to
        # replace these tags by empty string, Use .strip() to also clean some
        # \r or \n.
        cleaner = re.compile('<.+?>', re.I|re.S)
        return [ re.sub(cleaner, '', x).strip() for x in defs ]

    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry dictionary.com -", word)
        return -1

def download_collins(word, pos="all"):
    URL = "http://www.collinsdictionary.com/dictionary/english/" + word

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urlopen(URL).read().decode('utf-8')

        # definitions are in the big block <div class="content [...] br">.
        # Use the next <div> with "copyright" for ending regex.
        block_p = re.compile(
          '<div class="content .*? br">(.*?)<div class="div copyri', re.I|re.S)
        block_one = re.findall(block_p, html)[0]

        # inside this block, definitions are in <div class="def">...</div>
        defs_pat = re.compile('<div class="def">(.+?)</div>', re.I|re.S)

        # need to extract definitions only if it's a certain pos type
        if pos in ["adjective", "noun", "verb"]:

            # each sense of the word is inside a <div class="hom">. Get all the
            # starting and ending indexes of these blocks
            sense_pat = re.compile('<div class="hom">', re.I|re.S)
            idx = [m.start() for m in sense_pat.finditer(block_one)]
            idx.append(len(block_one))
            span = [(idx[i], idx[i+1]) for i in range(len(idx)-1)]

            # then for each sense, I only extract the definitions if it matches
            # the pos argument
            pos_pat = re.compile('class="pos">(.*?)</span>', re.I|re.S)
            defs = []

            for start, end in span:
                pos_extracted = re.search(pos_pat, block_one[start:end])

                # sometimes, sense is just a sentence or an idiom, so no pos
                # extracted
                if pos_extracted is None:
                    continue

                # noun is written as "countable noun". The split()[-1] trick is
                # to only extract "noun" from "countable noun", but leaves
                # "verb" as "verb".
                pos_extracted = pos_extracted.group(1).split()[-1]

                if pos_extracted != pos:
                    continue

                defs += re.findall(defs_pat, block_one[start:end])

        # otherwise extract all definitions available
        else:
            defs = re.findall(defs_pat, block_one)

        # need to clean definitions of <a> and <span> tags. Use cleaner to
        # replace these tags by empty string, Use .strip() to also clean some
        # \r or \n, and replace because sometimes there are \n inside a sentence
        cleaner = re.compile('<.+?>', re.I|re.S)
        return [re.sub(cleaner, '', x).replace('\n', ' ').strip() for x in defs]

    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry Collins -", word)
        return -1

def download_oxford(word):
    URL = "http://en.oxforddictionaries.com/definition/"+ word
    try:
        html = urlopen(URL).read().decode('utf-8')

        # definitions (for every senses of a word) are all inside a <section>
        # tag that has the class "gramb"
        block_p = re.compile('<section class="gramb">(.*?)</section>', re.I|re.S)
        blocks  = re.findall(block_p, html)

        # inside these <section>, definitions are in a <span class="ind">. One
        # <section> block can contain many definitions so we need to first
        # combine all <section>, then extract the <span> with only one call to
        # findall()
        pattern = re.compile('<span class="ind">(.*?)</span>', re.I|re.S)
        blocks  = ' '.join(blocks)
        defs    = re.findall(pattern, blocks)

        # need to clean definitions of <a> and <span> tags. Use cleaner to
        # replace these tags by empty string
        cleaner = re.compile('<.+?>', re.I|re.S)
        return [ re.sub(cleaner, '', x) for x in defs ]

    except HTTPError:
        return -1
    except UnicodeDecodeError:
        return -1
    except IndexError:
        return -1
    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry Oxford -", word)
        return -1

MAP_DICT = {
    "Cam": download_cambridge,
    "Dic": download_dictionary,
    "Col": download_collins,
    "Oxf": download_oxford,
}

STOPSWORD = set()
with open('stopwords.txt') as f:
    for line in f:
        STOPSWORD.add(line.strip().lower())

def download_word_definition(dict_name, word, clean=True):
    """
    Download the definition(s) for word from the dictionary dict_name. If clean
    is True, clean the definition before returning it (remove stopwords and non
    letters characters in definition).
    """
    words      = []
    download   = MAP_DICT[dict_name]
    res        = download(word)
    if res == -1: # no definition fetched
        res = []

    for definition in res: # there can be more than one definition fetched
        # if no cleaning needed, add the whole definition
        if not clean:
            words.append(definition)
            continue

        for word in definition.split():
            word = ''.join([c.lower() for c in word
                         if c.isalpha() and ord(c) < 128])
            if not word in STOPSWORD:
                words.append(word)

    return words

if __name__ == '__main__':
    #print("Cambridge")
    #print("\n- ".join(download_cambridge("jump", "verb")))
    #print("dictionary.com")
    #print("\n- ".join(download_dictionary("motor", "noun")))
    print("\nCollins")
    print(download_collins("jump", "verb"))
    #print("\nOxford")
    #print(download_word_definition("Oxf", 'wick'))


    #print("Oxford (no clean)")
    #print(download_word_definition("Oxf", "wick", False))
    #print()
    #print(download_oxford("car"))
    #print()
    #print(download_oxford("change"))
