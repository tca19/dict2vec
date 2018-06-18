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

from urllib.error import HTTPError
import urllib.request
import re

def download_cambridge(word, pos="all"):
    URL = "http://dictionary.cambridge.org/dictionary/english/" + word

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urllib.request.urlopen(URL).read().decode('utf-8')

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
                pos_extracted = re.search(pos_pat, html[start:end])

                # some words (like mice) do not have a pos info, so no pos
                # extracted
                if pos_extracted is None:
                    continue

                pos_extracted = pos_extracted.group(1)

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

    except HTTPError:
        return -1
    except UnicodeDecodeError:
        return -1
    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry Cambridge -", word)
        return -1

def download_dictionary(word, pos="all"):
    URL = "http://www.dictionary.com/browse/" + word

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urllib.request.urlopen(URL).read().decode('utf-8')

        # definitions are in <section> tags with class "css-1sdcacc". Each POS
        # type has its own <section>, so extract them all.
        block_pat = re.compile('<section class="css-1sdcacc(.*?)</section>',
                               re.I|re.S)
        blocks = re.findall(block_pat, html)

        # inside each block, definitions are in <span> tags with the class
        # "css-9sn2pa". Sometimes there is another class, so use the un-greedy
        # regex pattern .+? to go until the closing '>' of the opening <span>
        # tag.
        defs_pat = re.compile('<span class="css-9sn2pa.+?>(.*?)</span>', re.I|re.S)

        # need to extract definitions only if it's a certain pos type
        if pos in ["adjective", "noun", "verb"]:

            # for each block, if the extracted POS matches the pos argument, add
            # the definitions in defs (.+ because class is either luna-pos or
            # pos)
            pos_pat = re.compile('class=.+pos">(.*?)</span>', re.I|re.S)
            defs = []

            for block in blocks:
                pos_extracted = re.search(pos_pat, block)

                # some words (like cia) do not have a pos info so no pos
                # extracted
                if pos_extracted is None:
                    continue

                pos_extracted = pos_extracted.group(1)

                if pos not in pos_extracted:
                    continue

                # remove possible sentence examples in definitions
                defs += [ re.sub('<span class="luna-example.+$', '', x)
                          for x in re.findall(defs_pat, block) ]

        # otherwise, concatenate all blocks and extract all definitions
        # available. Remove possible sentence examples in definitions
        else:
            defs = re.findall(defs_pat, " ".join(blocks))
            defs = [ re.sub('<span class="luna-example.+$', '', x)
                     for x in defs ]

        # need to clean definitions of <span> tags. Use cleaner to replace these
        # tags by empty string, Use .strip() to also clean some \r or \n.
        cleaner = re.compile('<.+?>', re.I|re.S)
        return [ re.sub(cleaner, '', x).strip() for x in defs ]

    except HTTPError:
        return -1
    except UnicodeDecodeError:
        return -1
    except IndexError:
        return -1
    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry dictionary.com -", word)
        return -1

def download_collins(word, pos="all"):
    URL = "https://www.collinsdictionary.com/dictionary/english/" + word

    # Collins has set some server restrictions. Need to spoof the HTTP headers
    headers = {
        'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like '
            'Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept':
            'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding':
            'none',
        'Accept-Language':
            'en-US,en;q=0.8',
        }
    req = urllib.request.Request(URL, headers=headers)

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')

        # definitions are in big blocks <div class="content definitions [...] >
        # Use the next <div> with "copyright" for ending regex. Regroup all
        # blocks.
        block_p = re.compile('<div class="content definitions.+?"(.*?)'
                             '<div class="div copyright', re.I|re.S)
        blocks = " ".join(re.findall(block_p, html))

        # inside this block, definitions are in <div class="def">...</div>
        defs_pat = re.compile('<div class="def">(.+?)</div>', re.I|re.S)

        # need to extract definitions only if it's a certain pos type
        if pos in ["adjective", "noun", "verb"]:

            # each sense of the word is inside a <div class="hom">. Get all the
            # starting and ending indexes of these blocks
            sense_pat = re.compile('<div class="hom">', re.I| re.S)
            idx = [m.start() for m in sense_pat.finditer(blocks)]
            idx.append(len(blocks))
            span = [(idx[i], idx[i+1]) for i in range(len(idx)-1)]

            # then for each sense, I only extract the definitions if it matches
            # the pos argument
            pos_pat = re.compile('class="pos">(.*?)</span>', re.I|re.S)
            defs = []

            for start, end in span:
                pos_extracted = re.search(pos_pat, blocks[start:end])
                # sometimes, sense is just a sentence or an idiom, so no pos
                # extracted
                if pos_extracted is None:
                    continue

                # noun is sometimes written as "countable noun". "verb" as
                # "verb transitive". Use the `in` trick to match these 2
                # categories with either "noun" or "verb"
                pos_extracted = pos_extracted.group(1)

                if pos not in pos_extracted:
                    continue

                defs += re.findall(defs_pat, blocks[start:end])

        # otherwise extract all definitions available
        else:
            defs = re.findall(defs_pat, blocks)

        # need to clean definitions of <a> and <span> tags. Use cleaner to
        # replace these tags by empty string, Use .strip() to also clean some
        # \r or \n, and replace because sometimes there are \n inside a sentence
        cleaner = re.compile('<.+?>', re.I|re.S)
        return [re.sub(cleaner, '', x).replace('\n', ' ').strip() for x in defs]

    except HTTPError:
        return -1
    except UnicodeDecodeError:
        return -1
    except IndexError:
        return -1
    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry Collins -", word)
        return -1

def download_oxford(word, pos="all"):
    URL = "http://en.oxforddictionaries.com/definition/"+ word

    if pos not in ["all", "adjective", "noun", "verb"]:
        pos = "all"

    try:
        html = urllib.request.urlopen(URL).read().decode('utf-8')

        # extract blocks containing POS type and definitions. For example, if
        # word is both a noun and a verb, there is one <section class="gramb">
        # block for the noun definitions, and another for the verb definitions
        block_p = re.compile('<section class="gramb">(.*?)</section>', re.I|re.S)
        blocks  = re.findall(block_p, html)

        # inside these blocks, definitions are in <span class="ind">
        defs_pat = re.compile('<span class="ind">(.*?)</span>', re.I|re.S)

        # need to extract definitions only if it's a certain pos type
        if pos in ["adjective", "noun", "verb"]:

            # for each block, I only extract the definitions if it matches the
            # pos argument
            pos_pat = re.compile('class="pos">(.*?)</span>', re.I|re.S)
            defs = []

            for block in blocks:
                pos_extracted = re.search(pos_pat, block).group(1)

                if pos_extracted != pos:
                    continue

                defs += re.findall(defs_pat, block)

        # otherwise extract all definitions available
        else:
            defs = re.findall(defs_pat, "".join(blocks))

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

def download_word_definition(dict_name, word, pos="all", clean=True):
    """
    Download the definition(s) for word from the dictionary dict_name. If clean
    is True, clean the definition before returning it (remove stopwords and non
    letters characters in definition).
    """
    words    = []
    download = MAP_DICT[dict_name]
    res      = download(word, pos=pos)
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
    #print(download_cambridge("wick", "all"))
    #print("dictionary.com")
    #print(download_dictionary("wick", "all"))
    #print("Collins")
    #print(download_collins("wick", "noun"))
    #print("\nOxford")
    #print("\n- ".join(download_oxford("wick", "adjective")))

    print("dictionary.com -- alert [ADJECTIVE]")
    print(download_dictionary("alert", "adjective"))
    print()
    print("dictionary.com -- alert [NOUN]")
    print(download_dictionary("alert", "noun"))
    print()
    print("dictionary.com -- alert [VERB]")
    print(download_dictionary("alert", "verb"))
    print()
    print("dictionary.com -- alert [ALL]")
    print(download_dictionary("alert", "all"))
    print()
