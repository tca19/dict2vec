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

def downloadCambridge(word):
    URL = "http://dictionary.cambridge.org/dictionary/english/" + word
    try:
        html = urlopen(URL).read().decode('utf-8')

        # Definitions are in a <b> tag that has the class "def"
        pattern = re.compile('<b class="def">(.*?)</b>',
                                re.IGNORECASE|re.DOTALL)
        defs = re.findall(pattern, html)
        # Need to clean definitions of <a> and <span> tags.
        cleaner = re.compile('<.+?>', re.IGNORECASE|re.DOTALL)
        return [ re.sub(cleaner, '', x) for x in defs ]

    except HTTPError:
        return -1
    except UnicodeDecodeError:
        return -1
    except Exception as e:
        print("\nERROR: * timeout error.")
        print("       * retry Cambridge -", word)
        return -1


def downloadDictionary(word):
    URL = "http://www.dictionary.com/browse/" + word
    try:
        html = urlopen(URL).read().decode('utf-8')

        # Definitions are in a big block <div class="def-list">, but
        # only the first block contains interesting definitions.
        # Can't use </div> for regex ending because there are some <div>
        # inside definitions, so i use the next big div.
        block_p = re.compile('<div class="def-list">(.*?)<div class="tail-wrapper">',
                                re.IGNORECASE|re.DOTALL)
        block_one = re.findall(block_p, html)[0]

        # Inside this block, all definitions are in <div class="def-content">
        # We stop at class="def-block" which is a sentence example.
        defs_p = re.compile('<div class="def-content">(.+?)<div class="def-block', re.IGNORECASE|re.DOTALL)
        defs = re.findall(defs_p, block_one)

        # Sometimes there are no <div class="def-clock"> after the definition, so no
        # definitions have been catched. But we can use the end of </div>
        # to catch them (ex: for the word 'wick').
        if len(defs) == 0:
            defs_p = re.compile('<div class="def-content">(.+?)</div>', re.IGNORECASE|re.DOTALL)
            defs = re.findall(defs_p, block_one)

        # Need to clean definitions of <a> and <span> tags.
        # Use .strip() to also clean some \r or \n.
        cleaner = re.compile('<.+?>', re.IGNORECASE|re.DOTALL)
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


def downloadCollins(word):
    URL = "http://www.collinsdictionary.com/dictionary/english/" + word
    try:
        html = urlopen(URL).read().decode('utf-8')

        # Definitions are in the first big block <div class="hom">.
        # Use the next <div> for ending regex.
        block_p = re.compile('<div class="hom">(.*?)<div class="div copyright">',
                                re.IGNORECASE|re.DOTALL)
        block_one = re.findall(block_p, html)[0]

        # Inside this block, all definitions are in <span class="def">...</span>.
        # Update 24/01/17 : now the fetched word is surrounded by <span></span>
        # so we can't use the </span> to get the entire definition. We need
        # to add the <div to make sure we are capturing the definition until
        # its end.
        # Update 29/11/17 : same thing but now surrounded in a <div> and no longer a <span>
        defs_p = re.compile('<div class="def">(.+?)<div', re.IGNORECASE|re.DOTALL)
        defs = re.findall(defs_p, block_one)

        # Need to clean definitions of <a> and <span> tags.
        cleaner = re.compile('<.+?>', re.IGNORECASE|re.DOTALL)
        return [ re.sub(cleaner, '', x) for x in defs ]

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


def downloadOxford(word):
    URL = "http://www.oxforddictionaries.com/definition/english/"+ word
    try:
        html = urlopen(URL).read().decode('utf-8')

        # Definitions are in a <span> tag that has a class "ind"
        pattern = re.compile('<span class="ind">(.*?)</span>',
                                re.IGNORECASE|re.DOTALL)
        defs = re.findall(pattern, html)

        # Need to clean definitions of <a> and <span> tags.
        cleaner = re.compile('<.+?>', re.IGNORECASE|re.DOTALL)
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
    "Cam": downloadCambridge,
    "Dic": downloadDictionary,
    "Col": downloadCollins,
    "Oxf": downloadOxford,
}

STOPSWORD = set()
with open('stopwords.txt') as f:
    for line in f:
        STOPSWORD.add(line.strip().lower())

def downloadCleaned(dict_name, word):
    words = []
    download = MAP_DICT[dict_name]
    res = download(word)
    if res == -1:
        res = []

    for definition in res:
        for word in definition.split():
            word = ''.join([c.lower() for c in word
                         if c.isalpha() and ord(c) < 128])
            if not word in STOPSWORD:
                words.append(word)
    return words

if __name__ == '__main__':
    print("Cambridge")
    print(downloadCleaned("Cam", 'wick'))
    print("dictionary.com")
    print(downloadCleaned("Dic", 'wick'))
    print("Collins")
    print(downloadCleaned("Col", 'change'))
    print("Oxford")
    print(downloadCleaned("Oxf", 'car'))

