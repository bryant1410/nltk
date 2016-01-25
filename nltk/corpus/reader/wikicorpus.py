# -*- coding: utf-8 -*-
# Natural Language Toolkit: Wikicorpus Corpus Reader
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Santiago Castro <bryant@montevideo.com.uy>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""Corpus reader for Wikicorpus."""

# TODO: tag correctly the failing lines in Catalan
# TODO: find documents by key
# TODO: search documents by regex or similar
# TODO: relate the token sense with a WordNet synset

from __future__ import print_function, unicode_literals
import re
from nltk.corpus.reader import concat, CorpusReader, StreamBackedCorpusView


class Document:
    def __init__(self, _id, title, db_index):
        self.id = _id
        self.title = title
        self.db_index = db_index
        self.sents = []

    def text(self):
        return '. '.join(' '.join(token for token, _, _, _ in sent) for sent in self.sents)

    def __repr__(self):
        return "{type}(\"{title}\", \"{text}\")".format(
                type=type(self).__name__,
                title=self.title,
                text=self.text()
        )


class WikicorpusCorpusReader(CorpusReader):
    def docs(self, lang='eng'):
        fileids = [fileid for fileid in self._fileids if fileid.startswith(lang)]
        return concat([WikicorpusCorpusView(lang, fileid, encoding)
                       for (fileid, encoding) in self.abspaths(fileids, True)])

    def langs(self):
        return sorted({fileid[:3] for fileid in self._fileids})

    def raw(self, lang='eng'):
        for doc in self.docs(lang):
            yield doc.text()

    def sents(self, lang='eng'):
        for doc in self.docs(lang):
            for sent in doc.sents:
                yield [word for word, _, _, _ in sent]

    def tagged_sents(self, lang='eng'):
        for doc in self.docs(lang):
            for sent in doc.sents:
                yield [(word, tag) for word, _, tag, _ in sent]

    def tagged_words(self, lang='eng'):
        for sent in self.tagged_sents(lang):
            for pair in sent:
                yield pair

    def words(self, lang='eng'):
        for sent in self.sents(lang):
            for word in sent:
                yield word


# The XML is plain and not well formed, so it is better to read it manually.
# It doesn't have an xml declaration and the tags and attributes content is not escaped.
PATTERN_DOC_START = re.compile(
        r'<doc id="(\d+)" title="(.*)" nonfiltered="\d+" processed="\d+" dbindex="(\d+)">',
        re.UNICODE)
PATTERN_WORD = re.compile(r'(\S+)\s+(\S+)\s+(\S+)\s+(\d+)', re.UNICODE)
PATTERN_WORD_DOUBLE = re.compile(r'(\S+\s+\S+)\s+(\S+\s+\S+)\s+(\S+)\s+(\d+)', re.UNICODE)
PATTERN_WORD_TRIPLE = re.compile(r'(\S+\s+\S+\s+\S+)\s+(\S+\s+\S+\s+\S+)\s+(\S+)\s+(\d+)', re.UNICODE)

FAILING_LINE_WORD = '30_de_gener_de_1994'
FAILING_LINE_EXTRA_WORD = 'després'
FAILING_LINE_LEMMA = '[??:30/1/-99999:??.??:??]'
FAILING_LINE_TAG = 'W'
FAILING_LINE_SENSE = '0'
FAILING_LINE = '{word} {extra_word} {lemma} {tag} {sense}'.format(
    word=FAILING_LINE_WORD,
    extra_word=FAILING_LINE_EXTRA_WORD,
    lemma=FAILING_LINE_LEMMA,
    tag=FAILING_LINE_TAG,
    sense=FAILING_LINE_SENSE,
)

FAILING_LINE_2 = '20 000_tones WG_tm:20 Zu 0'

FAILING_LINE_3 = 'f'

LINE_END_OF_ARTICLE = 'ENDOFARTICLE endofarticle NP00000 0'
LINE_DOC_TAG_CLOSE = '</doc>'


def read_clean_line(stream):
    line = stream.readline()
    if line == '':  # EOF
        return None
    else:
        return line.strip().replace(chr(160), " ")


class WikicorpusCorpusView(StreamBackedCorpusView):
    def __init__(self, lang, fileid, encoding):
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)
        self.lang = lang

    def read_block(self, stream):
        line = read_clean_line(stream)

        match = PATTERN_DOC_START.match(line)
        if match is None:
            if line == '':
                # This is due to an error in a file in which two consecutive closing tags are present
                line = read_clean_line(stream)
                assert line == LINE_DOC_TAG_CLOSE, "Expected a closing body tag"

                return []
            else:
                raise ValueError("Expected an opening document tag")

        _id, title, db_index = match.groups()

        doc = Document(_id, title, db_index)

        line = read_clean_line(stream)
        while line is not None and line != LINE_END_OF_ARTICLE and line != LINE_DOC_TAG_CLOSE:
            sent = []

            while line is not None and line != '' and line != LINE_END_OF_ARTICLE and line != LINE_DOC_TAG_CLOSE:
                if line == 'Fz 0':
                    word = ''
                    lemma = ''
                    tag = 'Fz'
                    sense = 0
                else:
                    match = PATTERN_WORD.match(line)
                    if match is None:
                        match = PATTERN_WORD_DOUBLE.match(line)
                        if match is None:
                            match = PATTERN_WORD_TRIPLE.match(line)
                            if match is None:
                                if line != FAILING_LINE and line != FAILING_LINE_2 and line != FAILING_LINE_3:
                                    next_line = read_clean_line(stream)
                                    if next_line is None:
                                        # It could be due to the file being truncated,
                                        # so this line could not be parsed and there is no following line.
                                        # So, we just ignore
                                        break
                                    else:
                                        raise ValueError("Could not parse the following non-ending line: {}".format(
                                                line))

                    if line == FAILING_LINE:
                        word = FAILING_LINE_WORD
                        lemma = FAILING_LINE_LEMMA
                        tag = FAILING_LINE_TAG
                        sense = FAILING_LINE_SENSE
                    elif line == FAILING_LINE_2:
                        word = '20000_tones'
                        lemma = 'WG_tm:20000'
                        tag = 'Zu'
                        sense = '0'
                    elif line == FAILING_LINE_3:
                        break
                    else:
                        word, lemma, tag, sense = match.groups()

                sent.append((word, lemma, tag, sense))

                if line == FAILING_LINE:
                    sent.append((FAILING_LINE_EXTRA_WORD, None, None, None))

                line = read_clean_line(stream)

            if line == FAILING_LINE_3:
                break

            doc.sents.append(sent)

            if line == '':
                line = read_clean_line(stream)
            elif line is None:
                break

        if line == LINE_END_OF_ARTICLE:
            line = read_clean_line(stream)
            assert line == '. . Fp 0', "Expected a dot after the end of article"

            line = read_clean_line(stream)
            assert line == '', "Expected a blank line"

            line = read_clean_line(stream)
            assert line == LINE_DOC_TAG_CLOSE, "Expected a closing body tag"

        return [doc]
