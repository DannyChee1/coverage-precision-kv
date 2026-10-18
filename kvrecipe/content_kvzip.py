import re

import numpy as np

STOPWORDS = frozenset('''
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its
itself they them their theirs themselves what which who whom this that these those am is are was were be been being
have has had having do does did doing a an the and but if or because as until while of at by for with about against
between into through during before after above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such no nor not only own same so than too
very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn
needn shan shouldn wasn weren won wouldn
'''.split())
MONTHS_DAYS = frozenset('''january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday'''.split())


def words(text):
    out = []
    for m in re.finditer(r'\S+', text):
        word = m.group().strip('.,;:!?()[]{}"\'')
        before = text[:m.start()].rstrip()
        out.append((m.start(), m.end(), word, not before or before[-1] in '.!?\n'))
    return out


def is_entity(word, sentence_start):
    return (any(c.isdigit() for c in word) or (word[:1].isupper() and not sentence_start)
            or word.lower() in MONTHS_DAYS)


def is_content(word, sentence_start):
    return is_entity(word, sentence_start) or (word.isalpha() and len(word) >= 3 and word.lower() not in STOPWORDS)


def token_tags(text, base, offsets):
    content, entity = np.zeros(len(offsets)), np.zeros(len(offsets))
    starts = np.array([a for a, _ in offsets])
    ends = np.array([b for _, b in offsets])
    for a, b, word, start in words(text):
        if not is_content(word, start):  # every entity word is a content word
            continue
        hit = (starts < base + b) & (ends > base + a) & (ends > starts)
        content[hit] = 1
        if is_entity(word, start):
            entity[hit] = 1
    return content, entity
