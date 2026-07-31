"""Roman Hindi/Urdu to Devanagari transliteration engine."""

# Mapping tables - longest match first
CONSONANTS = [
    ('shh', 'ष'), ('sh', 'श'), ('chh', 'छ'), ('ch', 'च'),
    ('kh', 'ख'), ('gh', 'घ'), ('ng', 'ङ'), ('jh', 'झ'),
    ('ny', 'ञ'), ('th', 'थ'), ('dh', 'ध'), ('ph', 'फ'),
    ('bh', 'भ'), ('nh', 'ण'),
    ('k', 'क'), ('g', 'ग'), ('j', 'ज'), ('t', 'त'), ('d', 'द'),
    ('n', 'न'), ('p', 'प'), ('b', 'ब'), ('m', 'म'), ('y', 'य'),
    ('r', 'र'), ('l', 'ल'), ('v', 'व'), ('w', 'व'), ('s', 'स'),
    ('h', 'ह'), ('q', 'क़'), ('f', 'फ़'), ('z', 'ज़'), ('x', 'क्स'),
]

VOWELS_INDEPENDENT = [
    ('aa', 'आ'), ('ai', 'ऐ'), ('au', 'औ'), ('ee', 'ई'), ('oo', 'ऊ'),
    ('a', 'अ'), ('i', 'इ'), ('u', 'उ'), ('e', 'ए'), ('o', 'ओ'),
]

VOWELS_MATRA = [
    ('aa', 'ा'), ('ai', 'ै'), ('au', 'ौ'), ('ee', 'ी'), ('oo', 'ू'),
    ('a', ''), ('i', 'ि'), ('u', 'ु'), ('e', 'े'), ('o', 'ो'),
]

COMMON_WORDS = {
    'kya': 'क्या', 'hai': 'है', 'hain': 'हैं', 'ka': 'का', 'ki': 'की',
    'ke': 'के', 'ko': 'को', 'se': 'से', 'me': 'में', 'par': 'पर',
    'aur': 'और', 'ya': 'या', 'nahi': 'नहीं', 'nahin': 'नहीं',
    'main': 'मैं', 'mein': 'में', 'hum': 'हम', 'tum': 'तुम',
    'aap': 'आप', 'wo': 'वो', 'ye': 'ये', 'woh': 'वह', 'yeh': 'यह',
    'tha': 'था', 'thi': 'थी', 'the': 'थे', 'hoga': 'होगा',
    'hogi': 'होगी', 'haal': 'हाल', 'hello': 'हेलो', 'suno': 'सुनो',
    'bhai': 'भाई', 'yaar': 'यार', 'dost': 'दोस्त', 'log': 'लोग',
    'kaise': 'कैसे', 'kaisa': 'कैसा', 'kaisi': 'कैसी',
    'bahut': 'बहुत', 'bohot': 'बहुत', 'accha': 'अच्छा',
    'bura': 'बुरा', 'theek': 'ठीक', 'sab': 'सब',
    'lekin': 'लेकिन', 'magar': 'मगर', 'kyunki': 'क्योंकि',
    'isliye': 'इसलिए', 'phir': 'फिर', 'abhi': 'अभी',
    'kahani': 'कहानी', 'sunao': 'सुनाओ', 'batao': 'बताओ',
    'dekho': 'देखो', 'chalo': 'चलो', 'jao': 'जाओ', 'aao': 'आओ',
    'karo': 'करो', 'karna': 'करना', 'hona': 'होना', 'jana': 'जाना',
    'aana': 'आना', 'khana': 'खाना', 'peena': 'पीना',
    'ek': 'एक', 'do': 'दो', 'teen': 'तीन', 'char': 'चार',
    'paanch': 'पांच', 'duniya': 'दुनिया', 'zindagi': 'ज़िंदगी',
    'dil': 'दिल', 'pyar': 'प्यार', 'mohabbat': 'मोहब्बत',
    'sachai': 'सच्चाई', 'jhooth': 'झूठ', 'insaan': 'इंसान',
    'waqt': 'वक़्त', 'raat': 'रात', 'din': 'दिन', 'subah': 'सुबह',
    'shaam': 'शाम', 'ghar': 'घर', 'darwaza': 'दरवाज़ा',
    'subscribe': 'सब्सक्राइब', 'like': 'लाइक', 'comment': 'कमेंट',
    'video': 'वीडियो', 'channel': 'चैनल', 'share': 'शेयर',
    'dilchasp': 'दिलचस्प', 'interesting': 'दिलचस्प',
}


def _match_consonant(text, pos):
    lower = text[pos:].lower()
    for roman, dev in CONSONANTS:
        if lower.startswith(roman):
            return dev, len(roman)
    return None, 0


def _match_vowel_matra(text, pos):
    lower = text[pos:].lower()
    for roman, dev in VOWELS_MATRA:
        if lower.startswith(roman):
            return dev, len(roman)
    return None, 0


def _match_vowel_independent(text, pos):
    lower = text[pos:].lower()
    for roman, dev in VOWELS_INDEPENDENT:
        if lower.startswith(roman):
            return dev, len(roman)
    return None, 0


def transliterate_word(word):
    """Convert a single Roman Hindi word to Devanagari."""
    low = word.lower().strip()
    if low in COMMON_WORDS:
        return COMMON_WORDS[low]

    result = []
    i = 0

    while i < len(word):
        c = word[i]

        # Skip non-alpha
        if not c.isalpha():
            result.append(c)
            i += 1
            continue

        # Try consonant
        cons, clen = _match_consonant(word, i)
        if cons:
            result.append(cons)
            i += clen
            # Check for vowel matra after consonant
            if i < len(word) and word[i].isalpha():
                vm, vlen = _match_vowel_matra(word, i)
                if vm is not None:
                    if vm:  # non-empty matra (not 'a')
                        # Word-final short "i" is conventionally written with
                        # the long matra in standard Hindi orthography, even
                        # when the spoken vowel is short (Stage 1 Analysis,
                        # F-12; e.g. "buri" -> "बुरी", not "बुरि").
                        if vm == 'ि' and i + vlen == len(word):
                            vm = 'ी'
                        result.append(vm)
                    i += vlen
                    continue
                # No vowel follows. If another consonant immediately
                # follows, this is a genuine cluster and needs an explicit
                # halant to suppress the inherent schwa (Stage 1 Analysis,
                # F-12; e.g. the "sp" in "dilchasp" -> "स्प", not "सप").
                # Known limitation: this is a heuristic approximation of
                # Hindi's schwa-deletion rules, not a full solution — it can
                # over-insert a halant for a word-medial single consonant
                # whose schwa is conventionally left bare in standard
                # spelling (e.g. the "l" in "dilchasp" itself, which is why
                # that specific word is also pinned in COMMON_WORDS above).
                next_cons, _ = _match_consonant(word, i)
                if next_cons:
                    result.append('्')
            continue

        # Try independent vowel
        vi, vilen = _match_vowel_independent(word, i)
        if vi:
            result.append(vi)
            i += vilen
            continue

        # Fallback
        result.append(c)
        i += 1

    return ''.join(result)


def roman_to_devanagari(text):
    """Convert full Roman Hindi/Urdu text to Devanagari, preserving punctuation."""
    import re
    tokens = re.split(r'(\s+|[,.\-!?;:।\'\"()]+)', text)
    result = []
    for token in tokens:
        if token.strip() and any(c.isalpha() for c in token):
            # Check if it's already Devanagari
            if any('\u0900' <= c <= '\u097F' for c in token):
                result.append(token)
            else:
                result.append(transliterate_word(token))
        else:
            result.append(token)
    return ''.join(result)


if __name__ == '__main__':
    tests = [
        "Hello kya haal hai bhai",
        "Aaj hum ek bahut hi dilchasp kahani sunenge",
        "Subscribe karo aur like karo",
        "Ye duniya bahut buri hai lekin hum theek hain",
    ]
    for t in tests:
        print(f"Roman:     {t}")
        print(f"Devanagari: {roman_to_devanagari(t)}")
        print()
