"""Query-aware context compression (a port of trim-1).

Given a context (a tool result, a file, a log, a long chat) and the current
ask, keep the lines the ask needs and drop the rest. Kept lines come back
verbatim, in their original order, with one short marker where lines were
removed. Nothing is rewritten or summarized.

The selection is made by a small embedded gradient-boosted tree model
(712 trees, 1.2 MB, deterministic, no network, no GPU) over lexical,
structural and positional features. It is not a language model: it spends
no model tokens, it just scores lines. That is why it belongs in CETI as
the compression technique — it shrinks what the oracle has to read.

Original: https://github.com/caveman-ai/trim-1 (Apache-2.0, see
LICENSE-trim-1). This file is a faithful Python port of trim.go, model.go,
features.go and bm25.go, stdlib only. The model file trim_model.bin is
vendored from the same repository. The port keeps the Go code's exact
behaviour: byte-length string sizes, float32 feature rounding, the
float32-threshold adjustment, and the same fail-closed rules. The Go
project's parity fixture (tests/testdata_trim/parity.bin) and golden output
(tests/testdata_trim/golden.txt) are ported in tests/test_trim.py.
"""

import math
import os
import re
import struct

# ---- limits (same as the Go original) -------------------------------------

MAX_CONTEXT_BYTES = 8 << 20
MAX_CONTEXT_LINES = 200_000
MAX_QUERY_BYTES = 8 << 10

# ---- features --------------------------------------------------------------

FEATURE_NAMES = [
    "bm25", "bm25_norm", "bm25_rank", "bm25_win_sum", "cover_win_sum",
    "bm25_tail_norm",
    "cover", "cover_idf", "cover_n", "cover_maxidf", "cover_rank",
    "cover_tail",
    "compound_hits", "compound_maxlen", "bigram_hits", "trigram_hits",
    "idf_mean", "idf_max", "uniq_frac",
    "k_error", "k_test", "k_path", "k_pathline", "k_heading", "k_diff",
    "k_section", "k_signature", "k_jsonkey", "k_separator", "k_frame",
    "k_kv", "k_bullet",
    "dup", "tmpl", "tmpl_first",
    "pos_rel", "pos_head", "pos_tail", "n_lines", "ctx_chars",
    "len_chars", "len_words", "alnum_frac", "upper_frac", "digit_frac",
    "indent", "indent_delta", "block_start", "header_dist", "header_bm25",
    "header_cover",
    "nb_bm25_max", "nb_bm25_mean", "nb_cover_max", "nb_compound_max",
    "prev_bm25", "next_bm25", "prev_blank", "next_blank",
    "q_words", "q_terms", "q_found", "ctx_bm25_max",
    "para_cover_max", "sec_cover", "prf_cover", "prf_compound",
]
N_FEATURES = len(FEATURE_NAMES)

(F_BM25, F_BM25_NORM, F_BM25_RANK, F_BM25_WIN_SUM, F_COVER_WIN_SUM,
 F_BM25_TAIL_NORM, F_COVER, F_COVER_IDF, F_COVER_N, F_COVER_MAXIDF,
 F_COVER_RANK, F_COVER_TAIL, F_COMPOUND_HITS, F_COMPOUND_MAXLEN,
 F_BIGRAM_HITS, F_TRIGRAM_HITS, F_IDF_MEAN, F_IDF_MAX, F_UNIQ_FRAC,
 F_K_ERROR, F_K_TEST, F_K_PATH, F_K_PATHLINE, F_K_HEADING, F_K_DIFF,
 F_K_SECTION, F_K_SIGNATURE, F_K_JSONKEY, F_K_SEPARATOR, F_K_FRAME,
 F_K_KV, F_K_BULLET, F_DUP, F_TMPL, F_TMPL_FIRST, F_POS_REL, F_POS_HEAD,
 F_POS_TAIL, F_N_LINES, F_CTX_CHARS, F_LEN_CHARS, F_LEN_WORDS,
 F_ALNUM_FRAC, F_UPPER_FRAC, F_DIGIT_FRAC, F_INDENT, F_INDENT_DELTA,
 F_BLOCK_START, F_HEADER_DIST, F_HEADER_BM25, F_HEADER_COVER,
 F_NB_BM25_MAX, F_NB_BM25_MEAN, F_NB_COVER_MAX, F_NB_COMPOUND_MAX,
 F_PREV_BM25, F_NEXT_BM25, F_PREV_BLANK, F_NEXT_BLANK, F_Q_WORDS,
 F_Q_TERMS, F_Q_FOUND, F_CTX_BM25_MAX, F_PARA_COVER_MAX, F_SEC_COVER,
 F_PRF_COVER, F_PRF_COMPOUND) = range(N_FEATURES)

# ---- small helpers ---------------------------------------------------------


def _blen(s: str) -> int:
    """Byte length, as Go's len() on a string."""
    return len(s.encode("utf-8"))


def _f32(x: float) -> float:
    """Round to float32, as Go's float32() conversion."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _f32_nextafter_down(x: float) -> float:
    """Next float32 value toward -inf (Go's math.Nextafter32)."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if bits & 0x80000000:
        bits += 1  # negative: magnitude grows
    else:
        bits -= 1  # positive or +0: value shrinks
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _is_letter(c: str) -> bool:
    return c.isalpha()


def _is_digit(c: str) -> bool:
    return c.isdecimal()  # Unicode Nd, as Go's unicode.IsDigit


def _is_upper(c: str) -> bool:
    return c.isupper()


def _is_lower(c: str) -> bool:
    return c.islower()


def _is_space(c: str) -> bool:
    return c.isspace()


def _ratio(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return a / b


def _max_of(values) -> float:
    m = 0.0
    for v in values:
        if v > m:
            m = v
    return m


# ---- model -----------------------------------------------------------------


class _Node:
    __slots__ = ("threshold", "feature", "left", "right")

    def __init__(self, threshold, feature, left, right):
        self.threshold = threshold
        self.feature = feature
        self.left = left
        self.right = right


class _Forest:
    def __init__(self, roots, nodes, leaf):
        self.roots = roots
        self.nodes = nodes
        self.leaf = leaf

    def raw(self, row) -> float:
        total = 0.0
        for i in self.roots:
            while i >= 0:
                nd = self.nodes[i]
                if row[nd.feature] <= nd.threshold:
                    i = nd.left
                else:
                    i = nd.right
            total += self.leaf[~i]
        return total

    def probability(self, row) -> float:
        return 1.0 / (1.0 + math.exp(-self.raw(row)))


_MODEL = None
_MODEL_ERROR = None
_MODEL_BYTES = None


def _model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "trim_model.bin")


def _parse_model(data: bytes) -> _Forest:
    """Parse model.bin. Layout (little endian), as documented in model.go:
    "KEEP", version uint32 (1), features/trees/nodes/leaves uint32; then
    each feature name as uint16 length + bytes; tree roots []int32; node
    feature []int32, threshold []float64, left []int32, right []int32;
    leaf values []float64. A child or root below zero is the leaf ^index.
    A node sends a row left when row[feature] <= threshold."""
    if len(data) < 24:
        raise ValueError("trim model: truncated header")
    magic, version, n_feat, n_trees, n_nodes, n_leaves = struct.unpack(
        "<4s5I", data[:24])
    if magic != b"KEEP" or version != 1:
        raise ValueError("trim model: bad magic or version")
    if (n_feat != N_FEATURES or n_trees == 0 or n_nodes > 1 << 24
            or n_leaves > 1 << 24):
        raise ValueError(
            "trim model: %d features, %d trees, %d nodes, %d leaves "
            "(module computes %d features)"
            % (n_feat, n_trees, n_nodes, n_leaves, N_FEATURES))
    # Check the implied size before allocating anything it sizes.
    size = 24 + 4 * n_trees + 20 * n_nodes + 8 * n_leaves
    for name in FEATURE_NAMES:
        size += 2 + len(name.encode("utf-8"))
    if size != len(data):
        raise ValueError("trim model: header implies %d bytes, model has %d"
                         % (size, len(data)))
    off = 24
    for i, want in enumerate(FEATURE_NAMES):
        (ln,) = struct.unpack_from("<H", data, off)
        off += 2
        got = data[off:off + ln].decode("utf-8")
        off += ln
        if got != want:
            raise ValueError("trim model feature %d is %r, module has %r"
                             % (i, got, want))
    roots = list(struct.unpack_from("<%di" % n_trees, data, off))
    off += 4 * n_trees
    feature = list(struct.unpack_from("<%di" % n_nodes, data, off))
    off += 4 * n_nodes
    threshold = list(struct.unpack_from("<%dd" % n_nodes, data, off))
    off += 8 * n_nodes
    left = list(struct.unpack_from("<%di" % n_nodes, data, off))
    off += 4 * n_nodes
    right = list(struct.unpack_from("<%di" % n_nodes, data, off))
    off += 4 * n_nodes
    leaf = list(struct.unpack_from("<%dd" % n_leaves, data, off))
    off += 8 * n_leaves
    if off != len(data):
        raise ValueError("trim model: %d trailing bytes" % (len(data) - off))

    def _valid(c):
        return (0 <= c < n_nodes) or (c < 0 and ~c < n_leaves)

    for c in roots:
        if not _valid(c):
            raise ValueError("trim model: root out of range")
    nodes = []
    for i in range(n_nodes):
        if (feature[i] < 0 or feature[i] >= N_FEATURES
                or not _valid(left[i]) or not _valid(right[i])
                or (left[i] >= 0 and left[i] <= i)
                or (right[i] >= 0 and right[i] <= i)
                or math.isnan(threshold[i])):
            raise ValueError("trim model: node %d out of range" % i)
        # Threshold is the largest float32 not above the float64 threshold,
        # so the float32 comparison decides exactly as the float64 one does.
        t = _f32(threshold[i])
        if t > threshold[i]:
            t = _f32_nextafter_down(t)
        nodes.append(_Node(t, feature[i], left[i], right[i]))
    return _Forest(roots, nodes, leaf)


def _model():
    global _MODEL, _MODEL_ERROR, _MODEL_BYTES
    if _MODEL is None and _MODEL_ERROR is None:
        try:
            with open(_model_path(), "rb") as fh:
                _MODEL_BYTES = fh.read()
            _MODEL = _parse_model(_MODEL_BYTES)
        except Exception as exc:  # noqa: BLE001 - surfaced as a load error
            _MODEL_ERROR = exc
    if _MODEL_ERROR is not None:
        raise _MODEL_ERROR
    return _MODEL


def model_info() -> str:
    """'712 trees, N nodes, 67 features, M bytes' — for logs and reports."""
    m = _model()
    return "%d trees, %d nodes, %d features, %d bytes" % (
        len(m.roots), len(m.nodes), N_FEATURES, len(_MODEL_BYTES))

# ---- BM25 (bm25.go) ---------------------------------------------------------

_BM25_K1 = 1.5
_BM25_B = 0.75

_BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}


def _bm25_term_freq(text: str) -> dict:
    freq: dict = {}
    current: list = []

    def flush():
        tok = "".join(current)
        if _blen(tok) >= 2 and tok not in _BM25_STOPWORDS:
            freq[tok] = freq.get(tok, 0) + 1
        current.clear()

    for ch in text.lower():
        if _is_letter(ch) or _is_digit(ch) or ch == "_":
            current.append(ch)
        else:
            flush()
    flush()
    return freq


def _bm25(query: str, docs: list) -> list:
    """BM25 scores, one per doc. Unnormalized; comparable only in-slice.
    Term order is fixed (sorted) because float addition is not associative."""
    query_terms = _bm25_term_freq(query)
    out = [0.0] * len(docs)
    if not query_terms or not docs:
        return out
    tfs = []
    doc_len = []
    df: dict = {}
    total_len = 0
    for doc in docs:
        tf = _bm25_term_freq(doc)
        tfs.append(tf)
        dl = sum(tf.values())
        doc_len.append(dl)
        total_len += dl
        for term in tf:
            df[term] = df.get(term, 0) + 1
    avgdl = total_len / len(docs) if total_len > 0 else 1.0
    for term in sorted(query_terms):
        for i in range(len(docs)):
            tf = float(tfs[i].get(term, 0))
            if tf == 0:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5)
                           / (df[term] + 0.5))
            denom = tf + _BM25_K1 * (1 - _BM25_B
                                    + _BM25_B * doc_len[i] / avgdl)
            out[i] += idf * (tf * (_BM25_K1 + 1)) / denom
    return out


# ---- tokenize / identifiers (features.go) -----------------------------------

_STOPWORDS = _BM25_STOPWORDS


def _may_have_parts(word: str) -> bool:
    digits = others = False
    for ch in word:
        o = ord(ch)
        if o >= 0x80 or ch == "_" or "A" <= ch <= "Z":
            return True
        if "0" <= ch <= "9":
            digits = True
        else:
            others = True
    return digits and others


def _split_ident(word: str) -> list:
    """Cut an identifier at underscores, lower-to-upper changes, the last
    capital of a capital run before a lowercase letter, and letter/digit
    changes; lowercase the parts."""
    parts = []
    start = 0

    def cut(end):
        nonlocal start
        if end > start:
            parts.append(word[start:end].lower())
        start = end

    for i in range(len(word)):
        ch = word[i]
        if ch == "_":
            cut(i)
            start = i + 1
            continue
        if i == start or i == 0:
            continue
        prev, cur = word[i - 1], ch
        if _is_lower(prev) and _is_upper(cur):
            cut(i)
        elif (_is_upper(prev) and _is_upper(cur) and i + 1 < len(word)
                and _is_lower(word[i + 1])):
            cut(i)
        elif _is_digit(prev) != _is_digit(cur) and prev != "_":
            cut(i)
    cut(len(word))
    return parts


def _tokenize(s: str):
    """Lowercased words (runs of letters/digits/underscore, 2+ bytes,
    stopwords dropped) and the words plus their identifier parts."""
    words: list = []
    terms: list = []
    start = -1

    def flush(end):
        nonlocal start
        if start < 0:
            return
        raw = s[start:end]
        start = -1
        word = raw.lower()
        if _blen(word) < 2 or word in _STOPWORDS:
            return
        words.append(word)
        terms.append(word)
        if not _may_have_parts(raw):
            return
        parts = _split_ident(raw)
        if len(parts) > 1:
            for p in parts:
                if _blen(p) >= 2 and p not in _STOPWORDS:
                    terms.append(p)

    for i, ch in enumerate(s):
        if _is_letter(ch) or _is_digit(ch) or ch == "_":
            if start < 0:
                start = i
        else:
            flush(i)
    flush(len(s))
    return words, terms


def _dedupe(items: list) -> list:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ngrams(words: list):
    bi: dict = {}
    tri: dict = {}
    for j in range(len(words) - 1):
        bi[words[j] + " " + words[j + 1]] = True
        if j + 2 < len(words):
            tri[words[j] + " " + words[j + 1] + " " + words[j + 2]] = True
    return bi, tri


_COMPOUND_SPLIT = re.compile(r"[\s\"'`()\[\]{}<>,;=|]+")
_MAX_COMPOUNDS = 64


def _compound_terms(query: str) -> list:
    """The query's identifiers that carry structure (dotted names, paths,
    snake_case, camelCase, flags), lowercased, first 64 distinct."""
    out = []
    seen = set()
    for fld in _COMPOUND_SPLIT.split(query):
        if not fld:
            continue
        fld = fld.strip(".:!?*#")
        if _blen(fld) < 4 or fld in seen:
            continue
        structured = any(c in "._/:-\\" for c in fld)
        if not structured:
            for i in range(1, len(fld)):
                if "a" <= fld[i - 1] <= "z" and "A" <= fld[i] <= "Z":
                    structured = True
                    break
        if not structured:
            continue
        seen.add(fld)
        out.append(fld.lower())
        if len(out) == _MAX_COMPOUNDS:
            break
    return out


def _is_hex_byte(c: int) -> bool:
    return (48 <= c <= 57) or (97 <= c <= 102) or (65 <= c <= 70)


def _lower_ascii_byte(c: int) -> int:
    if 65 <= c <= 90:
        return c + 32
    return c


def _template(s: str) -> bytes:
    """Mask digit runs and long hex runs so repeated log lines differing
    only in numbers, ids or timestamps share one key. Byte-wise, as Go."""
    b = s.encode("utf-8")
    out = bytearray()
    i = 0
    while i < len(b):
        c = b[i]
        if 48 <= c <= 57:  # digit
            while i < len(b) and (_is_hex_byte(b[i])
                                  or b[i] in (ord("."), ord(":"))):
                i += 1
            out.append(ord("0"))
            continue
        if _is_hex_byte(c):
            j = i
            while j < len(b) and _is_hex_byte(b[j]):
                j += 1
            if j - i >= 8:
                out.append(ord("0"))
                i = j
                continue
        out.append(_lower_ascii_byte(c))
        i += 1
    return bytes(out)


def _indent_of(line: str) -> int:
    n = 0
    for ch in line:
        if ch == " ":
            n += 1
        elif ch == "\t":
            n += 4
        else:
            return n
    return n


def _non_blank_neighbours(blank: list):
    prev = [-1] * len(blank)
    nxt = [-1] * len(blank)
    last = -1
    for i, b in enumerate(blank):
        prev[i] = last
        if not b:
            last = i
    last = -1
    for i in range(len(blank) - 1, -1, -1):
        nxt[i] = last
        if not blank[i]:
            last = i
    return prev, nxt


def _block_headers(indent: list, blank: list) -> list:
    """Each non-blank line's nearest earlier non-blank line with a smaller
    indent (its enclosing block's header), or -1."""
    out = [-1] * len(indent)
    stack: list = []
    for i in range(len(indent)):
        if blank[i]:
            continue
        while stack and indent[stack[-1]] >= indent[i]:
            stack.pop()
        if stack:
            out[i] = stack[-1]
        stack.append(i)
    return out


def _char_mix(s: str):
    alnum = upper = digit = total = 0.0
    for ch in s:
        if _is_space(ch):
            continue
        total += 1
        if _is_letter(ch) or _is_digit(ch):
            alnum += 1
        if _is_upper(ch):
            upper += 1
        if _is_digit(ch):
            digit += 1
    return alnum, upper, digit, total


_ERROR_WORDS = ["error", "exception", "traceback", "fatal", "panic", "fail",
                "assert", "warning", "denied", "refused", "not found",
                "cannot", "unable to", "invalid", "segmentation",
                "undefined", "unexpected", "errno", "critical"]
_TEST_WORDS = [" passed", " failed", " skipped", "xfail", "test session",
               "test summary"]
_TEST_PREFIXES = ["ok ", "fail", "pass", "=== run", "--- pass", "--- fail",
                  "ran "]
_SIG_WORDS = ["def ", "async def ", "class ", "func ", "function ", "fn ",
              "pub ", "struct ", "interface ", "type ", "impl ", "public ",
              "private ", "protected ", "static ", "export ", "module ",
              "package ", "import ", "from ", "#include", "const ", "var ",
              "let ", "@"]


def _contains_any(s: str, words) -> bool:
    return any(w in s for w in words)


def _has_any_prefix(s: str, prefixes) -> bool:
    return any(s.startswith(p) for p in prefixes)


def _strip_line_number(s: str) -> str:
    i = 0
    while i < len(s) and "0" <= s[i] <= "9":
        i += 1
    if i == 0 or i == len(s) or s[i] not in " \t:|":
        return s
    return s[i + 1:].strip()


def _has_path(s: str) -> bool:
    if "/" not in s and "\\" not in s:
        return False
    for fld in s.split():
        fld = fld.strip("\"'`(),;:[]")
        slash = max(fld.rfind("/"), fld.rfind("\\"))
        if (slash >= 0 and slash < len(fld) - 1
                and "." in fld[slash:]
                and any(_is_letter(c) for c in fld)):
            return True
    return False


def _has_path_line(s: str) -> bool:
    for i in range(1, len(s) - 1):
        if s[i] != ":" or not ("0" <= s[i + 1] <= "9"):
            continue
        j = i
        while j > 0 and i - j < 6 and (
                ("a" <= s[j - 1] <= "z") or ("A" <= s[j - 1] <= "Z")):
            j -= 1
        if j < i and j >= 2 and s[j - 1] == "." and s[j - 2] not in ". ":
            return True
    return False


def _is_section(body: str, trimmed: str) -> bool:
    if ((body.startswith("[") and body.endswith("]"))
            or body.startswith("==>")
            or body.startswith("file:")):
        return True
    if (_blen(body) < 80 and body.endswith(":")
            and not any(c in "(){}=" for c in body)):
        return True
    letters = uppers = 0
    for ch in trimmed:
        if _is_letter(ch):
            letters += 1
            if _is_upper(ch):
                uppers += 1
    return letters >= 4 and letters == uppers and _blen(trimmed) < 60


def _is_key_value(body: str) -> bool:
    i = 0
    while i < len(body) and (body[i] in "_.-"
                             or ("a" <= body[i] <= "z")
                             or ("0" <= body[i] <= "9")):
        i += 1
    if i == 0:
        return False
    while i < len(body) and body[i] == " ":
        i += 1
    return (i + 1 < len(body) and body[i] in ":=" and body[i + 1] != "=")


def _numbered(body: str) -> bool:
    i = 0
    while i < len(body) and "0" <= body[i] <= "9":
        i += 1
    return (i > 0 and i + 1 < len(body) and body[i] in ".)"
            and body[i + 1] == " ")


def _line_kinds(trimmed: str, lower: str) -> list:
    """The k_* flags, F_K_ERROR through F_K_BULLET."""
    k = [0.0] * 13

    def _set(idx, ok):
        if ok:
            k[idx - F_K_ERROR] = 1.0

    if trimmed == "":
        return k
    body = _strip_line_number(lower)
    _set(F_K_ERROR, _contains_any(lower, _ERROR_WORDS))
    _set(F_K_TEST, _contains_any(body, _TEST_WORDS)
         or _has_any_prefix(body, _TEST_PREFIXES))
    _set(F_K_PATH, _has_path(trimmed))
    _set(F_K_PATHLINE, _has_path_line(trimmed))
    _set(F_K_HEADING, body.startswith("#")
         and body.lstrip("#").startswith(" "))
    _set(F_K_DIFF, body.startswith("diff --git")
         or body.startswith("@@") or body.startswith("+++ ")
         or body.startswith("--- a/") or body.startswith("index "))
    _set(F_K_SECTION, _is_section(body, trimmed))
    _set(F_K_SIGNATURE, _has_any_prefix(body, _SIG_WORDS))
    _set(F_K_JSONKEY, body.startswith('"') and '":' in body)
    _set(F_K_SEPARATOR, _blen(body) >= 3 and not any(
        _is_letter(c) or _is_digit(c) for c in body))
    _set(F_K_FRAME, ('file "' in lower and ", line " in lower)
         or (body.startswith("at ") and "(" in body)
         or body.startswith("goroutine "))
    _set(F_K_KV, _is_key_value(body))
    _set(F_K_BULLET, _has_any_prefix(body, ["- ", "* ", "• ", "+ "])
         or _numbered(body))
    return k

# ---- feature extraction (features.go) ---------------------------------------

def _query_tail(s: str) -> str:
    """The query's last three non-empty lines: in an agent turn, the intent
    and the tool call that the context answers."""
    lines = s.strip().split("\n")
    tail = []
    for i in range(len(lines) - 1, -1, -1):
        t = lines[i].strip()
        if t:
            tail.append(t)
            if len(tail) == 3:
                break
    tail.reverse()
    return "\n".join(tail)


def _truncate_query(query: str) -> str:
    """Only the query's last 8 KiB count (byte cut, past a partial rune)."""
    b = query.encode("utf-8")
    if len(b) > MAX_QUERY_BYTES:
        b = b[-MAX_QUERY_BYTES:]
        query = b.decode("utf-8", errors="ignore")
    return query


def _rank_frac(values: list) -> list:
    """Each value's share of values strictly greater than it."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = [0.0] * len(values)
    greater = 0
    for k, i in enumerate(order):
        if k > 0 and values[i] < values[order[k - 1]]:
            greater = k
        out[i] = greater / max(1, len(values))
    return out


def line_features(context: str, query: str):
    """Split context at "\\n" and return its lines with one 67-float
    feature row per line."""
    query = _truncate_query(query)
    lines = context.split("\n")
    n = len(lines)
    rows = [[0.0] * N_FEATURES for _ in range(n)]

    bm = _bm25(query, lines)
    tail = _query_tail(query)
    bm_tail = bm if tail == query.strip() else _bm25(tail, lines)
    bm_max = _max_of(bm)
    tail_max = _max_of(bm_tail)

    # Term index: each line's distinct terms and how many lines hold each.
    ids: dict = {}
    df: list = []
    line_terms: list = []
    line_words: list = []
    for line in lines:
        words, terms = _tokenize(line)
        line_words.append(words)
        term_ids = []
        for t in terms:
            if t not in ids:
                ids[t] = len(df)
                df.append(0)
            term_ids.append(ids[t])
        term_ids = sorted(set(term_ids))
        for tid in term_ids:
            df[tid] += 1
        line_terms.append(term_ids)
    idf = [math.log(1 + (n - d + 0.5) / (d + 0.5)) for d in df]

    q_words, q_terms = _tokenize(query)
    q_terms = _dedupe(q_terms)
    in_query = [False] * len(df)
    q_idf_sum = 0.0
    found = 0
    for t in q_terms:
        if t in ids:
            tid = ids[t]
            in_query[tid] = True
            q_idf_sum += idf[tid]
            found += 1
    _, tail_terms = _tokenize(tail)
    tail_terms = _dedupe(tail_terms)
    in_tail = [False] * len(df)
    tail_idf_sum = 0.0
    for t in tail_terms:
        if t in ids:
            tid = ids[t]
            in_tail[tid] = True
            tail_idf_sum += idf[tid]
    bigrams, trigrams = _ngrams(q_words)
    compounds = _compound_terms(query)

    blank = [line.strip() == "" for line in lines]
    indent = [_indent_of(line) for line in lines]
    tmpl_keys = []
    dup_count: dict = {}
    tmpl_count: dict = {}
    for line in lines:
        trimmed = line.strip()
        dup_count[trimmed] = dup_count.get(trimmed, 0) + 1
        tk = _template(trimmed)
        tmpl_keys.append(tk)
        tmpl_count[tk] = tmpl_count.get(tk, 0) + 1
    prev_nb, next_nb = _non_blank_neighbours(blank)
    headers = _block_headers(indent, blank)
    tmpl_seen: set = set()
    cover = [0.0] * n
    compound = [0.0] * n

    for i, line in enumerate(lines):
        r = rows[i]
        trimmed = line.strip()
        lower = trimmed.lower()
        term_ids = line_terms[i]
        cov = cov_idf = cov_max = idf_sum = idf_max = uniq = tail_idf = 0.0
        for tid in term_ids:
            if in_tail[tid]:
                tail_idf += idf[tid]
            idf_sum += idf[tid]
            if idf[tid] > idf_max:
                idf_max = idf[tid]
            if df[tid] == 1:
                uniq += 1
            if in_query[tid]:
                cov += 1
                cov_idf += idf[tid]
                if idf[tid] > cov_max:
                    cov_max = idf[tid]
        hits = 0.0
        longest = 0.0
        for c in compounds:
            if c in lower:
                hits += 1
                if _blen(c) > longest:
                    longest = float(_blen(c))
        bi = tri = 0.0
        w = line_words[i]
        for j in range(len(w) - 1):
            if (w[j] + " " + w[j + 1]) in bigrams:
                bi += 1
            if j + 2 < len(w) and (w[j] + " " + w[j + 1] + " "
                                   + w[j + 2]) in trigrams:
                tri += 1
        cover[i] = _ratio(cov_idf, q_idf_sum)
        compound[i] = hits

        r[F_BM25] = _f32(bm[i])
        r[F_BM25_NORM] = _f32(_ratio(bm[i], bm_max))
        r[F_COVER_TAIL] = _f32(_ratio(tail_idf, tail_idf_sum))
        r[F_BM25_TAIL_NORM] = _f32(_ratio(bm_tail[i], tail_max))
        r[F_COVER] = _f32(_ratio(cov, float(found)))
        r[F_COVER_IDF] = _f32(cover[i])
        r[F_COVER_N] = _f32(cov)
        r[F_COVER_MAXIDF] = _f32(cov_max)
        r[F_COMPOUND_HITS] = _f32(hits)
        r[F_COMPOUND_MAXLEN] = _f32(math.log1p(longest))
        r[F_BIGRAM_HITS] = _f32(bi)
        r[F_TRIGRAM_HITS] = _f32(tri)
        if term_ids:
            r[F_IDF_MEAN] = _f32(idf_sum / len(term_ids))
            r[F_UNIQ_FRAC] = _f32(uniq / len(term_ids))
        r[F_IDF_MAX] = _f32(idf_max)
        kinds = _line_kinds(trimmed, lower)
        r[F_K_ERROR:F_K_BULLET + 1] = [_f32(x) for x in kinds]
        r[F_DUP] = _f32(math.log1p(dup_count[trimmed] - 1))
        r[F_TMPL] = _f32(math.log1p(tmpl_count[tmpl_keys[i]] - 1))
        if tmpl_keys[i] not in tmpl_seen:
            tmpl_seen.add(tmpl_keys[i])
            r[F_TMPL_FIRST] = 1.0
        if n > 1:
            r[F_POS_REL] = _f32(i / (n - 1))
        r[F_POS_HEAD] = _f32(math.log1p(i))
        r[F_POS_TAIL] = _f32(math.log1p(n - 1 - i))
        r[F_N_LINES] = _f32(math.log1p(n))
        r[F_CTX_CHARS] = _f32(math.log1p(_blen(context)))
        r[F_LEN_CHARS] = _f32(math.log1p(_blen(line)))
        r[F_LEN_WORDS] = _f32(math.log1p(len(line.split())))
        alnum, upper, digit, total = _char_mix(trimmed)
        if total > 0:
            r[F_ALNUM_FRAC] = _f32(alnum / total)
            r[F_UPPER_FRAC] = _f32(upper / total)
            r[F_DIGIT_FRAC] = _f32(digit / total)
        r[F_INDENT] = _f32(math.log1p(indent[i]))
        if prev_nb[i] >= 0 and not blank[i]:
            r[F_INDENT_DELTA] = _f32(float(indent[i] - indent[prev_nb[i]]))
        if (next_nb[i] >= 0 and not blank[i]
                and indent[next_nb[i]] > indent[i]):
            r[F_BLOCK_START] = 1.0
        r[F_Q_WORDS] = _f32(math.log1p(len(q_words)))
        r[F_Q_TERMS] = _f32(math.log1p(len(q_terms)))
        r[F_Q_FOUND] = _f32(_ratio(float(found), float(len(q_terms))))
        r[F_CTX_BM25_MAX] = _f32(bm_max)

    # Paragraphs (runs of non-blank lines), the nearest earlier heading, and
    # feedback terms: rare terms of the three lines that cover the query best.
    para_max = [0.0] * n
    i = 0
    while i < n:
        j = i
        best = 0.0
        while j < n and not blank[j]:
            if cover[j] > best:
                best = cover[j]
            j += 1
        for k in range(i, j):
            para_max[k] = best
        i = max(j, i + 1)
    by_cover = sorted(range(n), key=lambda x: cover[x], reverse=True)
    feedback = [False] * len(df)
    fb_sum = 0.0
    for i in by_cover[:min(3, n)]:
        if cover[i] == 0:
            break
        for tid in line_terms[i]:
            if (not in_query[tid] and not feedback[tid]
                    and df[tid] <= max(2, n // 50)):
                feedback[tid] = True
                fb_sum += idf[tid]
    # Structured identifiers the best lines name and the query does not.
    in_compounds = set(compounds)
    hops: list = []
    for i in by_cover[:min(3, n)]:
        if cover[i] == 0:
            break
        for c in _compound_terms(lines[i]):
            if c not in in_compounds and len(hops) < _MAX_COMPOUNDS:
                in_compounds.add(c)
                hops.append(c)
    heading = -1
    for i in range(n):
        r = rows[i]
        r[F_PARA_COVER_MAX] = _f32(para_max[i])
        if heading >= 0:
            r[F_SEC_COVER] = _f32(cover[heading])
        if rows[i][F_K_HEADING] == 1:
            heading = i
        fb = 0.0
        for tid in line_terms[i]:
            if feedback[tid]:
                fb += idf[tid]
        r[F_PRF_COVER] = _f32(_ratio(fb, fb_sum))
        if hops:
            low = lines[i].lower()
            r[F_PRF_COMPOUND] = _f32(float(sum(1 for c in hops if c in low)))

    bm_rank = _rank_frac(bm)
    cov_rank = _rank_frac(cover)
    for i in range(n):
        r = rows[i]
        r[F_BM25_RANK] = _f32(bm_rank[i])
        r[F_COVER_RANK] = _f32(cov_rank[i])
        if headers[i] >= 0:
            r[F_HEADER_DIST] = _f32(math.log1p(i - headers[i]))
            r[F_HEADER_BM25] = _f32(_ratio(bm[headers[i]], bm_max))
            r[F_HEADER_COVER] = _f32(cover[headers[i]])
        win_bm = win_cov = 0.0
        for j in range(max(0, i - 2), min(n - 1, i + 2) + 1):
            win_bm += _ratio(bm[j], bm_max)
            win_cov += cover[j]
        r[F_BM25_WIN_SUM] = _f32(win_bm)
        r[F_COVER_WIN_SUM] = _f32(win_cov)
        nb_max = nb_sum = cov_nb = comp_nb = 0.0
        cnt = 0
        for j in range(max(0, i - 3), min(n - 1, i + 3) + 1):
            if j == i:
                continue
            v = _ratio(bm[j], bm_max)
            if v > nb_max:
                nb_max = v
            nb_sum += v
            if cover[j] > cov_nb:
                cov_nb = cover[j]
            cnt += 1
        for j in range(max(0, i - 5), min(n - 1, i + 5) + 1):
            if j != i and compound[j] > comp_nb:
                comp_nb = compound[j]
        r[F_NB_BM25_MAX] = _f32(nb_max)
        if cnt > 0:
            r[F_NB_BM25_MEAN] = _f32(nb_sum / cnt)
        r[F_NB_COVER_MAX] = _f32(cov_nb)
        r[F_NB_COMPOUND_MAX] = _f32(comp_nb)
        if i > 0:
            r[F_PREV_BM25] = _f32(_ratio(bm[i - 1], bm_max))
            if blank[i - 1]:
                r[F_PREV_BLANK] = 1.0
        if i + 1 < n:
            r[F_NEXT_BM25] = _f32(_ratio(bm[i + 1], bm_max))
            if blank[i + 1]:
                r[F_NEXT_BLANK] = 1.0
    return lines, rows

# ---- policy / selection / compress (trim.go) --------------------------------

_MARKER_LINE = re.compile(r"^[ \t]*\u2026 \d+ lines? \u2026[ \t\r]*$",
                          re.MULTILINE)
_SECTION_PREFIXES = ("# ", "## ", "### ", "#### ", "[start of ", "==> ",
                     "diff --git ", "File: ")


class Policy:
    """Turns line scores into a selection."""

    def __init__(self, floor=0.008, top_k=3, context=1, headers=1, gap=2,
                 min_chars=2000, min_cut=0.1):
        # Floor keeps every line scoring at least this share of the context's
        # best line. Relative, because calibrated probabilities shrink as a
        # context grows: one absolute threshold cuts long contexts too hard.
        self.floor = floor
        # TopK always keeps the K best-scoring lines.
        self.top_k = top_k
        # Context keeps this many lines on each side of a kept line.
        self.context = context
        # Headers keeps up to this many enclosing block headers of every kept
        # line and, when above zero, the line opening its section.
        self.headers = headers
        # Gap fills a run of at most this many removed lines between kept
        # lines; a marker would cost about as much as the lines.
        self.gap = gap
        # MinChars returns contexts shorter than this unchanged.
        self.min_chars = min_chars
        # MinCut returns the context unchanged when the selection would
        # remove less than this share of its bytes.
        self.min_cut = min_cut

    def __repr__(self):
        return ("Policy(floor=%r, top_k=%r, context=%r, headers=%r, "
                "gap=%r, min_chars=%r, min_cut=%r)"
                % (self.floor, self.top_k, self.context, self.headers,
                   self.gap, self.min_chars, self.min_cut))

    def apply(self, lines, scores):
        """Select lines by score and render them: kept lines verbatim, one
        marker line per removed run. Returns (output, kept count)."""
        lines = list(lines)
        scores = list(scores)
        final = len(lines) > 1 and lines[-1] == ""
        if final:
            lines = lines[:-1]
            scores = scores[:-1]
        n = len(lines)
        blank = [line.strip() == "" for line in lines]
        indent = [_indent_of(line) for line in lines]
        section = [-1] * n
        open_idx = -1
        for i, line in enumerate(lines):
            trimmed = line.strip()
            section[i] = open_idx
            if (_has_any_prefix(trimmed, _SECTION_PREFIXES)
                    or (_blen(trimmed) < 120 and trimmed.startswith("[")
                        and trimmed.endswith("]"))):
                open_idx = i
        out, kept = _render(lines, _select_lines(scores, blank, indent,
                                                 section, self))
        if final:
            out += "\n"
        return out, kept


# Operating points, tuned on the benchmark dev splits.
POINTS = {
    "conservative": Policy(floor=0.0035, top_k=3, context=1, headers=1,
                           gap=2, min_chars=2000, min_cut=0.1),
    "balanced": Policy(floor=0.008, top_k=3, context=1, headers=1,
                       gap=2, min_chars=2000, min_cut=0.1),
    "aggressive": Policy(floor=0.035, top_k=3, context=1, headers=1,
                         gap=2, min_chars=2000, min_cut=0.1),
}
DEFAULT_POINT = "balanced"


class Result:
    def __init__(self, output, kept_lines, removed_lines, point, policy,
                 unchanged=False, reason=""):
        # Output is the compressed text, or the context itself when unchanged.
        self.output = output
        self.kept_lines = kept_lines
        self.removed_lines = removed_lines
        self.point = point
        self.policy = policy
        # Unchanged is True when the context was returned as is; reason says why.
        self.unchanged = unchanged
        self.reason = reason


def _line_count(context: str) -> int:
    """Lines of context; the empty piece after a final "\\n" is not one."""
    return context.removesuffix("\n").count("\n") + 1


def _marker(n: int) -> str:
    if n == 1:
        return "\u2026 1 line \u2026"
    return "\u2026 %d lines \u2026" % n


def _unchanged(context, name, policy, reason) -> Result:
    return Result(output=context, kept_lines=_line_count(context),
                  removed_lines=0, point=name, policy=policy,
                  unchanged=True, reason=reason)


def compress(context: str, query: str, point: str = "balanced",
             policy: Policy | None = None) -> Result:
    """Keep the lines of context that the query needs. Fails closed: an
    unknown operating point, an unusable model, a blank query, a small or
    oversized input, a line that reads as a marker, or a cut too small to
    matter returns the context unchanged."""
    if policy is not None:
        name, pol = "custom", policy
    else:
        name = point or DEFAULT_POINT
        if name not in POINTS:
            return _unchanged(context, name, Policy(),
                              "unknown operating point")
        pol = POINTS[name]
    if query.strip() == "":
        return _unchanged(context, name, pol, "no query")
    if _blen(context) < pol.min_chars:
        return _unchanged(context, name, pol,
                          "context below the minimum size")
    if _blen(context) > MAX_CONTEXT_BYTES:
        return _unchanged(context, name, pol, "context over 8 MiB")
    if _line_count(context) > MAX_CONTEXT_LINES:
        return _unchanged(context, name, pol, "context over 200,000 lines")
    if "\u2026" in context and _MARKER_LINE.search(context):
        return _unchanged(context, name, pol,
                          "context has a line that reads as an elision marker")
    try:
        lines, scores = score(context, query)
    except Exception as exc:  # noqa: BLE001 - fail closed, reason carries it
        return _unchanged(context, name, pol, str(exc))
    out, kept = pol.apply(lines, scores)
    if _blen(out) > _blen(context) * (1 - pol.min_cut):
        return _unchanged(context, name, pol, "cut below the minimum")
    return Result(output=out, kept_lines=kept,
                  removed_lines=_line_count(context) - kept,
                  point=name, policy=pol)


def score(context: str, query: str):
    """Split context at "\\n" and return its lines with each line's keep
    probability for the query; blank lines score 0."""
    m = _model()
    lines, rows = line_features(context, query)
    scores = [0.0] * len(lines)
    for i, line in enumerate(lines):
        if line.strip() != "":
            scores[i] = m.probability(rows[i])
    return lines, scores


def _select_lines(scores, blank, indent, section, p: Policy):
    n = len(scores)
    order = [i for i in range(n) if not blank[i]]
    order.sort(key=lambda i: -scores[i])  # stable: ties keep line order
    seed = [False] * n
    best = scores[order[0]] if order else 0.0
    for k, i in enumerate(order):
        seed[i] = k < p.top_k or scores[i] >= p.floor * best
    keep = list(seed)
    headers = _block_headers(indent, blank)
    for i in range(n):
        if not seed[i]:
            continue
        for j in range(max(0, i - p.context), min(n - 1, i + p.context) + 1):
            keep[j] = True
        h = headers[i]
        cnt = 0
        while h >= 0 and cnt < p.headers:
            keep[h] = True
            h = headers[h]
            cnt += 1
        if p.headers > 0 and section[i] >= 0:
            keep[section[i]] = True
    # Fill short gaps and gaps of blank lines between kept lines.
    last = -1
    for i in range(n):
        if not keep[i]:
            continue
        if last >= 0:
            gap = blank[last + 1:i]
            if gap and (len(gap) <= p.gap or all(gap)):
                for j in range(last + 1, i):
                    keep[j] = True
        last = i
    return keep


def _render(lines, keep):
    """Write the kept lines verbatim and one marker line per removed run."""
    pieces = []
    kept = 0
    i = 0
    while i < len(lines):
        if i > 0:
            pieces.append("\n")
        if keep[i]:
            pieces.append(lines[i])
            kept += 1
            i += 1
            continue
        j = i
        while j < len(lines) and not keep[j]:
            j += 1
        pieces.append(_marker(j - i))
        i = j
    return "".join(pieces), kept
