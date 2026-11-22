"""Format-only cleaning for product titles and descriptions.

Only markup and whitespace are changed. Wording, including marketing language, is kept:
removing it is what the LLM extraction arms test.
"""
import html
import re

# Whole blocks whose content is never shown to a reader: Word/Office settings, CSS, scripts, comments.
_HIDDEN_BLOCK = re.compile(r"<(xml|style|script)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Office conditional-comment markers whose "<!--" was already stripped upstream,
# e.g. "[if gte mso 9]>", "[if !supportLists]", "<![endif]", "[endif]".
_OFFICE_MARKER = re.compile(
    r"<?!?\[endif\]-*>?|\[if (?:gte |lte |gt |lt |!)?(?:mso|support)[^\]]*\]>?", re.IGNORECASE
)
# Only real HTML (or namespaced Office) tags are removed; "<alwaysonline>" or "<Model X>" is product text.
_HTML_TAGS = {
    "a", "abbr", "b", "big", "blockquote", "body", "br", "button", "caption", "center", "cite", "code",
    "col", "colgroup", "dd", "del", "div", "dl", "dt", "em", "embed", "font", "form", "h1", "h2", "h3",
    "h4", "h5", "h6", "head", "hr", "html", "i", "iframe", "img", "input", "ins", "label", "li", "link",
    "meta", "nobr", "object", "ol", "option", "p", "param", "pre", "s", "select", "small", "source",
    "span", "strike", "strong", "sub", "sup", "table", "tbody", "td", "textarea", "tfoot", "th",
    "thead", "title", "tr", "tt", "u", "ul", "wbr",
}
_TAG = re.compile(r"</?([a-zA-Z][\w:-]*)[^<>]*>")

# CSS left as plain text (its <style> tag was stripped upstream): "/* ... */" comments, and rules
# "selector { prop: value; ... }". Braces only count as CSS if every property is a known CSS
# property, so product text such as "{8oz}" or "{mutant: ...}" is kept. A selector token must start
# with ".", "#", "@" or be an HTML element name, so words glued in front ("images.GreyNote") are kept.
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_PROP = (
    r"\*?(?:-(?:webkit|moz|ms|o)-)?(?:mso-[\w-]+|(?:font|text|margin|padding|border|background|line|letter|word|vertical|list|page"
    r"|outline|box|white)(?:-[\w-]+)?|color|(?:max-|min-)?(?:width|height)|display|position|top|left"
    r"|right|bottom|float|clear|overflow|visibility|z-index|cursor|zoom|opacity|panose-1|src|size)"
)
_CSS_DECL = rf"{_CSS_PROP}\s*:\s*[^;{{}}]*"
_ELEMENT = "|".join(sorted(_HTML_TAGS, key=len, reverse=True))
_CSS_TOKEN = rf"(?<![\w-])(?:(?:{_ELEMENT})?[.#:@][\w-]+(?:[.#:][\w-]+)*|(?:{_ELEMENT})(?![\w-])|\*)"
_CSS_SELECTOR = rf"{_CSS_TOKEN}(?:\s*[,>+~]?\s*{_CSS_TOKEN})*"
_CSS_RULE = re.compile(
    rf"(?:{_CSS_SELECTOR}\s*)?\{{\s*(?:{_CSS_DECL};\s*)*{_CSS_DECL};?\s*\}}|{_CSS_SELECTOR}\s*\{{\s*\}}",
    re.IGNORECASE,
)
_DOCTYPE = re.compile(r'!?DOCTYPE\s+html[^"<>]*(?:"[^"]*"\s*){0,2}', re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def _unescape(text: str, max_rounds: int = 10) -> str:
    # Some descriptions are HTML-encoded more than once (e.g. "&amp;amp;gt;").
    for _ in range(max_rounds):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def _drop_tag(match):
    name = match.group(1).lower()
    return " " if name in _HTML_TAGS or ":" in name else match.group(0)


def clean_text(text) -> str:
    """Decode HTML entities, drop markup, collapse whitespace. Missing values become ""."""
    if not isinstance(text, str):  # NaN / None from pandas
        return ""
    text = _unescape(text)
    for pattern in (_HIDDEN_BLOCK, _COMMENT, _OFFICE_MARKER, _CSS_COMMENT, _CSS_RULE, _DOCTYPE):
        text = pattern.sub(" ", text)
    text = _TAG.sub(_drop_tag, text)
    return _WHITESPACE.sub(" ", text).strip()


def build_original_text(title, content) -> str:
    """Node text for the "original text" arms (A/B), in the format fixed by the team."""
    return f"Product: {clean_text(title)}; Description: {clean_text(content)}"
