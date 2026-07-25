import re

TECH_KEYWORDS = ["código", "função", "banco", "bug", "php"]
USER_KEYWORDS = ["como usar", "manual", "passo a passo", "login", "tutorial", "ajuda"]


def route_query(query: str) -> dict:
    query_lower = query.lower()

    has_tech = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in TECH_KEYWORDS)
    has_user = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in USER_KEYWORDS)

    if has_tech and not has_user:
        return {"doc_type": ["tech", "support"]}
    if has_user and not has_tech:
        return {"doc_type": ["user", "support"]}
    if has_tech and has_user:
        return {"doc_type": ["user", "tech", "support"]}

    return {}
