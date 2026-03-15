from __future__ import annotations

import re

ALLOWED_TABLES = {
    "orders",
    "customers",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "sellers",
    "geolocation",
    "vw_eligible_orders",
    "vw_fact_items",
    "vw_delivered_orders",
}

FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "copy",
    "attach",
    "detach",
    "merge",
    "replace",
    "grant",
    "revoke",
    "call",
    "execute",
    "pragma",
    "vacuum",
}


class SQLGuardError(ValueError):
    pass


class SQLGuard:
    def __init__(self, default_limit: int = 2000, allowed_tables: set[str] | None = None):
        self.default_limit = int(default_limit)
        self.allowed_tables = allowed_tables or ALLOWED_TABLES

    def validate_and_rewrite(self, sql: str) -> str:
        if not sql or not sql.strip():
            raise SQLGuardError("Empty SQL is not allowed.")

        cleaned = sql.strip()
        cleaned = self._strip_trailing_semicolon(cleaned)

        self._ensure_single_statement(cleaned)
        self._ensure_select_or_with(cleaned)
        self._deny_forbidden_keywords(cleaned)
        self._deny_comment_tokens(cleaned)
        self._check_table_whitelist(cleaned)

        return self._ensure_limit(cleaned)

    @staticmethod
    def _strip_trailing_semicolon(sql: str) -> str:
        return re.sub(r";\s*$", "", sql)

    @staticmethod
    def _ensure_single_statement(sql: str) -> None:
        if ";" in sql:
            raise SQLGuardError("Only single SQL statement is allowed.")

    @staticmethod
    def _ensure_select_or_with(sql: str) -> None:
        if not re.match(r"(?is)^\s*(with|select)\b", sql):
            raise SQLGuardError("Only SELECT/CTE SQL is allowed.")

    @staticmethod
    def _deny_comment_tokens(sql: str) -> None:
        if "--" in sql or "/*" in sql or "*/" in sql:
            raise SQLGuardError("SQL comments are not allowed.")

    @staticmethod
    def _normalize_ident(ident: str) -> str:
        ident = ident.strip().strip("`\"")
        if "." in ident:
            ident = ident.split(".")[-1]
        return ident.lower()

    def _deny_forbidden_keywords(self, sql: str) -> None:
        pattern = r"(?is)\b(" + "|".join(re.escape(k) for k in sorted(FORBIDDEN_KEYWORDS)) + r")\b"
        hit = re.search(pattern, sql)
        if hit:
            raise SQLGuardError(f"Forbidden keyword detected: {hit.group(1).upper()}")

    def _check_table_whitelist(self, sql: str) -> None:
        cte_names = set(self._extract_cte_names(sql))
        refs = re.findall(r"(?is)\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", sql)
        for raw in refs:
            table = self._normalize_ident(raw)
            if table in cte_names:
                continue
            if table not in self.allowed_tables:
                raise SQLGuardError(f"Table/view is not allowed: {table}")

    @staticmethod
    def _extract_cte_names(sql: str) -> list[str]:
        names = re.findall(r"(?is)(?:with|,)\s*([a-zA-Z_][\w]*)\s+as\s*\(", sql)
        return [n.lower() for n in names]

    def _ensure_limit(self, sql: str) -> str:
        if re.search(r"(?is)\blimit\s+\d+\b", sql):
            return sql
        return f"{sql}\nLIMIT {self.default_limit}"
