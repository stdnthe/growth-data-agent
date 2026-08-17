from __future__ import annotations

import unittest

from agent.sql_guard import SQLGuard, SQLGuardError


class SQLGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = SQLGuard(default_limit=100)

    def test_allows_read_only_query_and_adds_limit(self) -> None:
        safe = self.guard.validate_and_rewrite("SELECT order_id FROM orders")
        self.assertTrue(safe.endswith("LIMIT 100"))

    def test_rejects_write_operation(self) -> None:
        with self.assertRaises(SQLGuardError):
            self.guard.validate_and_rewrite("DELETE FROM orders")

    def test_rejects_unknown_table(self) -> None:
        with self.assertRaises(SQLGuardError):
            self.guard.validate_and_rewrite("SELECT * FROM private_customer_pii")

    def test_rejects_second_statement_injection(self) -> None:
        with self.assertRaises(SQLGuardError):
            self.guard.validate_and_rewrite("SELECT * FROM orders; DROP TABLE orders")


if __name__ == "__main__":
    unittest.main()
