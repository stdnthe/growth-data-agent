from __future__ import annotations

import unittest

from eval.run_trajectory_eval import DEFAULT_CASES, load_cases, run_case


class TrajectoryEvalTests(unittest.TestCase):
    def test_all_policy_contract_trajectories_pass(self) -> None:
        results = [run_case(case, use_llm=False, model=None) for case in load_cases(DEFAULT_CASES)]
        self.assertEqual(len(results), 12)
        self.assertTrue(all(item["passed"] for item in results), results)


if __name__ == "__main__":
    unittest.main()
