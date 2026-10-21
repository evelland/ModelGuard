import unittest
from prototype.report import assess, render, score


class ReportTests(unittest.TestCase):
    def cases(self, regressed=True):
        return [
            {'id': 'p', 'group': 'protected', 'category': 'fmt', 'prompt': 'Hello', 'expected': 'yes', 'base_response': 'yes', 'fine_response': 'no' if regressed else 'yes'},
            {'id': 't', 'group': 'target', 'category': 'task', 'prompt': 'Route', 'expected': 'billing', 'base_response': 'wrong', 'fine_response': 'billing'},
        ]

    def test_blocks_regression(self):
        r = assess(self.cases(), max_protected_drop=0.10)
        self.assertFalse(r['gate_passed'])
        self.assertEqual(len(r['regressions']), 1)
        self.assertIn('DEPLOY GATE: BLOCK', render(r))

    def test_passes_retained_behavior(self):
        r = assess(self.cases(False))
        self.assertTrue(r['gate_passed'])
        self.assertEqual(r['metrics']['target']['delta'], 1.0)

    def test_validators(self):
        self.assertTrue(score({'validator': 'json_fields', 'expected': {'ok': True}}, '{"ok":true}'))
        self.assertFalse(score({'validator': 'json_fields', 'expected': {'ok': True}}, 'not json'))
        self.assertTrue(score({'validator': 'contains', 'expected': 'world'}, 'hello WORLD'))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            assess(self.cases() + self.cases())


if __name__ == '__main__':
    unittest.main()
