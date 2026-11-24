import unittest
from prototype.report import assess, mcnemar_exact, render, score


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

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 1 / 32)
        self.assertAlmostEqual(mcnemar_exact(1, 1), 0.75)
        # Midterm BANKING77 protected suite: 129 regressions vs 11 gains
        self.assertLess(mcnemar_exact(129, 11), 1e-20)

    def test_mcnemar_gate(self):
        def suite(regress, gain, keep):
            rows = [{'id': 't', 'group': 'target', 'category': 'task', 'prompt': 'Route', 'expected': 'b',
                     'base_response': 'x', 'fine_response': 'b'}]
            for kind, count, base, fine in (('r', regress, 'yes', 'no'), ('g', gain, 'no', 'yes'), ('k', keep, 'yes', 'yes')):
                rows += [{'id': f'{kind}{i}', 'group': 'protected', 'category': 'fmt', 'prompt': 'Q',
                          'expected': 'yes', 'base_response': base, 'fine_response': fine} for i in range(count)]
            return rows
        blocked = assess(suite(6, 0, 94), gate='mcnemar')
        self.assertFalse(blocked['gate_passed'])
        self.assertTrue(blocked['paired_test']['significant_regression'])
        self.assertIn('Paired test', render(blocked))
        # 1 regression in 100 prompts: threshold gate at 0.5 pp blocks, McNemar does not
        noisy = suite(1, 0, 99)
        self.assertFalse(assess(noisy, max_protected_drop=0.005)['gate_passed'])
        self.assertTrue(assess(noisy, gate='mcnemar')['gate_passed'])
        with self.assertRaises(ValueError):
            assess(noisy, gate='bogus')

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            assess(self.cases() + self.cases())


if __name__ == '__main__':
    unittest.main()
