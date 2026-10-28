"""Offline tests for frozen new-template benchmark evaluation."""
import tempfile
import unittest
from pathlib import Path

from prototype.sae_v23_newtest import outcome_summary, validate_rows, checksum


class NewTestTests(unittest.TestCase):
    def rows(self):
        cases = [
            {"id":"a","group":"protected","category":"Arithmetic",
             "template_family":"two_step_arithmetic","prompt":"calculate","expected":"4","validator":"exact"},
            {"id":"b","group":"protected","category":"Strict JSON formatting",
             "template_family":"strict_json_status","prompt":"json","expected":{"status":"ready","amount":7},"validator":"json_fields"},
        ]
        predictions = [dict(cases[0], source="model_inference", base_response="4", fine_response="3"),
                       dict(cases[1], source="model_inference", base_response='{"status":"ready","amount":7}',
                            fine_response='{"status":"ready","amount":7}')]
        return cases, predictions

    def test_behavior_summary_and_new_regressions(self):
        cases, preds = self.rows()
        result=outcome_summary(cases,preds)
        self.assertEqual(result["total"]["n"],2)
        self.assertEqual(result["total"]["new_regressions"],1)
        self.assertEqual(result["total"]["retained_base_correct"],1)

    def test_alignment_and_mismatch(self):
        cases, preds = self.rows()
        validate_rows(cases,preds)
        with self.assertRaises(ValueError):
            validate_rows(cases,list(reversed(preds)))
        preds[0]["prompt"]="altered"
        with self.assertRaises(ValueError):
            validate_rows(cases,preds)

    def test_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.txt"
            p.write_text("frozen")
            a=checksum(p)
            p.write_text("modified")
            self.assertNotEqual(a,checksum(p))


if __name__=="__main__":
    unittest.main()
