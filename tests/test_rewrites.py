import unittest

from assemble_rewrites import assemble
from check_answers import audit
from mario_data import record


class RewriteTests(unittest.TestCase):
    def setUp(self):
        self.original = record({"question": "How many apples are left after eating 7 of 18?",
                                "answer": "Subtract: 18 - 7 = <<18-7=11>>11 apples.\n#### 11"},
                               0, "train", False, 42)
        self.rewrite = {"id": self.original["id"],
                        "answer": "Hungry, eh? I take the 7 eaten apples away from the 18 we had: "
                                  "18 - 7 = <<18-7=11>>11. Wahoo, 11 apples left to share!\n#### 11"}

    def test_real_paraphrase_is_allowed(self):
        rows, report = assemble([self.original], [self.rewrite])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["semantic_fidelity"], "not_evaluated")
        self.assertEqual(report["persona_quality"], "not_evaluated")
        self.assertEqual(rows[0]["messages"][-1]["content"], self.rewrite["answer"])
        self.assertEqual(audit([self.original], rows, mode="legacy")[0]["failed"], 1)

    def test_changed_math_is_rejected(self):
        for answer in (self.rewrite["answer"].replace("#### 11", "#### 12"),
                       self.rewrite["answer"].replace("<<18-7=11>>", "<<18-6=12>>")):
            with self.assertRaises(ValueError):
                assemble([self.original], [{**self.rewrite, "answer": answer}])

    def test_partial_requires_explicit_flag(self):
        references = [self.original, {**self.original, "id": "another"}]
        with self.assertRaises(ValueError):
            assemble(references, [self.rewrite])
        rows, report = assemble(references, [self.rewrite], allow_partial=True)
        self.assertEqual(len(rows), 1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["missing"], 1)

    def test_duplicate_and_unknown_rejected(self):
        for rewrites in ([self.rewrite] * 2, [{**self.rewrite, "id": "unknown"}]):
            with self.assertRaises(ValueError):
                assemble([self.original], rewrites)

    def test_reference_order_controls_assembly(self):
        references = [self.original, {**self.original, "id": "second"}]
        rewrites = [{**self.rewrite, "id": "second"}, self.rewrite]
        rows, report = assemble(references, rewrites)
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in references])
        self.assertTrue(report["complete"])

    def test_omitted_annotation_is_rejected(self):
        with self.assertRaises(ValueError):
            assemble([self.original], [{**self.rewrite, "answer": self.rewrite["answer"].replace("<<18-7=11>>", "")}])

    def test_dataset_final_line_must_be_exact(self):
        with self.assertRaises(ValueError):
            assemble([self.original], [{**self.rewrite, "answer": self.rewrite["answer"].replace("#### 11", "#### 11.0")}])


if __name__ == "__main__":
    unittest.main()
