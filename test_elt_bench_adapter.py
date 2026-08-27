import tempfile
import unittest
from pathlib import Path

from elt_bench_adapter import load_elt_bench_task


class TestELTBenchAdapter(unittest.TestCase):
    def test_loads_official_metadata_and_local_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "elt-bench" / "databricks" / "books"
            task_root.mkdir(parents=True)
            (task_root / "config.yaml").write_text("databricks: {}\n")
            (task_root / "data_model.yaml").write_text(
                "models:\n  - name: publishers\n    description: publisher model\n"
            )
            (root / "evaluation" / "sql" / "books").mkdir(parents=True)
            (root / "evaluation" / "sql" / "books" / "publishers.sql").write_text(
                "select * from publishers"
            )
            source_root = root / "local_data" / "databricks" / "books"
            source_root.mkdir(parents=True)
            (source_root / "books.csv").write_text("id,name\n1,A\n")
            truth_root = root / "truth" / "books"
            truth_root.mkdir(parents=True)
            (truth_root / "publishers.csv").write_text("publisher_id,num_books\n1,1\n")

            bundle = load_elt_bench_task(root, "databricks", "books", root / "truth")

            self.assertEqual(bundle.task.task_id, "databricks/books")
            self.assertIn("books", bundle.task.source_tables)
            self.assertEqual(bundle.task.target_table, "publishers")
            self.assertEqual(bundle.evaluation_sql["publishers"], "select * from publishers")


if __name__ == "__main__":
    unittest.main()
