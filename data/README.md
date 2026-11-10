# Local sample data

Provide these files (not included in Git):

- `pdfs/test_report/report.pdf`: the sample sustainability report.
- `processed/embeddings/test_embeddings/test_embeddings.duckdb`: existing page
  and search-query embeddings made with `text-embedding-ada-002`.
- `evaluation_dataset/test_evaluation/gold_standard.csv`: reference extractions;
  `report_name` must match the PDF filename.

These are the local sample artifacts copied from the ClimateXtract checkout.
To add reports, update `configs/climatextract_test.toml` and provide matching
cached embeddings and gold-standard entries. Do not reuse embeddings for a
changed PDF with the same name: this cache identifies reports by filename.
The sample runner does not generate missing embeddings or download datasets.
