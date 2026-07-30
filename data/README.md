# data — cookiecutter layout

| Folder | Meaning |
|--------|---------|
| **raw/** | Original price CSVs. Do not edit by hand. |
| **interim/** | Half-built tables. |
| **processed/** | Ready for modeling. |
| **external/** | Third-party dumps. |

Curriculum gold file:

```text
data/raw/XAUUSD_curriculum_2026.csv
```

Code resolves CSVs via `core.configs.data_file(...)` or `path("data", "raw", ...)`.
