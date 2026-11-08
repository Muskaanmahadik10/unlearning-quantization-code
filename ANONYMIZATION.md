# Removed account identifiers

Personal account identifiers were removed before publication. Only these strings were replaced, everywhere they occur:

| Original | Replacement |
|---|---|
| Cluster user name | `anon_user` |
| Scheduler project accounts | `anon_account`, `anon_account2`, `anon_account3` |
| E-mail address in scheduler notifications (job history only) | `<email>` |
| Numeric user ID in "CANCELLED by …" (job history only) | `<uid>` |

Nothing else was edited. Facility, hardware, model and dataset names are unchanged.

`FILE_MANIFEST.csv` lists every file with two fingerprints:
- `sha256`: the bytes published here.
- `original_sha256`: the bytes as originally run.

The `anonymized` column marks the files where the two differ. The authors keep the original files and can provide them to the editor on request, so that the `original_sha256` values can be checked independently.
