# Anonymous release checklist

The repository contents do not embed an author name or owner-bound notebook
URL. For double-blind review, the repository itself must also be hosted at an
anonymous URL; changing files cannot anonymize the owner shown in the current
hosting address or the existing Git commit metadata.

1. Create an anonymous repository mirror.
2. Import the `initial-release` tree as a new squashed root commit with neutral
   author metadata.
3. Copy the release assets and preserve the tags `probe-checkpoints-v1`,
   `wall-probe-data-v1`, and `pusht-probe-cache-v1`.
4. Set `PROBING_VLMS_REPO_URL` to the anonymous clone URL.
5. Set `PROBING_VLMS_RELEASE_BASE` to the anonymous repository's
   `releases/download` base URL.
6. Verify the three notebooks from a signed-out browser before submission.

Do not redirect the anonymous mirror to an author-owned repository during the
review period.
