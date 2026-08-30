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
4. Replace the two public default URLs in each notebook setup cell with the
   anonymous mirror URLs, or set `PROBING_VLMS_REPO_URL` and
   `PROBING_VLMS_RELEASE_BASE` before the setup cell.
5. Verify that the anonymous release base serves all three preserved tags.
6. Verify the three notebooks from a signed-out browser before submission.

Do not redirect the anonymous mirror to an author-owned repository during the
review period.
