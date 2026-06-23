# Releasing varguid

## One-time repository setup

1. Rename the existing repository to `zionwzz/varguid-python` and update all
   local remotes and external links.
2. In PyPI, create a pending Trusted Publisher for project `varguid`, GitHub
   repository `zionwzz/varguid-python`, workflow file
   `.github/workflows/publish.yml`, and environment `pypi`.
3. In GitHub, create the protected `pypi` environment. Add required reviewers
   or other deployment protection rules.
4. Do not store a long-lived PyPI API token in repository secrets; publishing
   uses GitHub's short-lived OIDC identity.
5. Protect the default branch, restrict permitted third-party Actions, and
   review Dependabot updates to pinned Action commit SHAs.

## DOI strategy for version 0.1.8

Use the manual Zenodo workflow described in `PUBLICATION_CHECKLIST.md` if the
software DOI must appear inside the initial 0.1.8 package metadata. Reserve the
DOI in a Zenodo draft before finalizing the tag. Do not archive the same release
through both manual upload and automatic GitHub integration.

An alternative is to enable Zenodo's GitHub integration and let a published
GitHub release mint the DOI automatically. With that route, 0.1.8 cannot contain
its own newly minted DOI; add the DOI in a later patch release instead.

## Prepare a release

1. Update the version in `pyproject.toml`, `src/varguid/__init__.py`,
   `CITATION.cff`, and `CITATION.bib`.
2. Update `CHANGELOG.md` and any version-specific audit logs.
3. Confirm that the title, repository URL, method-paper citation, release date,
   and any reserved software DOI agree across all metadata files.
4. Run:

   ```bash
   python -m pytest
   ruff check src tests
   ruff format --check src tests
   mypy src tests
   check-manifest
   rm -rf build dist src/*.egg-info
   python -m build
   python -m twine check dist/*
   ```

5. Inspect the wheel and source distribution to confirm that the README,
   citations, documentation, license files, workflows, and package data are
   present as intended while caches and previous build products are absent.
6. Tag the reviewed commit and publish the corresponding GitHub release only
   after the Zenodo record is ready under the selected DOI workflow.

Use a signed tag or a tag pointing to a verified signed commit whenever
possible.

Publishing the GitHub release triggers `.github/workflows/publish.yml`. The
workflow builds and validates distributions, produces provenance attestations,
and publishes through PyPI Trusted Publishing.
