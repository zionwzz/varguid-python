# varguid 0.1.8 publication checklist

Project title:

> **{varguid}: Variance-Guided Regression Improving Upon OLS and ANOVA for Python**

Canonical source repository:

> `https://github.com/zionwzz/varguid-python`

## Correct release order

This checklist uses a **manual Zenodo software deposit with a DOI reserved in
advance**. That route allows the version DOI to be embedded in the 0.1.8 source,
wheel metadata links, README, and `CITATION.cff` before publication. Do not also
turn on automatic Zenodo GitHub archiving for the same `v0.1.8` release, because
that could create a second record for the same object.

1. Rename the existing GitHub repository from `varGuid-py` to
   `varguid-python`. Preserve its history rather than creating a second
   repository.
2. Replace the repository contents with this reviewed 0.1.8 source tree.
3. In Zenodo, create a new upload with resource type **Software**, save it as a
   draft, and select **Get a DOI now**. Do not publish the record yet.
4. Insert the reserved version DOI in all of the following places:
   - `CITATION.cff`: add a top-level `doi:` value;
   - `CITATION.bib`: add `doi = {...}` and a DOI URL;
   - `README.md`: add the DOI to the software BibTeX entry;
   - `pyproject.toml`: add a `Software DOI` entry under `[project.urls]`.
5. Add `date-released: YYYY-MM-DD` to `CITATION.cff`.
6. Run every local validation command in `RELEASING.md`, then inspect the wheel
   and source distribution.
7. Commit the final files, create the immutable tag `v0.1.8`, and build the
   distributions from that exact tag.
8. Upload **one source-code ZIP** for tag `v0.1.8` to the Zenodo draft, verify
   the creators, GPL-2.0-or-later license, title, version, method-paper relation,
   and repository URL, then publish the Zenodo record.
9. Configure a PyPI pending Trusted Publisher for repository
   `zionwzz/varguid-python`, workflow `.github/workflows/publish.yml`, and
   environment `pypi`.
10. Publish the GitHub release for tag `v0.1.8`. The release workflow should
    build from that tag and publish the wheel and sdist to PyPI.
11. Verify the GitHub release, Zenodo record, DOI resolution, PyPI page, and
    installed-package metadata.

## What PyPI does and does not store

PyPI does not have a dedicated BibTeX or citation-metadata field. Citation
information is communicated through the rendered README and project URLs in
`pyproject.toml`. `CITATION.cff` is primarily consumed by GitHub and software
archives such as Zenodo.

## Later software paper

The software paper should cite the version-specific Zenodo DOI for the release
used in the paper and cite the method article DOI `10.1002/sim.70632`. After the
software paper is accepted or published, add its final DOI to `CITATION.cff`,
`CITATION.bib`, README, and `[project.urls]`, then issue a new patch release. Do
not alter the archived files for version 0.1.8.
