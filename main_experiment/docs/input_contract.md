# Prepared input contract

No raw CSV, taxonomy parsing, Word2Vec fitting or event-memory construction is performed by this package. The prepared workspace must contain:

```text
data/
  processed/
    views/{CN,IN,US}_{All,B,F,C}.json
    {cn,in,us}/
      manifest.json
      subclasses.json
      organization_first_seen.npz
      counts_<year>.npz
      collaborators_<year>.npz
  text/{cn,in,us}/
    manifest.json
    taxonomy_anchors.npz
  memory/<country[_section]>/f180_s1095_n1/
    manifest.json
    <year>.npz
```

Only the requested country's files are needed for one run. A country-All memory directory is named `cn`, `in` or `us`; section directories are `cn_B`, `in_C`, etc. Other half lives and novelty values select `f<fast>_s<slow>_n<novelty>` using the same naming convention as existing artifacts.

`counts_<year>.npz` is a scipy CSR matrix of organization-by-subclass event counts. Organization and subclass IDs must agree with the view and all saved memories. `organization_first_seen.npz` contains the `year` array. `collaborators_<year>.npz` contains `partners`, with exactly three columns; `-1` denotes missing partners.

`taxonomy_anchors.npz` contains matching `codes` and 100-dimensional `anchors`, derived from the corresponding country's IPCSection/IPCClass/IPCSubClass hierarchy. The package reads these fixed vectors without updating or rebuilding the taxonomy.

Each memory snapshot contains `org_fast`, `org_slow`, `ipc_fast`, `ipc_slow`, their `_mass` and `_age` arrays. Each fast/slow state has 200 columns: text first, structural identity last. The package concatenates fast then slow, preserving the trained 400-column layout. For level 1 or 2 prediction, subclass states aggregate to their selected parent using the stored slow mass; this is the existing experiment's coarse-level readout, not a new event replay at the parent level.

Protocol: CN/US use 2010–2019, training through 2017, validation 2018 and test 2019. IN uses 2018–2023, training through 2021, validation 2022 and test 2023. Query organizations must exist before the target calendar year. Labels are the next calendar year's IPC set; all previously known organizations are queried, with active, empty and zero-filled populations separately reported. Candidate classes are training-observed; out-of-catalog positives remain misses.

The fixed protocol is copied into `configs/protocol.json`. Its SHA256 is `060b1d8d3a7a05587310f5d2f1292c2bc0137361be54504b13c9ecbc366ef9e7`. Keep original prepared manifest hashes and checkpoint hashes with a released artifact bundle. Raw text and raw patent records are not necessary for the main training entry point once these inputs are prepared.
