# Blue Book Encounter Phenomenology

**An NLP pipeline for extracting and clustering witness narratives from the U.S. Air Force's Project Blue Book archive (1947-1969).**

This project applies computational phenomenology to the 12,618 case files declassified under Project Blue Book, focusing on what witnesses reported experiencing rather than what investigators concluded. By encoding witness narratives as dense vector representations and clustering them in embedding space, the pipeline surfaces phenomenological signatures — recurring structures of perceptual experience — that cut across the Air Force's own classificatory scheme.

## Research Context

This work belongs to a broader research program investigating the relationship between perceptual experience, cognitive architecture, and the forms through which experience gets reported. The central question — whether encounters classified as "Unidentified" exhibit phenomenologically distinct narrative structures compared to resolved cases — connects to several adjacent projects:

- **"A Post-Kantian Model of Poetic Cognition"** (in preparation for *Representations*): develops a Kantian and post-Kantian theory of how imagination mediates between sensibility and understanding, drawing on T. S. Eliot's critical writings and Milton's verse. The theoretical architecture — Kant's two stems of cognition, the synthesizing activity of imagination, the role of inner sense in tracking whether a manifold of intuitions coheres — provides the phenomenological vocabulary applied here to witness testimony.

- **[poetry-bert-formalism](https://github.com/JustinStec/poetry-bert-formalism)**: period-specific BERT models for diachronic poetry analysis (1595-2025). Shares the methodological commitment to embedding-based discovery of formal signatures in textual corpora.

- **[phonemic-analysis-dashboard](https://github.com/JustinStec/phonemic-analysis-dashboard)**: interactive visualization of phonemic patterns across five centuries of English verse. Demonstrates the same computational approach to sensory and formal features of language.

The convergence across these projects is methodological and theoretical: embedding representations capture aspects of textual structure that resist traditional categorical analysis, whether the texts in question are poems or witness reports.

## Pipeline Overview

The pipeline runs as a [Google Colab notebook](blue_book_ocr_pipeline.ipynb) optimized for GPU runtime (A100 recommended).

### Stage 1: Document Acquisition & OCR
- Downloads NARA microfilm reels from the [Internet Archive](https://archive.org/details/nara-pbb) (90 rolls, ~130,000 pages total; POC targets rolls 1-10, covering 1947-1952)
- Assesses existing OCR quality using character-count and garbage-ratio heuristics
- Re-OCRs degraded scans using [Marker](https://github.com/VikParuchuri/marker) with forced OCR mode, optimized for 1940s-1960s military typewriter documents

### Stage 2: Case Segmentation & Narrative Extraction
- Segments multi-case microfilm rolls into individual case documents using AF Form 112 boundary patterns
- Extracts witness narrative sections (Section 11: "Brief Summary of Sighting," attached witness statements, interview transcripts)
- Filters to cases with substantive narratives (>200 characters)

### Stage 3: Unsupervised Phenomenological Analysis
- Encodes witness narratives with `all-mpnet-base-v2` sentence-transformer
- Reduces dimensionality with UMAP (cosine metric, 2D projection)
- Clusters with HDBSCAN (density-based, no predetermined cluster count)
- Interprets clusters with BERTopic for automatic keyword extraction

### Planned Extensions
- **Layer 1 — Annotation**: Export narratives to Label Studio; annotate phenomenological spans (initial awareness, perceptual features, affective response, interpretive framing); fine-tune DeBERTa-v3-base for span classification
- **Layer 2 — Affect Trajectories**: Sentence-level valence-arousal-dominance scoring across each narrative; compare affective arcs between clusters
- **Layer 3 — Matched Control Analysis**: Overlay USAF classification (Unidentified vs. Identified) on embedding space; statistical tests for phenomenological distinctiveness
- **Layer 4 — Cross-Corpus Transfer**: Apply pipeline to MUFON case files, Colares incident reports, and other encounter archives to test whether phenomenological signatures generalize

## Data Sources

| Source | Content | Access |
|--------|---------|--------|
| [NARA Microfilm T1206](https://archive.org/details/nara-pbb) | 90 reels of declassified Project Blue Book case files | Public domain, Internet Archive |
| [Brad Sparks Catalog](https://www.nicap.org/bb/BB_Unknowns.pdf) | Comprehensive catalog of 701 Project Blue Book "Unknowns" | Public, via NICAP |
| [National Archives](https://www.archives.gov/research/military/air-force/ufos) | Original records, research room access | Public |

## Technical Requirements

- **Runtime**: Google Colab with GPU (A100: ~36 pages/min OCR throughput; T4: ~12 pages/min)
- **Storage**: Google Drive (~3-5 GB for POC subset; ~30 GB for full archive)
- **Key dependencies**: `marker-pdf`, `sentence-transformers`, `umap-learn`, `hdbscan`, `bertopic`, `polars`, `PyMuPDF`

## Repository Structure

```
blue_book_ocr_pipeline.ipynb   # Full pipeline notebook (Colab-ready)
README.md                      # This file
```

Pipeline outputs (generated on Google Drive during execution):
```
blue_book_phenomenology/
  pdfs/                        # Downloaded microfilm roll PDFs
  ocr_raw/                     # Text extracted from good-OCR PDFs
  marker_output/               # Marker re-OCR'd text (markdown)
  corpus/
    blue_book_cases.parquet     # Segmented individual cases
    blue_book_narratives.parquet # Extracted witness narratives
    narrative_embeddings.npy    # Sentence-transformer embeddings
  metadata/
    ocr_assessment.json         # Per-PDF OCR quality scores
    marker_results.json         # Marker processing log
    bb_unknowns.csv             # Case index (user-supplied)
  bertopic_model/               # Saved BERTopic model
  checkpoint.json               # Session state for resumption
```

## Author

**Justin Thomas Stec**

Research at the intersection of philosophy, literary theory, cognitive science, and computational text analysis. Nine article projects in development, unified by questions of impersonality, emotion, intersubjectivity, and the relationship between form and cognitive-affective processes.

## License

This pipeline code is released under the MIT License. Project Blue Book records are U.S. government documents in the public domain.
