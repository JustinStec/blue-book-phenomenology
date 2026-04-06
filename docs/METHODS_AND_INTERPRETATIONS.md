# Blue Book Encounter Phenomenology: Methods and Interpretations

## The Question

When a witness describes an encounter that defeats every available category — when the Air Force investigator checks "balloon," "aircraft," "astronomical," "insufficient data," and none of them hold — what remains in the testimony? Is the residue noise, or does it carry structure? If 701 cases independently resisted categorization across 22 years, do the testimonies that survived that resistance share phenomenological features that the resolved cases lack?

This project builds the computational instrument to find out. The complete declassified Project Blue Book archive (10,808 case files, 1947-1969) has been OCR'd, cleaned, corrected, and embedded in a high-dimensional vector space. The question is whether the 701 "unidentified" cases occupy a distinct region of that space — whether "unidentified" names a phenomenological category or a bureaucratic one.

## Why Computational Phenomenology

Phenomenology studies the structures of experience. The Blue Book archive preserves 12,618 attempts by witnesses to describe experiences that were, by definition, anomalous — outside the observer's existing framework. Each testimony records not just what was seen but how it was seen: the sequence of perceptual attention, the search for analogies ("like a giant firecracker," "resembling a C-54 without motors," "as if on wheels"), the moment where analogy fails ("That is impossible. I couldn't move as it did. It just moved. It didn't walk. It moved evenly. It didn't jump.").

No human researcher can hold 10,000 testimonies in memory simultaneously. Hynek, Vallee, and Sparks worked through the archive over years, case by case, building catalogs manually. They found patterns, but they found them through selective human attention. The computational approach does not replace that attention. It complements it by detecting structural regularities across the full corpus that no individual reader could perceive — recurring combinations of perceptual features, shared narrative structures, systematic relationships between what witnesses describe and what categories analysts impose.

The instrument is a sentence embedding model that converts each testimony into a 768-dimensional vector representing its semantic content. Testimonies with similar content land near each other in the vector space. Unsupervised clustering then identifies groups of testimonies that share structure without any prior labeling. The question becomes testable: do the 701 unidentified cases cluster differently from the resolved ones, or do they scatter uniformly?

## The Inscription Problem

The deepest methodological challenge is not computational but epistemological. The archive does not preserve raw experience. It preserves experience already processed through an inscription technology — the AF Form 112, the ATIC Form 329, the standardized questionnaire — designed to convert anomalous encounters into administratively tractable reports. The forms impose structure: TYPE OF OBSERVATION (Ground-Visual / Air-Visual / Ground-Radar). NUMBER OF OBJECTS. LENGTH OF OBSERVATION. COURSE. CONCLUSIONS (Was Balloon / Probably Balloon / Possibly Balloon / Was Aircraft / Probably Aircraft / Possibly Aircraft / Was Astronomical / Other / Insufficient Data for Evaluation / Unknown).

This inscription operates at every level. The form determines what questions get asked. The questions determine what details get recorded. The recorded details determine what categories are available for conclusion. The conclusion determines whether a case stays open or closes. At each stage, the institutional frame narrows what the testimony can say.

The initial embedding results confirm this. When the full corpus is embedded and clustered, the 185 clusters that emerge are organized primarily by geography (California, Texas, Ohio, Michigan) and secondarily by decade (1950s vs. 1960s). The embedding model clusters on document provenance — where and when a report was filed — rather than on phenomenological content — what witnesses saw. The institutional frame dominates the semantic signal so thoroughly that even a model trained on general English semantics cannot see past it to the experiential content beneath.

This is itself a finding. The forms work. The inscription technology the Air Force designed to standardize and close anomalous reports succeeded so completely that it obscures the phenomenological content not just from human readers accustomed to the form's structure but from a computational model encountering the text without any such habituation. The AF Form 112 does not merely surround the witness testimony. It infiltrates it at the level of word frequency and semantic structure.

The phenomenological question therefore becomes: can the witness voice be computationally separated from the institutional frame? If so, what structure does it carry?

## The Archive

Project Blue Book was the U.S. Air Force's official investigation of unidentified flying objects, running from 1952 to 1969 (preceded by Projects Sign and Grudge from 1947-1952). The archive contains 12,618 cases across approximately 130,000 pages. Of these, 701 cases remain officially "unidentified."

The archive is hosted on the Internet Archive as decade-based ZIP files (archive.org/details/bluebook), pre-segmented into individual case PDFs. The original documents were microfilmed when Blue Book closed in 1970, transferred to the National Archives, and later digitized. Each step introduced quality loss. Many pages are severely degraded 1950s military typewriter scans; others are handwritten witness statements, photographs, or blank microfilm frames.

The case files are heterogeneous. A single case may contain: an ATIC Form 329 record card (structured summary), an AF Form 112 intelligence report (semi-structured narrative), witness questionnaires, handwritten letters, FBI or ONI teletypes, newspaper clippings, inter-agency correspondence, analyst conclusions, investigator field notes, photographs, sketches, weather data, radar reports, and flight logs. The document types vary across decades and across the organizational transitions from Project Sign (1947-1949) to Grudge (1949-1952) to Blue Book (1952-1969).

## OCR Pipeline

### Phase 1: Marker OCR

Initial OCR used Marker (datalab-to/marker), a GPU-accelerated document understanding tool running on Google Colab Pro with A100 GPU. Marker performs layout detection, line detection, table recognition, and text recognition.

Results: 1,258 files processed over approximately 50 hours. Processing speed: ~30 files/hour. The bottleneck was Marker's computational overhead (layout analysis, model inference), not I/O.

Key decisions:
- `force_ocr=True` to bypass garbled embedded text layers
- `--disable_image_extraction` (photographs unnecessary for text analysis)
- Batches of 20 with per-batch checkpointing for session resilience
- `--workers 2` after observing GPU competition without throughput gains

### Phase 2: PyMuPDF Text Extraction (Abandoned)

Attempted direct extraction of embedded text using PyMuPDF (fitz). Of 9,550 files processed, the embedded text was predominantly Unicode garbage from the original digitization. Only 3,485 files (36%) produced readable text (>85% printable ASCII). Abandoned.

### Phase 3: Google Cloud Vision (Final)

Remaining 9,550 files processed via GCV's Document Text Detection API through Cloud Storage batch processing:

1. PDFs uploaded from Drive to GCS bucket (Google-to-Google)
2. Async batch API: 100 files per request, processed server-side in parallel
3. Results downloaded to Drive as markdown

Results: 9,414 readable files (98.6%). 134 empty (image-only). 2 garbage. Cost: ~$207 for ~139,000 pages.

### Combined Corpus

10,614 readable case files. Approximately 98% of the archive's text-bearing documents.

## Text Processing

### Cleaning

Regex-based pipeline strips:
- OCR artifacts (page markers, image placeholders, table borders, noise loops)
- Form boilerplate (ATIC Form 329, AF Form 112, FTD headers)
- Classification stamps and declassification markings
- Espionage Act boilerplate (identical paragraph in hundreds of documents)
- Hynek evaluation index (duplicated across many case files)
- Readability filter: files with <85% printable ASCII removed

### Spell Correction

Domain-aware correction using SymSpell with a protected terms dictionary covering military ranks, aircraft designators, base names, project names, personnel names, and abbreviation patterns. Prevents "correcting" domain vocabulary while fixing OCR misspellings in narrative content.

## Embedding and Clustering (Current State)

### Model

nomic-ai/nomic-embed-text-v1.5: 768 dimensions, 8,192-token context window. Chosen because standard sentence transformers (384-token limit) would silently truncate most case files. Full document text embedded with "search_document:" prefix per model protocol.

### Results

UMAP reduction to 2D + HDBSCAN clustering produced 185 clusters with 33% noise (3,542 unclustered cases).

**The clusters are geographic and temporal, not phenomenological.** The 20 largest clusters correspond to states (California: 311, Texas: 158, Michigan/Wisconsin: 109), cities (Chicago: 89, St. Louis: 72), and regions (Alaska: 71, Hawaii/Pacific: 81, Japan: 51). Ohio dominates 15+ clusters due to Wright-Patterson AFB's administrative presence. Most clusters are >70% from a single decade.

**The noise cases** (3,542) are evenly distributed across decades and have similar document lengths to clustered cases. They are not low-quality files. They are documents that resist the geographic/temporal clustering pattern.

**Interpretation:** The institutional frame — location names, base references, period-specific bureaucratic language — dominates the embedding signal. The model clusters on where and when reports were filed, not on what witnesses experienced. To reach the phenomenological content, the pipeline must separate witness voice from institutional frame.

## Toward Phenomenological Analysis

### The Problem

The current embedding captures document similarity, not experiential similarity. A case from Bakersfield, California describing a silent V-shaped object making instantaneous 90-degree turns clusters with other Bakersfield cases — not with a case from Bethel, Alaska describing a silent wingless object outrunning a DC-3, even though the phenomenological structure (silent operation, non-aerodynamic movement, trained observer, failure to categorize) may be identical.

### Approach 1: Geographic and Administrative Stripping

Remove all location names, base names, state names, dates, form field labels, analyst names, military unit designations, and bureaucratic language before embedding. What remains should be descriptive content: object shape, color, speed, behavior, sound, physical effects, witness response. Re-embed and re-cluster on stripped text. If the 701 cluster differently under this treatment, the phenomenological distinction is real.

### Approach 2: LLM-Based Narrative Extraction

Use a large language model to read each case file and extract only the witness's own description of what they experienced, discarding all administrative wrapper. This handles the document heterogeneity problem (FBI teletypes, handwritten letters, structured forms, newspaper clippings all contain witness testimony in different formats) better than regex patterns can. The extracted narratives would then be embedded and clustered.

### Approach 3: Structured Feature Extraction

Instead of embedding raw text, extract structured phenomenological features from each case:

**Perceptual features:**
- Object shape (disc, sphere, cigar, V-shaped, amorphous, light only)
- Color (metallic/reflective, self-luminous, dark/silhouette, color-changing)
- Sound (silent, buzzing, hissing, roaring, humming, none)
- Size estimate and distance estimate
- Duration of observation

**Behavioral features:**
- Movement pattern (straight, hovering, pacing vehicle/aircraft, formation, terrain-following, instantaneous acceleration, right-angle turns)
- Interaction with observer (approaching, retreating, circling, station-keeping, apparently observing)
- Departure (gradual, instantaneous, fading, behind terrain)

**Physical effects:**
- Electromagnetic interference (car engine/radio/lights stopping, radar return)
- Physiological effects (nausea, skin irritation, eye injury, paralysis)
- Ground traces (impressions, burned areas, broken vegetation)
- Animal reactions

**Witness and investigation features:**
- Observer type (civilian, military pilot, ground observer, radar operator, multiple independent)
- Credibility assessment in the file
- Gap between witness description and official conclusion

Cluster on these structured features rather than document text. This sidesteps geographic contamination entirely.

### Approach 4: Contrastive Embedding

Rather than clustering all 10,614 cases, directly compare the 701 unidentified cases against matched identified cases. For each unidentified case, find the most semantically similar identified case and measure the distance. If unidentified cases are systematically farther from their nearest identified neighbor than identified cases are from each other, the distinction carries phenomenological weight.

### The 701 Overlay (Immediate Next Step)

Tag each case with its official conclusion using the Brad Sparks catalog and overlay on the current embedding map. Even with geographic clustering, the distribution of the 701 across the space reveals whether unidentified cases concentrate in particular regions, periods, or document types — or scatter uniformly, confirming that the distinction operates at a level the current embedding cannot detect.

### BERTopic Analysis (In Progress)

BERTopic topic modeling has been run on the full corpus to extract keyword-based topic descriptions for each cluster. This provides interpretable labels for what the embedding model is detecting in each cluster — whether the topics correspond to geography (as the cluster analysis suggests), to document type, to phenomenological content, or to some combination. Results pending local inspection of the trained model.

## Cross-Referencing the VASCO Transients

A separate line of evidence intersects the Blue Book archive chronologically. The VASCO Project (Villarroel et al.) identified clusters of sub-second optical transients on Palomar Sky Survey photographic plates from April 1950 — brief, star-like flashes that appear and vanish within single exposures, sometimes aligned in groups. Busko (2025, arxiv:2507.15896) analyzed the image profiles of these transients and found they exhibit systematically narrower full width at half maximum (FWHM) than stellar point spread functions, consistent with reflections from flat, rotating objects in orbit. A March 2026 follow-up (Busko, arxiv:2603.20407) independently confirmed these findings using Hamburg Observatory plates from the mid-1950s, providing cross-institutional validation. A critical evaluation (arxiv:2601.21946) raised methodological concerns about dataset definitions but did not account for the independent Hamburg replication.

The April 1950 date places the VASCO transients directly within the Blue Book archive's temporal coverage. If sub-second optical flashes on photographic plates correspond to reflective objects in low Earth orbit — before any known satellite launch — the Blue Book witness testimonies from the same period become a potential correlate. The pipeline should cross-reference VASCO transient dates and sky coordinates against Blue Book case dates and locations to test whether witness sighting reports cluster temporally around the photographic plate exposures that captured the transients. A positive correlation would suggest that whatever produced the photographic anomalies also produced visual encounters reportable through military channels. A null result would indicate the two phenomena are unrelated.

### Nuclear Testing Correlation

Villarroel and Bruehl (2025, *Scientific Reports*) found that the Palomar transients are 45% more likely to appear on dates within one day of a nuclear test. On dates when transients were identified, every additional UAP report on that date correlated with an 8.5% increase in the number of transients. The Blue Book archive covers 1947-1969, the peak period of atmospheric nuclear testing. The temporal and geographic correlation between nuclear activity and Blue Book case density is directly testable with our database — case dates are already parsed and indexed, and nuclear test dates and locations are publicly available through the Oklahoma Geological Survey and DOE records. If Blue Book witness reports cluster around nuclear test events independently of the photographic plate transients, the correlation has a third leg.

## What the Archive Teaches Before Computation

Reading the archive case by case reveals structures that the pipeline should eventually detect computationally:

**The assumption-first analysis.** Project Grudge's evaluations follow a consistent rhetorical pattern: assume a conventional explanation, then show that the testimony does not preclude it. "There is nothing at all in the evidence that cannot be explained under the assumption that the object was a meteor." The burden falls on the witness to actively disprove the conventional explanation, which eyewitness testimony can never do. The pipeline should be able to detect and quantify the semantic gap between witness descriptions and analyst conclusions.

**The phenomenological signatures of unresolved cases.** Close reading of the 701 reveals recurring experiential structures that cut across geography and time: silent operation at close range, objects that pace vehicles or aircraft while maintaining precise station-keeping, electromagnetic interference that correlates with proximity and reverses on departure, witnesses who reach for analogies and explicitly report their failure ("unlike any conventional type aircraft," "I couldn't move as it did"), physical effects on witnesses and animals. Whether these features co-occur at statistically significant rates — and whether their co-occurrence distinguishes the 701 from resolved cases — is the testable phenomenological question.

**The institutional processing of the uncategorizable.** The archive preserves not just what witnesses saw but how the institution handled testimony that exceeded its categories. The Flatwoods entity, the Bethel Alaska pursuit, the Philadelphia multi-witness case, the Bakersfield drive-in — each generated extensive investigation that systematically eliminated conventional explanations, then closed with conclusions ("mundane creature of the woods," "partially blinded by the evening sky," "no further action") that bear no structural relationship to the testimony they purport to explain. The distance between testimony and conclusion, measurable in embedding space, is itself a phenomenological datum.
