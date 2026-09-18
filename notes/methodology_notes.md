# Methodology notes — taakvenster-indeling (PsychoPy)

Status: vastgelegd op basis van participant **p20** (bestand `p20_test builder exp_2026-03-18_13h57.33.295.csv`/`.log`). Definitie is generiek geformuleerd (kolomnamen, niet hardcoded tijdstippen) zodat hij per participant herhaald kan worden; nog niet geverifieerd op de rest van de steekproef.

## Experimentstructuur (zoals die uit de PsychoPy-data blijkt)

Het experiment bevat **4 blokken** met trials, elk gevolgd door een rating-routine (`Rating1`–`Rating4`, identieke template: polygon + text + `key_resp_1x`, single-keypress antwoord):

1. `trials_stroop_pr` (oefenblok, 3 trials) → `trials_stroop` (hoofdtaak, 144 trials) → **Rating1**
2. `trials_faceMemory` (encodeerfase, 64 trials, geen response-kolom/geen reactietijd-druk) → **Rating2**
3. `trials_arithmetic_pr` (oefenblok, 3 trials) → `trials_arithmetic` (hoofdtaak, 50 trials) → **Rating3**
4. `trials_faceRecall` (hoofdtaak, 64 trials, mét response `key_resp_8`) → **Rating4**

Elke Rating start binnen enkele milliseconden na de laatste `.stopped`-timestamp van het voorgaande blok (geverifieerd, zie vorige analyse) — dus de koppeling blok→rating is eenduidig en niet toevallig.

## Beslissing: 3 taakvensters, encodeerfase uitgesloten

Voor het stress-detectiemodel worden alleen **3 taken** gebruikt: **Stroop, Arithmetic, Face Recall**. De FaceMemory-encodeerfase (blok 2) wordt bewust **niet** gebruikt als taakvenster, en `Rating2` (de rating na encoding) wordt niet gebruikt als target-label.

**Onderbouwing:** de encodeerfase kent geen actieve respons en geen tijdsdruk (participant bekijkt passief gezichten, er is geen `key_resp`-kolom in `trials_faceMemory` die op reactie/prestatie wijst zoals bij de andere drie taken). Cognitieve belasting is daardoor kwalitatief anders dan bij Stroop (interferentie onder tijdsdruk), Arithmetic (rekenen onder tijdsdruk) en Face Recall (herkenning met respons). De encodeerfase is wel functioneel noodzakelijk — zonder encoderen is er niets om bij Face Recall te herkennen — maar levert zelf geen vergelijkbaar stressvenster op.

Dit betekent: bij N participanten zijn er 4×N ratingmomenten in de ruwe data, maar slechts **3×N** worden gebruikt als target (bij N=76 dus 228 observaties, conform proposal).

## Definitief knipschema (per taak)

| Taak | Trial-loop (kolomprefix) | Practice-blok (genegeerd) | Start-marker | Eind-marker (laatste trial) | Target-rating |
|---|---|---|---|---|---|
| Stroop | `trials_stroop` | `trials_stroop_pr` | `stroop_begin.started` | laatste `blank.stopped` van `trials_stroop` | `Rating1` (`key_resp_11`) |
| Arithmetic | `trials_arithmetic` | `trials_arithmetic_pr` | `arithmetic_begin.started` | laatste `arithmetic_2.stopped` van `trials_arithmetic` | `Rating3` (`key_resp_13`) |
| Face Recall | `trials_faceRecall` | — | `instruct_faceRecall.started` | laatste `face_recall_blank_2.stopped` van `trials_faceRecall` | `Rating4` (`key_resp_14`) |

**Volledig genegeerd voor taakvensters:** `trials_stroop_pr`, `trials_arithmetic_pr`, `trials_faceMemory`, `Rating2`.

### Ter illustratie/validatie: waarden voor p20 (seconden vanaf `expStart`)

| Taak | Start | Eind | Duur |
|---|---|---|---|
| Stroop | ~431,8 (eerste trial `num1_text.started`) | 763,65 | ~332 s |
| Arithmetic | ~1435,9 (eerste trial `arithmetic_2.started`) | 1881,62 | ~446 s |
| Face Recall | ~2019,7 (eerste trial `face_recall.started`) | 2289,79 | ~270 s |

Deze absolute waarden zijn **per participant uniek** (afhankelijk van reactietijden, instructietempo, etc.) en dus niet herbruikbaar als constantes — elk participant-bestand moet opnieuw doorzocht worden volgens de kolomlogica hierboven, niet op basis van deze concrete getallen.

## Extractielogica (voor scripting)

Voor een gegeven participant-CSV:
1. Filter rijen op niet-lege waarde in `<loopnaam>.thisN` om de trials van één taak te isoleren (`trials_stroop`, `trials_arithmetic`, `trials_faceRecall`).
2. Start van het taakvenster = `.started`-tijdstip van de instructie-/begin-routine die direct voorafgaat (`stroop_begin`, `arithmetic_begin`, `instruct_faceRecall`), óf — als een strakkere afbakening gewenst is — het `.started`-tijdstip van de eerste stimulus-component in de eerste trial-rij van het blok.
3. Einde van het taakvenster = laatste relevante `.stopped`-tijdstip in de laatste trial-rij van het blok (kolom verschilt per taak, zie tabel).
4. Target-label = het `key_resp_1x`-antwoord van de bijbehorende Rating-routine, gekoppeld via volgorde (Rating1→Stroop, Rating3→Arithmetic, Rating4→Face Recall) — **niet** Rating2.
5. Koppeling naar PPG/video: tijden hierboven zijn relatief aan `expStart` (zelfde as als het `.log`-bestand); voor absolute klok­tijd `expStart`-timestamp per participant gebruiken om te synchroniseren met de Unix-timestamps in de PPG-data en de OBS start/stop-tijden.

## Synchronization Approach

- No hardware trigger/sync signal available between PPG, video, and PsychoPy.
- Synchronization via wall-clock alignment:
  - PPG: Unix timestamp (ms), absolute.
  - Video/OBS: absolute start/stop time from OBS log.
  - PsychoPy: relative time (as in .log, 0.000 = experiment start) converted to absolute time via `expStart` column (timezone-aware).
- **No buffer/margin applied** at task window edges — full task window used as-is. Post-hoc trimming may be applied later during signal inspection if warranted (not pre-emptively).
- **Limitation to note in thesis:** absence of hardware sync introduces small residual timing uncertainty (likely sub-second), accepted given task durations are much longer than expected clock drift/logging delay.

## PPG Processing (open decision point)

- Raw data: Shimmer sensor, ~128 Hz, tab-separated CSV. Columns include `S5E1C_Timestamp_Unix_CAL` (ms) and `S5E1C_PPG_A13_CAL` (mV, PPG signal).
- Proposal specifies HeartPy (v1.2.4) for peak detection and HRV/PRV feature extraction.
- **Currently evaluating NeuroKit2 as an alternative** (Makowski et al., 2021, Behavior Research Methods, https://doi.org/10.3758/s13428-020-01516-y) — broader HRV metric coverage, more actively maintained package. Comparison test pending on p20 Stroop window; decision not yet finalized.
- GSR (skin conductance/resistance) and accelerometer columns are present in the raw Shimmer data but NOT used, per current proposal scope (PPG only).

## Video Processing (per proposal, not yet started)

- OpenFace 2.0, following Giannakakis et al. (2017) pipeline.
- Features: blink rate, gaze direction, head pose.
- Blink rate is not a direct OpenFace output — must be derived from eye landmarks or AU45 (blink-related action unit).

## Output structure

- `data/processed/synced/{participant}_task_windows.csv` — per-task absolute + relative timestamps
- `data/processed/ppg/` — extracted HRV/PRV features per task
- `data/processed/video/` — extracted facial features per task

## Volgende stap

Dit schema eerst toepassen/verifiëren op **p20** (enige participant momenteel aanwezig in `data/raw/`), daarna dezelfde logica herhalen voor de overige participanten zodra hun data beschikbaar is. Verwacht: dezelfde blok-/ratingvolgorde en dezelfde kolomnamen, aangezien het dezelfde PsychoPy Builder-experimentdefinitie betreft — timing zal per participant verschillen.

## PPG Processing — DECIDED: NeuroKit2 (preliminary, based on p20)

- Raw data: Shimmer sensor, ~128 Hz, tab-separated CSV. Columns include `S5E1C_Timestamp_Unix_CAL` (ms) and `S5E1C_PPG_A13_CAL` (mV, PPG signal).
- **Decision: NeuroKit2 (`neurokit2` package) instead of HeartPy**, based on exploratory comparison on p20 (all 3 task windows). Citation: Makowski et al. (2021), Behavior Research Methods, https://doi.org/10.3758/s13428-020-01516-y.
- **Primary reason (content-based):** HeartPy's peak detector systematically misidentifies the dicrotic notch as a separate systolic peak on this low-amplitude Shimmer signal, causing heartbeat over-counting. Quantified on the Face Recall window (285s, ~352 expected beats at ~74 bpm): HeartPy detected 417 peaks (18% over-count) with 26% of intervals implausibly short (>150 bpm equivalent), vs. NeuroKit2's 351 peaks (near-exact match) with only 0.9% implausible intervals. This directly inflates HRV metrics sensitive to short-interval noise (notably RMSSD: HeartPy 281.8ms vs NeuroKit2's plausible value on this window).
- **Secondary factor (practical):** the pinned HeartPy version (1.2.4) also has two compatibility bugs on Python 3.12 + current NumPy (`time.clock()` removed since Python 3.8; `np.linspace` incompatibility in LF/HF calculation), making frequency-domain metrics (LF/HF) unavailable via HeartPy on this system regardless of the signal-quality issue above.
- Signal characteristics: low amplitude (~10-20 mV fluctuation on ~1400 mV baseline), confirmed via Welch PSD to contain a physiologically plausible heart rate peak (71-77 bpm, 47-57% in-band power) despite appearing "flat" under naive std/unique-value checks.
- GSR (skin conductance/resistance) and accelerometer columns are present in the raw Shimmer data but NOT used, per current proposal scope (PPG only).
- **Status: preliminary — based on single participant (p20).** To be re-verified once more participant data is available, to confirm this pattern (NeuroKit2 more robust on low-amplitude signal) generalizes rather than being specific to p20's signal characteristics.
- **Deviation from proposal to document:** proposal specifies HeartPy (v1.2.4); this deviation and its justification must be reflected in the thesis methodology section and Table 1 (software packages).

### Known limitations (to revisit before final analysis)
- LF/HF and other frequency-domain metrics are computed in absolute power units (ms²), not normalized. Task windows vary in duration (270–450s across Stroop/Arithmetic/Face Recall), which could confound absolute power values with task duration. To be checked: whether NeuroKit2's normalized variants (e.g. LFn/HFn) should be used instead, or duration should be added as a covariate.
- ~0.9% of detected peaks in exploratory testing (p20, Face Recall) appeared to be artifacts; `nk.signal_fixpeaks()` was not yet applied. To be added to the extraction pipeline before running across all participants, since RMSSD/pNN50 are sensitive to short-interval noise.

### Task-window trimming — DECIDED: 30s trim at task onset (own reasoned choice)

- **Decision:** the first 30 seconds of each task window are excluded before PPG/HRV processing (`scripts/ppg/03_extract_ppg_features.py --trim-seconds`, default 30). The task window itself is effectively shortened at the start (e.g. Stroop: 424.54s–763.65s becomes 454.54s–763.65s); the end boundary is unchanged.
- **Rationale (cardiac orienting response):** the onset of a new task/stimulus reliably evokes a transient cardiac orienting response — a brief, stereotyped heart-rate deceleration followed by recovery, distinct from the sustained physiological response to the task's actual cognitive/emotional demand (Sokolov's orienting-response framework; e.g. Graham & Clifton, 1966, *Psychological Bulletin*, on heart-rate orienting responses to stimulus onset). Including this transient in the HRV window for the *whole* task risks conflating a brief reflexive response to the task *starting* with the sustained physiological state *during* the task, which is what this project actually wants to relate to the self-reported stress_label.
- **Why 30s specifically, and why this is a reasoned choice rather than a copied standard value:** the orienting response itself typically resolves within several seconds to a few tens of seconds after stimulus onset in the general orienting-response literature. 30 seconds is chosen as a conservative round-number margin comfortably past that resolution window, rather than a value taken from a study with an identical task/setup (no such directly-matching precedent was used). **This is an own, argued choice**, not a value copied from a source using the same experimental setup — it should be reported as such in the thesis (i.e. "30s was chosen based on general orienting-response literature, as a conservative margin, not derived from an identical prior study").
- **Why a long stabilization period is not additionally needed:** each task is preceded by a rest/baseline period in the experiment structure (see task-window extraction logic above), so participants are not entering the task from an acute physiological transient of some *other* kind — the only expected transient specifically at task *onset* is the orienting response itself, which is what the 30s trim targets. A longer trim to allow for general "settling in" was therefore judged unnecessary on top of the orienting-response margin.
- **Implementation:** `--trim-seconds` is a CLI parameter (default 30, `0` reproduces the original untrimmed behavior exactly) so this can be revisited without a code change. Trimmed output is written to a separate file (`{participant}_ppg_features_trimmed.csv`) rather than overwriting the original, specifically so the trimmed and untrimmed versions can be compared side by side before deciding whether trimmed features become the pipeline's default going forward. As of this writing, the main orchestrator (`scripts/run_full_pipeline_all_participants.py`) still explicitly pins `--trim-seconds 0` and therefore still produces the untrimmed files — trimming has not yet been adopted as the pipeline's official behavior, pending review of the comparison (`scripts/evaluation/04_ppg_trim_comparison.py`, `results/eda_preliminary/ppg_trim_comparison.csv`).
- **Known trade-off to watch:** trimming shortens an already-short task window further, which mechanically reduces the number of detected beats available for HRV estimation. This is most likely to matter for Face Recall (the shortest task, ~220–285s across participants) — checked explicitly in the comparison output (`effective_duration_s` column, `ppg_trim_peaks_comparison.csv`) rather than assumed.

## Video Processing — DECIDED: OpenFace 2.0

- Tool comparison conducted on p20 (45s Stroop clip): OpenFace 2.0 vs. LibreFace vs. Py-Feat. Full results in results/video_exploration/tool_comparison.md.
- **Decision: OpenFace 2.0** (matches original proposal). Pre-built Windows binary works without requiring Visual Studio compilation.
- Rejected alternatives:
  - LibreFace: fails on Python 3.12/Windows — pinned mediapipe==0.10.5 has no Python 3.12 wheel, and newer mediapipe versions removed the API LibreFace depends on.
  - Py-Feat: ~169x slower than OpenFace on CPU (14.9s/frame vs. 0.088s/frame); extrapolated to ~84 hours per single task window. Not scalable to 76 participants × 3 tasks without GPU access.
- Blink rate derived from AU45_c, with a data-driven duration threshold (42.5–450.5ms) applied to exclude noise (single-frame artifacts) and long non-blink eye closures (e.g. looking down), based on gap analysis of event durations on p20's Stroop clip. To be re-validated once more participant data is available.

## Video Processing — DECIDED: OpenFace 2.0 (with blink-contamination filtering)

- Tool comparison conducted on p20 (45s Stroop clip): OpenFace 2.0 vs. LibreFace vs. Py-Feat. Full results in results/video_exploration/tool_comparison.md.
- **Decision: OpenFace 2.0** (matches original proposal). Pre-built Windows binary works without requiring Visual Studio compilation.
- Rejected alternatives:
  - LibreFace: fails on Python 3.12/Windows — pinned mediapipe==0.10.5 has no Python 3.12 wheel, and newer mediapipe versions removed the API LibreFace depends on.
  - Py-Feat: ~169x slower than OpenFace on CPU (14.9s/frame vs. 0.088s/frame); extrapolated to ~84 hours per single task window. Not scalable to 76 participants × 3 tasks without GPU access.

### Feature set
- Core features (per proposal, Giannakakis et al. 2017 pipeline): blink rate, gaze direction, head pose.
- Extended candidate set (own addition, exploratory): all 17 OpenFace Action Units, to let model-based feature importance (SHAP, permutation importance) determine relevance rather than pre-selecting AUs based on general stress literature.
- All features (except blink rate itself) aggregated per task window as mean, median, standard deviation, skewness, kurtosis (per proposal methodology).

### Blink-contamination discovery and correction (own finding)
Manual inspection of AU07_r peak frames (visual validation on p20) revealed these coincide almost entirely with blink onsets (AU45_c), not independent brow/eyelid tension. Quantified via frame-level overlap check: 100% (Stroop), 60-70% (Arithmetic), 100% (Face Recall) of top-20 AU07 peaks coincide with AU45_c activity within ±5 frames — well above chance baseline (35-52%). Likely explanation: AU07 and AU45 share overlapping musculature (orbicularis oculi), making them difficult to disambiguate for a frame-based classifier — an own finding based on manual + quantitative validation, not a documented issue in existing OpenFace literature.

Systematic validation across all 28 candidate features (17 AUs + 8 gaze + 3 pose) revealed the contamination was far broader than AU07 alone: 15/28 features flagged red (peak values disproportionately coinciding with blinks), including nearly all gaze and pose columns, likely due to landmark jitter during blinks propagating into all landmark-derived measurements.

**Correction applied:** frames within a blink episode (AU45_c==1, using the same duration threshold as blink rate: 42.5–450.5ms, to exclude noise and non-blink eye closures) plus a ±15 frame margin are excluded before computing any aggregate statistic (mean/median/sd/skewness/kurtosis) for any feature. Blink rate itself remains computed over the full, unfiltered signal.

**Filter tuning history:** an initial broad filter (all AU45_c==1 frames + ±5 frame margin, no duration threshold) achieved 0 red features but excluded up to 51.8% of frames (Arithmetic) — including many long non-blink eye closures (e.g., looking down), not just true blinks. Adding the duration threshold reduced exclusion substantially (22.0% for Arithmetic) but reintroduced contamination (9/28 features red again). Widening the margin to ±15 frames restored 0 red features while still excluding less data than the original broad filter for 2 of 3 tasks.

**Final exclusion rates (p20):** Stroop 48.4%, Arithmetic 40.7%, Face Recall 27.4%.

**Known limitations:**
- Even after correction, 23/28 features remain flagged yellow (mild residual association with blink activity persists, though extreme contamination is removed). Only 4 features are fully green.
- At exclusion rates above ~40% (Stroop, Arithmetic), higher-order statistics (skewness, kurtosis) should be interpreted with caution given the reduced effective sample of frames within the task window.
- Exclusion rates are based on p20's individual blink behavior and may vary substantially across participants. **When scaling to the full sample, exclusion rates per participant/task should be logged and flagged if they exceed a threshold (e.g. 60%)**, to avoid silently including unreliable features for high-blink-rate participants.
- This entire validation is based on a single participant (p20); the filter parameters (threshold, margin) should be spot-checked on a few additional participants once available, to confirm generalizability.

### Confidence filtering
- Frames with low OpenFace tracking confidence are subject to a separate exclusion threshold (to be finalized), based on manual inspection showing occasional brief tracking failures (e.g. hand near face, extreme eye closure) that should not contaminate feature aggregation.

Overweging: rust-baseline normalisatie (zoals in Delaney & Brodie 2000 en vergelijkbare literatuur) niet toegepast — huidige RQ's vergelijken taken onderling, niet taak-tegen-rust. Mogelijke uitbreiding indien tijd toelaat.

## Qualtrics/PSS processing

Bron: `data/raw/qualtrics/NERVE_background.csv` (Qualtrics-export, 3 header­rijen — variabelnaam/vraagtekst/ImportId — daarna 79 datarijen). Verwerkt door `scripts/qualtrics/01_process_pss_demographics.py` naar `data/processed/qualtrics/pss_demographics_cleaned.csv` (75 rijen).

### Niet-participant-rijen uitgesloten (4, niet 3)
Van de 79 datarijen zijn er **4** uitgesloten als niet-participant:

| Rij (participantNumber) | Finished | Progress | Duration | StartDate | Reden |
|---|---|---|---|---|---|
| *(leeg)* | 1 | 100% | 10.241 s (~2u51m) | 2026-02-26 04:15 | Zie hieronder — géén p1 |
| *(leeg)* | 0 | 6% | 10 s | 2026-02-26 04:30 | Afgebroken poging |
| `wrong` | 1 | 100% | — | 2026-03-05 | Leeg/testinvoer (Age=1, implausibel) |
| `p` | 0 | 26% | — | 2026-02-26 09:04 | Afgebroken poging |

Uitsluitingsregel in code: `participantNumber` is leeg, OF gelijk aan `'wrong'` (case-insensitive), OF `Finished == 0`.

**Belangrijk:** een eerdere analysestap identificeerde slechts 3 "niet-participant-rijen" (via `Finished==0` OF `id=='wrong'`) en nam daarbij aan dat de resterende lege-ID-rij (Finished=1, 100%) participant **p1** was, die vergeten was zijn/haar ID in te vullen. Verificatie tegen de OBS-log wees dit af: die rij dateert van **2026-02-26**, drie weken vóór p1's daadwerkelijke sessie (**2026-03-18, 13:57:23**), en de duur (~2u51m) is fysiek onwaarschijnlijk voor een achtergrondvragenlijst. Vermoedelijke verklaring: een pilot-/testinvoer van de onderzoeker vóór aanvang van de dataverzameling (dezelfde dag als de eerste echte respons, `p2`). Deze rij is daarom als 4e niet-participant-rij uitgesloten, **niet** toegeschreven aan p1 of enige andere deelnemer.

**Gevolg: p1 ontbreekt volledig in de Qualtrics-data.** Van de verwachte 76 deelnemers (p1–p76) zijn er **75** terug te vinden (p2–p76); voor p1 is geen enkele Qualtrics-respons betrouwbaar te koppelen. `pss_demographics_cleaned.csv` bevat daarom bewust **geen rij voor p1** — dit moet bij het samenvoegen met de rest van de dataset met een **LEFT/OUTER join** tegen de volledige p1–p76 participantenlijst gebeuren, zodat p1 als `NaN` verschijnt in age/gender/pss_total_score/etc., in plaats van stilzwijgend te worden overgeslagen of ingevuld.

Consistentiecheck: na uitsluiting van deze 4e rij is de gender-verdeling **49 vrouw / 26 man** (in plaats van de eerder gerapporteerde 50/26) — precies 1 vrouw minder dan de proposal's bekende steekproefsamenstelling (50 vrouw/26 man). Dit is intern consistent met p1 ontbrekend (als p1 vrouw is, wat niet elders bevestigd is, verklaart dit exact het verschil) en wordt hier gemeld als bevestiging, niet als nieuw probleem.

### Participant-ID-normalisatie
Ruwe `participantNumber`-waarden waren inconsistent: hoofdletter-varianten (`P6`, `P11`, ...), losse cijferreeksen zonder `p`-prefix (`14`, `15`, `34`, `36`, `59`), typo's met verkeerde letter (`b53`→p53, `B58`→p58, `u76`→p76), en een trailing newline (`p50\n`). Normalisatie: strip whitespace, lowercase, dan `"p" + laatste cijferreeks in de string` (regex `(\d+)$`). Dit lost alle bovenstaande gevallen eenduidig op zonder een hardcoded mapping per typo.

### PSS-10 offset-fout
Alle 10 `Perceived_stress_1`–`_10`-kolommen zijn intern gecodeerd als **1–5**, terwijl de vraagtekst het antwoordschaal-label 0–4 (Never–Very Often) toont — bevestigd doordat elk van de 10 kolommen exact min=1/max=5 heeft, zonder uitzondering (dus een systematische Qualtrics-coderingsfout, geen toevallig ontbrekende "0"-respons). Correctie: **eerst 1 aftrekken van elk item**, dan pas de standaard PSS-10-scoring toepassen: items 4, 5, 7, 8 (1-indexed) zijn reverse-scored (`4 - gecorrigeerde_waarde`), overige items ongewijzigd, som van alle 10 = `pss_total_score` (geldig bereik 0–40; waargenomen bereik in de data: 6–34).

### Gender-hercodering
`Gender` is intern numeriek gecodeerd zonder label in de header. Bevestigd via aantal-matching tegen de proposal (50 vrouw/26 man; zie hierboven voor de nuance na uitsluiting van de 4e niet-participant-rij): **1.0 = female, 2.0 = male**.

### Output
`data/processed/qualtrics/pss_demographics_cleaned.csv` — 75 rijen, kolommen: `participant` (genormaliseerd, p2–p76), `age`, `gender` (female/male), `pss_total_score`, plus overige demografische/gezondheidskolommen (`ed_level`, `handedness`, `chronic_disease`, `taking_medication`, `caffeinated_drinks`, `smoking`, `drinking`, `vision`, `sleep`, `physical_act`) — deze laatste zijn nog **numeriek/ongelabeld** overgenomen uit Qualtrics (geen antwoordlabels beschikbaar in de header) en moeten vóór analyse/rapportage nog gedecodeerd worden.

## Video Processing — DECIDED: session-wide OpenFace processing (2026-09-07), superseding per-task clips

**Decision:** `scripts/video/02_extract_video_features.py` now runs OpenFace's `FeatureExtraction.exe`
**once per participant on the full, uncut session video**, then slices the 3 task windows out of that
single long output table by timestamp — replacing the original approach of cutting 3 separate
ffmpeg clips and running OpenFace on each independently.

**Reason:** direct test on p20 (`results/video_exploration/session_wide_vs_per_task_normalization_p20.md`)
found that OpenFace's per-recording AU normalization is sensitive to how much surrounding video
context it sees. On IDENTICAL physical frames, session-wide vs. per-clip processing gave AU04_r/AU07_r
values differing by a mean 7.1% (max 25.7%, Face Recall) — and, more importantly, the AU45_c BINARY
blink classification itself differed by ~34% (4639 vs. 3067 positive frames on p20's Stroop footage).
Since blink classification drives this pipeline's exclusion filter directly, that is not a cosmetic
amplitude difference — it changes which frames get excluded from every downstream feature.

**Cost:** session-wide processing is slower overall, since OpenFace now also processes the
rest/breathing-baseline/instruction footage between tasks that used to be skipped entirely by cutting
clips first. On p20: ~2.16× the combined runtime of the 3 task clips alone (~39min of session footage
processed vs. ~18min of task footage previously). Accepted as the price of removing the
normalization-context confound above. OpenFace output is cached per participant
(`data/processed/video/openface_raw_session_wide/{participant}_full_session/`) so this cost is paid
once, not on every re-run.

**Blink-threshold pooling stays restricted to the 3 task windows**, not the full continuous session
(which also contains long rest/instruction segments with plausibly different blink behavior) — this
was a deliberate choice to keep old-vs-new a controlled comparison (same frames pooled, different
processing context) rather than also changing which frames get pooled. See the "Pooling note" in
`02_extract_video_features.py`'s docstring.

**Result on p20** (full comparison: `results/video_exploration/session_wide_extraction_validation_p20.md`):
- New adaptive blink threshold: max_duration_ms dropped from 680.0ms (old, per-task) to **391.0ms**
  (new, session-wide) — a ~42.5% drop, despite the pooled blink-event count being nearly unchanged
  (838 vs. 832). The underlying AU45 "on"-run duration *distribution*, not just individual frame
  values, measurably changed shape under session-wide processing — not anticipated going in.
- AU04 remains highest during Arithmetic — the core finding this was testing for is unaffected.
- Feature-validation check (`04_validate_features_blink_overlap.py --post-filter`) re-run: ROOD count
  unchanged at 4, but GEEL dropped 20→14 and GROEN tripled 3→9 — a net improvement (9 features
  improved, 3 worsened, 2 of the "worsened" ones newly crossing into ROOD by a small margin: AU01_r,
  gaze_angle_y). `pose_Rx` and `gaze_1_z` remain ROOD in both versions.
- **Correction to prior documentation:** the "0 ROOD" result referenced earlier in this file (Filter
  tuning history, above) describes an earlier, since-superseded fixed-threshold tuning stage. The
  actual current per-task baseline this session-wide switch should be compared against is **4 ROOD /
  20 GEEL / 3 GROEN** (`results/video_exploration/p20_feature_validation_post_filter.csv`, generated
  2026-09-01 alongside the current adaptive-per-participant-threshold pipeline), not 0 ROOD. Any future
  reference to "0 red" as the video-pipeline baseline should be corrected to this figure instead.

**Not yet done:** this switch has only been run/validated on p20. Re-running the full extraction
pipeline for the other participants already processed under the old per-clip approach (see
`data/processed/video/openface_raw/` for the old per-clip cache, kept for reference/backup rather than
deleted) is a separate, substantial batch job (each participant's full session takes ~3-4h of OpenFace
processing) not yet undertaken.

**Update (2026-09-07): now underway** — see the two DECIDED sections directly below (camera
calibration, feature exclusion) for the final pipeline configuration, and the "Full-batch reprocessing"
section further down for status/progress of applying it to all participants.

## Video Processing — DECIDED: explicit camera calibration (2026-09-07)

**Decision:** every OpenFace run now passes explicit camera intrinsics for the OBSBOT Tiny 2 webcam
used to record all participants (fixed 1.2x zoom, 1920x1080): `-fx 1637 -fy 1637 -cx 960 -cy 540`,
instead of relying on OpenFace's own focal-length estimate from image size (its fallback when these
are omitted). Same value for every participant, since they were all recorded on the same camera at the
same fixed zoom — this is a per-camera constant, not something derived per participant.

**Reason:** a direct calibrated-vs-uncalibrated comparison on p20, both run session-wide otherwise
identically (`results/video_exploration/camera_calibration_comparison_p20.md`), found a **systematic,
same-direction correction across all 3 tasks**: pose_pitch_mean +14% to +24%, and vertical gaze
(gaze_0_y/gaze_1_y/gaze_angle_y mean) +7% to +13%. The consistency of direction and magnitude across
Stroop/Arithmetic/Face Recall (not random noise) indicates OpenFace's own focal-length estimate was
introducing a genuine, non-trivial geometric bias into absolute pose/gaze values.

**What it does NOT fix:** the blink-related feature-contamination problem (see the exclusion decision
directly below). `pose_Rx`'s contamination ratio in the post-filter validation check was **identical to
many decimal places** with vs. without calibration — camera geometry and blink-related landmark jitter
are unrelated problems, confirmed empirically rather than assumed.

**Caveat for relative (within-participant, task-vs-task) comparisons**, which is this project's primary
use of these features: since the bias runs in the same direction and is similar in magnitude across all
3 tasks for a given participant, it should largely cancel out when comparing tasks against each other.
Calibration mainly matters for absolute-value correctness/reportability (e.g. citing a mean head-pitch
in degrees, or comparing against other studies/cameras), not for the Stroop-vs-Arithmetic-vs-Face-Recall
comparisons this project centers on.

**Implementation:** `CAMERA_FX`/`CAMERA_FY`/`CAMERA_CX`/`CAMERA_CY` constants in
`02_extract_video_features.py`, passed unconditionally in every `run_openface()` call. A cache-staleness
check (`camera_params_used.json` sidecar per cached OpenFace output, compared against the current
constants before reuse) prevents old, pre-calibration cached runs from being silently reused as if they
were calibrated — a mismatched or missing marker forces a re-run rather than assuming compatibility.

## Video Processing — DECIDED: exclude pose_Rx and gaze_1_z from the feature set (2026-09-07)

**Decision:** `pose_Rx` (head pitch) and `gaze_1_z` are dropped from the aggregated per-task feature
table (`compute_task_features()` in `02_extract_video_features.py`) going forward. They remain fully
present in the raw cached OpenFace output (nothing is deleted at that stage) — only the aggregated
mean/std columns that would otherwise feed the master feature table are omitted.

**Reason:** both were flagged **ROOD** (severe blink contamination, overlap ratio ~1.9-2.1x chance
baseline) in `04_validate_features_blink_overlap.py --post-filter`, and stayed ROOD after BOTH of the
two independent fixes attempted:
1. Adaptive per-participant blink-duration filtering (±15 frames around every detected blink,
   Tukey-derived duration threshold) — see the blink-filtering DECIDED section above.
2. Explicit camera calibration (directly above) — ruled out as the cause since it left `pose_Rx`'s
   contamination ratio unchanged to many decimal places.

Two unrelated, independently-motivated fixes both failing to move these two features is reasonably
strong evidence the contamination isn't caused by either camera geometry or blink-duration
mis-thresholding, but by something more structural (most likely residual landmark-tracking jitter
immediately around eye closure that a temporal exclusion window doesn't fully capture). Rather than
keep publishing known-unreliable values in the final feature set, or invest further tuning effort
against two independent negative results, they are excluded outright.

**Scope of the exclusion:** only `pose_Rx`/`pose_pitch` and `gaze_1_z` — the other 2 pose columns
(yaw, roll) and 7 gaze columns are unaffected and remain in the feature table as before. `pose_Rx` and
`gaze_1_z` are still computed and cached in the raw per-participant OpenFace output, so a future
different remediation (if found) would not require re-running OpenFace, only re-aggregating.

## Full-batch reprocessing with the final pipeline configuration (session-wide + calibration + exclusions)

Status as of 2026-09-07: this final configuration (session-wide processing, mandatory camera
calibration, pose_Rx/gaze_1_z excluded) is being applied to (a) the 13 participants already processed
under earlier pipeline versions, requiring reprocessing, and (b) all remaining participants with video
data not yet processed at all. **p1 is excluded from this batch** (per instruction) — consistent with
p1's PPG recording also being corrupted/truncated (see PPG Processing section above), p1 is being
treated as an unusable participant throughout, not just for PPG.

p20 required no OpenFace re-run (the already-computed calibrated session-wide run from the camera-
calibration test above was moved into the production cache path and re-validated via the camera-params
marker) — features recomputed in ~1.6 min. All other in-scope participants require a fresh ~3-4h
session-wide OpenFace run each. See `results/pipeline_status_overview.md` for live progress.

### Two bugs found and fixed (2026-09-09) while spot-checking p3-p7

1. **`EXCLUDED_CONTAMINATED_FEATURES` was not applied in the validation script.** The pose_Rx/gaze_1_z
   exclusion (above) only touched `02_extract_video_features.py`'s aggregation step;
   `04_validate_features_blink_overlap.py` had its own separate hardcoded `GAZE_COLS`/`POSE_COLS` lists
   and kept validating all 28 features, not the 26 that actually ship. Fixed by moving
   `EXCLUDED_CONTAMINATED_FEATURES` into the shared `blink_utils.py` module and having both scripts
   import it from there, so they cannot silently diverge again.
2. **The orchestrator's validation call used the wrong (stale) cache directory.**
   `run_full_pipeline_all_participants.py` called `04_validate_features_blink_overlap.py --post-filter`
   without `--openface-raw-dir`, so it silently fell back to the script's default
   (`data/processed/video/openface_raw`, the OLD per-clip cache) instead of
   `openface_raw_session_wide`. For p3-p7 this meant the validation numbers written into
   `pipeline_status_overview.md` were computed against pre-calibration, pre-session-wide data —
   unrelated to the actual pipeline output. Fixed by passing the correct dir explicitly.

### Generalization finding: the fixed blink-filter margin does NOT generalize well beyond p20

After both bugs above were fixed and p20/p3-p7 were re-validated on a like-for-like basis (26 features,
correct cache dir), the contrast is stark:

| Participant | ROOD (of 26 non-trivial features) |
|---|---|
| p20 | 2 (8%) |
| p3 | 20 (80%) |
| p4 | 17 (68%) |
| p5 | 19 (76%) |
| p6 | 19 (76%) |
| p7 | 18 (72%) |

This gap is too large to be ordinary individual blink-behavior variance. The likely cause: the
blink-filter margin (±15 frames, `BLINK_FILTER_MARGIN_FRAMES`) and the camera-calibration decision were
both derived and validated **on p20 alone** — a risk this file already flagged before more participants
were available ("filter parameters should be spot-checked on a few additional participants ... to
confirm generalizability", see the blink-contamination filter-tuning history above). That check has now
happened, and the parameters tuned on p20 do not carry over: most other participants retain far more
residual blink contamination than p20 did. **This needs follow-up before the current 26-feature set is
treated as final for the full cohort** — e.g. re-deriving the exclusion margin from pooled multi-
participant data (the same fix already applied to the blink-DURATION threshold, which IS adaptive per
participant) rather than a fixed constant. Not addressed yet; the full-cohort batch is proceeding in the
meantime so processing isn't blocked on this, but the feature set (beyond the already-excluded
pose_Rx/gaze_1_z) should be re-audited once more participants are available rather than assumed clean.

## Video Processing — EXPLORED AND REJECTED (for now): OpenFace 3.0 (2026-09-12)

**Status: exploratory only, tested in an isolated venv (`scratchpad_video_tools/openface3_venv/`),
never touched the production pipeline.** OpenFace 3.0 (Hu et al., 2025, arXiv:2506.02891;
`openface-test` v0.1.26 on pip) was tested as a potential replacement for OpenFace 2.0, motivated by its
claimed ~2x CPU speedup and simpler pip-based install.

**Verdict: not adopted, documented as future work instead.** Three independently confirmed, disqualifying
findings, cross-checked against the project's own public GitHub issue tracker (not just our own testing):
1. **No head pose output at all** — confirmed absent from the source code; no GitHub issue, roadmap, or
   commit suggests this is planned. A PnP-based workaround (using the package's own 98-point landmarks,
   the same approach OpenFace 2.0 uses internally) was actually implemented and tested, but produced
   physically-impossible angles for a frontal face (yaw≈176° instead of ≈0°) — a fixable but non-trivial
   sign/coordinate-convention bug, not a quick fix.
2. **AU output is only 8 unlabeled channels** (vs. this project's 17 named AUs from OpenFace 2.0, and vs.
   12/27 claimed in the paper itself for different training configs) — confirmed via the model's own
   default constructor (`au_numbers=8`). A targeted test (3 images from p20, selected using OpenFace
   2.0's own AU output as a guide: neutral/smile/brow-tension) tentatively identified one channel as
   smile-related, but **found no channel responding to brow tension — AU04 remains unidentified**, which
   directly blocks the AU04/glasses-bias comparison this exploration was partly meant to answer.
3. **The `detect-video` CLI command is broken** (confirmed via source-code trace: it passes an in-memory
   frame array to a function that only accepts file paths) — and this is a **publicly known, open GitHub
   issue since November 2025 with no maintainer response**, on a repository whose last code commit
   predates that report (June 2025) — i.e., not a fluke of our environment, and not something likely to
   be fixed on a timeline relevant to this thesis.

Full writeup, all three workaround attempts, and the speed benchmark: `results/video_exploration/
openface3_comparison.md`. Bottom line: OpenFace 2.0 remains the production choice; OpenFace 3.0 is worth
citing as future work in the thesis discussion/limitations section once head pose, the video CLI, and an
AU name mapping are addressed upstream.

## Data integrity finding: p66/p68 raw Shimmer files are byte-identical (2026-09-09)

**Confirmed:** `data/raw/ppg/P66_Shimmer_firstMeasurement_66.csv` and
`data/raw/ppg/P68_Shimmer_secondMeasurement_68.csv` are byte-for-byte identical (same file size 80,585,789
bytes, same MD5 hash `3bf0af7f50683259c88e2100bf9f9edc`, same modification date). Not a coincidence or a
summary-statistic artifact — verified at the raw-byte level.

**Root cause identified (not just "a duplicate"):** the Shimmer device recorded continuously across BOTH
p66's and p68's sessions without being stopped/restarted in between. The shared file spans
2026-05-21 10:44:17 to 12:20:06 (95.8 min) — p66's true session (PsychoPy expStart 10:44:30, OBS
10:44:09-11:22:57) is the FIRST ~39 min of this file; p68's true session (PsychoPy expStart 11:41:48, OBS
11:41:34-12:20:15) is the LAST ~39 min; there's an ~18.6 min gap between them (11:23-11:41) where the
sensor kept recording but neither participant's task was running. This single continuous export was then
(erroneously) saved into both participants' raw-data slots instead of being split per session.

**Consequence for existing processing — checked directly, not assumed:** because
`scripts/ppg/03_extract_ppg_features.py` (and now `scripts/gsr/01_extract_gsr_features.py`) slice by
ABSOLUTE Unix timestamp (each participant's own `task_windows.csv`, computed independently from their own
PsychoPy `expStart`/OBS log), and both participants' true windows fall inside this one shared file, the
already-computed PPG features for p66 and p68 are genuinely participant-specific, not duplicates of each
other: p66's HRV_MeanNN ≈ 907-935ms (~64-66 bpm) vs. p68's ≈ 671-701ms (~86-89 bpm) across all 3 tasks —
clearly two different people's physiology, confirming the pipeline already handles this correctly by
construction.

**Recommendation: do NOT exclude either participant.** The absolute-timestamp slicing already isolates
each participant's genuine data correctly. However:
- This is fragile and easy to break: any future code path that treats "the whole raw file = one
  participant's session" (rather than slicing by absolute task-window time) would silently mix p66's and
  p68's data. Flagging this explicitly so it isn't lost.
- The new GSR pipeline (see `results/gsr_exploration/gsr_pipeline_proposal.md`) uses the same
  absolute-timestamp slicing approach and was spot-checked to behave the same way, but this should be kept
  in mind for any OTHER future signal added from these raw files.
- Full investigation and the p66/p68 GSR-specific check: see `results/gsr_exploration/gsr_pipeline_proposal.md`.

## GSR/EDA literature update (2026-09-12)

Boucsein (2012) remains the standard reference for EDA terminology (SCL/SCR, orienting response) used
throughout the GSR pipeline notes above. Supplemented with two more recent, directly relevant sources
(verified via web search, not from memory, to avoid citing details incorrectly):

- Stržinar, Ž., Sanchis, A., Ledezma, A., Sipele, O., Pregelj, B., & Škrjanc, I. (2023). Stress Detection
  Using Frequency Spectrum Analysis of Wrist-Measured Electrodermal Activity. *Sensors*, 23(2), 963.
  https://doi.org/10.3390/s23020963. Frequency-spectrum EDA features for stress classification (WESAD
  dataset) — relevant as a candidate additional feature family (spectral-band EDA features) beyond the
  current SCL/SCR-morphology feature set, if the current feature set is later found insufficiently
  discriminative; not yet implemented here.
- Pataca, A. O., Zdravevski, E., Coelho, P. J., Garcia, N. M., Deryuck, M., Albuquerque, C., & Pires, I. M.
  (2025). Use of machine learning for predicting stress episodes based on wearable sensor data: A systematic
  review. *Computers in Biology and Medicine*, 198, 111166. https://doi.org/10.1016/j.compbiomed.2025.111166.
  PRISMA-based review (Jan 2010 - Jun 2025); confirms EDA, HRV, and PPG as the most-used signal types for
  wearable stress detection and reports RF/DNN models reaching up to ~99% accuracy in the reviewed studies,
  alongside small-dataset-size and lack-of-standard-protocol as recurring limitations across the field —
  both directly relevant framing for this thesis's own sample size and protocol.
- Tsirmpas, C., Konstantopoulos, S., Andrikopoulos, D., Kyriakouli, K., & Fatouros, P. (2025).
  Transformer-Based Decomposition of Electrodermal Activity for Real-World Mental Health Applications.
  *Sensors*, 25(14), 4406. https://doi.org/10.3390/s25144406. **Grounds the SCR-over-detection
  methodological check below**: this paper directly quantifies cvxEDA's and Ledalab's tendency to fit false
  phasic peaks on real-world wearable EDA, particularly around abrupt tonic-level (SCL) changes and
  low-amplitude signal stretches — reporting, e.g., cvxEDA peak amplitudes averaging 0.661 µS vs. Ledalab's
  0.196 µS on the same real-world data, and describing cvxEDA's global optimization as more prone than
  Ledalab's per-peak optimization to "fit peaks that are noise artifacts" when tonic/phasic separation is
  ambiguous. This is the EDA-domain analogue of the HeartPy dicrotic-notch over-counting problem found
  earlier in the PPG pipeline (see PPG Processing section above) — same underlying risk (a peak/response
  detector over-fitting a low-amplitude or ambiguous signal), different signal modality.

See `results/gsr_exploration/gsr_pipeline_proposal.md` for the scaled-up (16-participant) SCR-plausibility
and tonic/phasic-decomposition validation this literature motivated, and its generalization findings.

**Update (2026-09-12): p10/p12/p60 confirmed excluded from GSR analysis.** Following the generalization
check above, three NeuroKit2 phasic-decomposition alternatives to cvxEDA (highpass, smoothmedian, sparseEDA
— NOT literal Ledalab, which is a standalone MATLAB toolbox unavailable in this environment) were tested on
exactly these 3 participants' flagged task windows. None resolved the issue; the linear-filter alternatives
(highpass/smoothmedian) made it markedly worse (10-30x more detected "peaks", all at implausibly tiny
amplitude — clear noise-fitting). Convergent failure across three structurally different decomposition
methods indicates the problem is in the underlying signal for these 3 participants, not the decomposition
method choice. **p10, p12, and p60 are now a documented GSR data-quality exclusion category** (~4% of the
76-participant cohort), analogous to the PPG failure categories above (corrupt/non-overlapping raw
recordings) — same treatment (permanent, signal-level, not a fixable processing choice), different signal
modality. Full comparison numbers: `results/gsr_exploration/gsr_pipeline_proposal.md`.

## Incident: orchestrator killed by an unattended-crash during Modern Standby (2026-09-11), and the watchdog/crash-recovery setup that followed

**What happened:** the batch orchestrator (`scripts/run_full_pipeline_all_participants.py`) was running
the full-cohort video pipeline (Phase B, mid-way through p18's session-wide OpenFace extraction) when the
machine became unresponsive and rebooted on its own. Windows Event Viewer (System log) shows the actual
sequence:
- `19:19:35` — power source changed (AC ↔ battery)
- `19:22:42` — system entered **Modern Standby**, reason logged as `Idle Timeout`
- `19:22:49` — standby connectivity disconnected; no further activity logged for ~53 minutes
- `20:16:05` — **Kernel-Power Event ID 41 (Critical)**: *"the system rebooted without cleanly shutting
  down first — possibly because it stopped responding, crashed, or lost power"*; EventLog ID 6008 confirms
  the previous shutdown (at 18:59:27) was unexpected.
- No BugCheck/dump event (ID 1001) was found, and no Windows-Update-triggered restart events were found in
  the same window — ruling out a driver BSOD or an update-forced reboot as the cause.

**Root cause:** `powercfg /query` showed the "slaapstand na" (STANDBYIDLE) timer was correctly set to
**never** for AC power, but was still at the Windows default of **180 seconds (3 minutes)** for **battery
(DC) power** — this had never been explicitly disabled, only the AC side had been. The elapsed time between
the power-source change (19:19:35) and the standby entry (19:22:42) is ~3 minutes, matching the DC timer
almost exactly. Working hypothesis: the AC adapter was briefly disconnected; 3 minutes later the DC idle
timer put the machine into Modern Standby; something failed during that standby window (most likely the
battery running out while suspended, given the absence of any crash-dump event) and the system power-cycled
uncleanly, killing the orchestrator (and everything else) with nothing left running to restart it.

**Fix applied (2026-09-11, via `powercfg /setdcvalueindex`):** `STANDBYIDLE` and `HIBERNATEIDLE` on DC
(battery) power are now both set to **0 (never)**, matching AC. Verified via `powercfg /query SCHEME_CURRENT
SUB_SLEEP` post-change. Note: this ASUS Modern-Standby device does not expose a separate "lid close action"
setting via `powercfg` (`SUB_BUTTONS` only shows the Start-menu power button action) — lid handling on
Modern Standby systems is not independently configurable the way it is on classic ACPI-sleep systems, and
was in any case not the trigger here (the log shows `Idle Timeout`, not a lid-close event, as the reason
the system entered standby).

**Watchdog / crash-recovery setup (replaces the earlier ad-hoc approach):** the watchdog that "should" have
caught this earlier was a loose, interactively-started PowerShell process — it was never saved to disk, so
it died in the same crash and left no trace to investigate or restart from. This has been replaced with a
proper, persisted setup:

- **`scripts/watchdog.ps1`** — checks `results/orchestrator.pid` against the live process list (matching
  both PID *and* command line, so a reused PID can't produce a false "still running" read); if the
  orchestrator isn't actually running, starts it fresh
  (`python scripts/run_full_pipeline_all_participants.py --exclude p1` — `p1` stays excluded per the
  full-batch-reprocessing decision above) and records the new PID. Logs every check/action to
  `results/watchdog_log.txt`. Idempotent and safe to run concurrently/repeatedly.
- **Registered in Windows Task Scheduler** as `ThesisPipelineWatchdog_Recurring` — runs `watchdog.ps1`
  every 10 minutes, `StartWhenAvailable` enabled (so a run missed while the machine was off/asleep fires as
  soon as Task Scheduler is available again, rather than waiting for the next 10-minute boundary), created
  via `Register-ScheduledTask` (PowerShell's `ScheduledTasks` module), not `schtasks.exe` (which failed here
  on quoting the space-containing repo path — `Register-ScheduledTask` avoids that entirely by building the
  task via objects instead of a raw command-line string).
- **Known limitation — no true `AtStartup`/`AtLogOn` boot trigger:** registering those trigger types
  requires an elevated (Administrator) PowerShell session; this setup was done non-elevated
  (`whoami` confirmed not running as Administrator), and both `AtStartup` and `AtLogOn` trigger registration
  failed with Access Denied under Register-ScheduledTask in that context. The registered recurring task's
  `LogonType` is therefore `Interactive` (runs only while the user has a logged-in session) rather than
  running unattended before/without login. **Practical effect:** after a future crash+reboot, the
  orchestrator resumes automatically as soon as the user logs back in (the missed recurring trigger fires
  immediately thanks to `StartWhenAvailable`) — it does **not** resume on its own if the machine reboots and
  sits at the login screen with nobody logging in. To close that last gap, run this from an **elevated**
  PowerShell session once (adds a genuine `AtStartup` trigger to the existing task set):
  ```powershell
  $wdPath = "D:\Data Science and Society\Block 3\Thesis\scripts\watchdog.ps1"
  $action = New-ScheduledTaskAction -Execute "powershell.exe" `
      -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wdPath`""
  $trigger = New-ScheduledTaskTrigger -AtStartup
  $trigger.Delay = "PT2M"
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName "ThesisPipelineWatchdog_OnStart" -Action $action -Trigger $trigger -Settings $settings
  ```
- **To check status manually at any time:** `Get-ScheduledTask -TaskName "ThesisPipelineWatchdog_*"` and
  `Get-Content results\watchdog_log.txt -Tail 20`.

**Recovery action taken:** `scripts/watchdog.ps1` was run once manually to perform the actual restart (PID
recorded in `results/orchestrator.pid`); the orchestrator resumed Phase A from p2 (re-running the fast
per-participant steps is cheap/idempotent) and will reach Phase B / p18 again, where the video pipeline's
existing completeness check (`load_or_run_openface_session_wide()` in `02_extract_video_features.py`)
automatically detects and discards p18's truncated OpenFace cache from the interrupted run rather than
silently treating it as complete — no manual cache cleanup was needed for that reason.

**Outcome (2026-09-12 09:38):** this restarted run completed the full batch successfully overnight, with no
further crashes — Phase B finished p18 (23:40), p19 (01:40), p20 (01:41), p21 (05:19), p22 (07:27), p23
(09:38, run then hit its watch-deadline and exited normally). Verified directly against the output files
(not just the log): all of p18-p23's `data/processed/video/{p}_video_features.csv` and
`data/processed/features/{p}_master_features.csv` exist with `video_pending=False` and the full 155-column
schema. **All 22 video participants (p2-p23) now have complete master tables.**

### Second issue found and fixed the same day: watchdog couldn't tell "finished" from "crashed"

The orchestrator's own Phase C ("watch `data/raw/video/` for new files") only runs until `--watch-hours`
elapses (default 5.5h, counted from process start, including Phases A/B), after which it logs "Full
pipeline run complete" and **exits normally, with return code 0**. To `watchdog.ps1`'s liveness check (is
there a live process matching the recorded PID?), a clean, successful exit and a crash are indistinguishable
— both mean "no matching process is running". Consequence, observed directly in `results/watchdog_log.txt`:
the overnight run legitimately finished and exited at 09:38:32; the watchdog's next 10-minute check (09:42)
correctly found nothing running and "recovered" it by restarting the **entire batch from scratch** — Phase A
for all 75 participants again, then Phase B for all 22 video participants again. Not data-destructive
(Phase A is idempotent; Phase B's OpenFace cache made this second pass fast, ~1min/participant instead of
hours) but wasteful and, left alone, would repeat indefinitely (finish → exit → watchdog restarts → watch
5.5h → exit → restart → ...) every ~5.5h forever.

**Fix:** `watchdog.ps1` now starts the orchestrator with `--watch-hours 8760` (~1 year) instead of the
default. This doesn't change what Phase C *does* (still just polls `data/raw/video/` for new files, sleeping
`--watch-interval-s` between checks, per the module docstring) — it only pushes the point where the
orchestrator would otherwise give up and exit far into the future, so in practice it now stays alive
indefinitely once the initial batch is done, and the watchdog's restart logic is reserved for genuine
crashes rather than ever needing to distinguish "done" from "dead". Applied by killing the then-current
(already-redundant, still mid-recompute) process and re-running `watchdog.ps1` once by hand to relaunch it
with the corrected argument; verified via `Get-CimInstance Win32_Process` that the new process's command
line actually includes `--watch-hours 8760`.