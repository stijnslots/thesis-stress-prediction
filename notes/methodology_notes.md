# Methodology notes: taakvenster-indeling (PsychoPy)

Status: vastgelegd op basis van participant **p20** (bestand `p20_test builder exp_2026-03-18_13h57.33.295.csv`/`.log`). Definitie is generiek geformuleerd (kolomnamen, niet hardcoded tijdstippen) zodat hij per participant herhaald kan worden; nog niet geverifieerd op de rest van de steekproef.

## Experimentstructuur (zoals die uit de PsychoPy-data blijkt)

Het experiment bevat **4 blokken** met trials, elk gevolgd door een rating-routine (`Rating1`–`Rating4`, identieke template: polygon + text + `key_resp_1x`, single-keypress antwoord):

1. `trials_stroop_pr` (oefenblok, 3 trials) → `trials_stroop` (hoofdtaak, 144 trials) → **Rating1**
2. `trials_faceMemory` (encodeerfase, 64 trials, geen response-kolom/geen reactietijd-druk) → **Rating2**
3. `trials_arithmetic_pr` (oefenblok, 3 trials) → `trials_arithmetic` (hoofdtaak, 50 trials) → **Rating3**
4. `trials_faceRecall` (hoofdtaak, 64 trials, mét response `key_resp_8`) → **Rating4**

Elke Rating start binnen enkele milliseconden na de laatste `.stopped`-timestamp van het voorgaande blok (geverifieerd, zie vorige analyse), dus de koppeling blok→rating is eenduidig en niet toevallig.

## Beslissing: 3 taakvensters, encodeerfase uitgesloten

Voor het stress-detectiemodel worden alleen **3 taken** gebruikt: **Stroop, Arithmetic, Face Recall**. De FaceMemory-encodeerfase (blok 2) wordt bewust **niet** gebruikt als taakvenster, en `Rating2` (de rating na encoding) wordt niet gebruikt als target-label.

**Onderbouwing:** de encodeerfase kent geen actieve respons en geen tijdsdruk (participant bekijkt passief gezichten, er is geen `key_resp`-kolom in `trials_faceMemory` die op reactie/prestatie wijst zoals bij de andere drie taken). Cognitieve belasting is daardoor kwalitatief anders dan bij Stroop (interferentie onder tijdsdruk), Arithmetic (rekenen onder tijdsdruk) en Face Recall (herkenning met respons). De encodeerfase is wel functioneel noodzakelijk, zonder encoderen is er niets om bij Face Recall te herkennen, maar levert zelf geen vergelijkbaar stressvenster op.

Dit betekent: bij N participanten zijn er 4×N ratingmomenten in de ruwe data, maar slechts **3×N** worden gebruikt als target (bij N=76 dus 228 observaties, conform proposal).

## Definitief knipschema (per taak)

| Taak | Trial-loop (kolomprefix) | Practice-blok (genegeerd) | Start-marker | Eind-marker (laatste trial) | Target-rating |
|---|---|---|---|---|---|
| Stroop | `trials_stroop` | `trials_stroop_pr` | `stroop_begin.started` | laatste `blank.stopped` van `trials_stroop` | `Rating1` (`key_resp_11`) |
| Arithmetic | `trials_arithmetic` | `trials_arithmetic_pr` | `arithmetic_begin.started` | laatste `arithmetic_2.stopped` van `trials_arithmetic` | `Rating3` (`key_resp_13`) |
| Face Recall | `trials_faceRecall` | geen | `instruct_faceRecall.started` | laatste `face_recall_blank_2.stopped` van `trials_faceRecall` | `Rating4` (`key_resp_14`) |

**Volledig genegeerd voor taakvensters:** `trials_stroop_pr`, `trials_arithmetic_pr`, `trials_faceMemory`, `Rating2`.

### Ter illustratie/validatie: waarden voor p20 (seconden vanaf `expStart`)

| Taak | Start | Eind | Duur |
|---|---|---|---|
| Stroop | ~431,8 (eerste trial `num1_text.started`) | 763,65 | ~332 s |
| Arithmetic | ~1435,9 (eerste trial `arithmetic_2.started`) | 1881,62 | ~446 s |
| Face Recall | ~2019,7 (eerste trial `face_recall.started`) | 2289,79 | ~270 s |

Deze absolute waarden zijn **per participant uniek** (afhankelijk van reactietijden, instructietempo, etc.) en dus niet herbruikbaar als constantes. Elk participant-bestand moet opnieuw doorzocht worden volgens de kolomlogica hierboven, niet op basis van deze concrete getallen.

## Extractielogica (voor scripting)

Voor een gegeven participant-CSV:
1. Filter rijen op niet-lege waarde in `<loopnaam>.thisN` om de trials van één taak te isoleren (`trials_stroop`, `trials_arithmetic`, `trials_faceRecall`).
2. Start van het taakvenster = `.started`-tijdstip van de instructie-/begin-routine die direct voorafgaat (`stroop_begin`, `arithmetic_begin`, `instruct_faceRecall`), of, als een strakkere afbakening gewenst is, het `.started`-tijdstip van de eerste stimulus-component in de eerste trial-rij van het blok.
3. Einde van het taakvenster = laatste relevante `.stopped`-tijdstip in de laatste trial-rij van het blok (kolom verschilt per taak, zie tabel).
4. Target-label = het `key_resp_1x`-antwoord van de bijbehorende Rating-routine, gekoppeld via volgorde (Rating1→Stroop, Rating3→Arithmetic, Rating4→Face Recall). **Niet** Rating2.
5. Koppeling naar PPG/video: tijden hierboven zijn relatief aan `expStart` (zelfde as als het `.log`-bestand); voor absolute klok­tijd `expStart`-timestamp per participant gebruiken om te synchroniseren met de Unix-timestamps in de PPG-data en de OBS start/stop-tijden.

## Synchronization Approach

- No hardware trigger/sync signal available between PPG, video, and PsychoPy.
- Synchronization via wall-clock alignment:
  - PPG: Unix timestamp (ms), absolute.
  - Video/OBS: absolute start/stop time from OBS log.
  - PsychoPy: relative time (as in .log, 0.000 = experiment start) converted to absolute time via `expStart` column (timezone-aware).
- **No buffer/margin applied** at task window edges, full task window used as-is. Post-hoc trimming may be applied later during signal inspection if warranted (not pre-emptively).
- **Limitation to note in thesis:** absence of hardware sync introduces small residual timing uncertainty (likely sub-second), accepted given task durations are much longer than expected clock drift/logging delay.

## Output structure

- `data/processed/synced/{participant}_task_windows.csv`: per-task absolute + relative timestamps
- `data/processed/ppg/`: extracted HRV/PRV features per task
- `data/processed/video/`: extracted facial features per task

## PPG Processing, DECIDED: NeuroKit2

Raw data: Shimmer sensor, 128 Hz. Proposal specified HeartPy, switched to
NeuroKit2 based on accuracy: HeartPy's peak detector misidentifies some
peaks.

Known limitations: LF/HF metrics are in absolute units, not normalized,
and task windows vary in duration, so this should be taken into account.
A small fraction of detected peaks look like artifacts and need to be
fixed.

Task-window trimming: the first 30 seconds of each task window are
excluded before PPG and GSR processing, for the cardiac orienting
response. Chosen as a reasoned own estimate, not taken from a study.

## Video Processing, DECIDED: OpenFace 2.0, with blink-contamination filtering

Tool comparison: OpenFace 2.0 vs LibreFace vs Py-Feat. Chose OpenFace 2.0
(matches the proposal), pre-built Windows binary, no compilation needed.
LibreFace failed on Python 3.12 (dependency incompatibility), Py-Feat was
about 170x slower on CPU.

Feature set: blink rate, gaze, head pose (per proposal), plus all 17
Action Units as an exploratory addition, letting later feature-importance
analysis decide relevance instead of pre-selecting.

Blink-contamination finding: visual inspection showed AU07 peaks coincide
almost entirely with blink onsets rather than independent brow tension,
confirmed quantitatively, well above chance. A systematic check across all
28 candidate features found the same problem in about half of them,
likely because blinks cause landmark jitter that propagates into every
landmark-derived measurement.

Correction: frames within a detected blink, plus a margin either side, are
excluded before computing any feature statistic. Blink rate itself is
still computed on the full signal.

Known limitations: most features still show some mild residual
association with blinks after filtering, only a few are fully clean.
Exclusion rates vary by participant and should be logged, not assumed
constant. Validated on one participant only, needs spot-checking on more.

Confidence filtering: frames with low tracking confidence get a separate
exclusion threshold, since blink filtering alone doesn't catch tracking
failures. Still needs improvement.

Rest-baseline normalization was considered but not used, since the
current research questions compare tasks to each other, not task to rest.
Still an open question.

## Qualtrics/PSS processing

Bron: `data/raw/qualtrics/NERVE_background.csv` (Qualtrics-export, 3 headerrijen: variabelnaam/vraagtekst/ImportId, daarna 79 datarijen). Verwerkt door `scripts/qualtrics/01_process_pss_demographics.py` naar `data/processed/qualtrics/pss_demographics_cleaned.csv` (75 rijen).

### Niet-participant-rijen uitgesloten (4, niet 3)
Van de 79 datarijen zijn er **4** uitgesloten als niet-participant:

| Rij (participantNumber) | Finished | Progress | Duration | StartDate | Reden |
|---|---|---|---|---|---|
| *(leeg)* | 1 | 100% | 10.241 s (ong. 2u51m) | 2026-02-26 04:15 | Zie hieronder, géén p1 |
| *(leeg)* | 0 | 6% | 10 s | 2026-02-26 04:30 | Afgebroken poging |
| `wrong` | 1 | 100% | onbekend | 2026-03-05 | Leeg/testinvoer (Age=1, implausibel) |
| `p` | 0 | 26% | onbekend | 2026-02-26 09:04 | Afgebroken poging |

Uitsluitingsregel in code: `participantNumber` is leeg, OF gelijk aan `'wrong'` (case-insensitive), OF `Finished == 0`.

**Belangrijk:** een eerdere analysestap identificeerde slechts 3 "niet-participant-rijen" (via `Finished==0` OF `id=='wrong'`) en nam daarbij aan dat de resterende lege-ID-rij (Finished=1, 100%) participant **p1** was, die vergeten was zijn/haar ID in te vullen. Verificatie tegen de OBS-log wees dit af: die rij dateert van **2026-02-26**, drie weken vóór p1's daadwerkelijke sessie (**2026-03-18, 13:57:23**), en de duur (ong. 2u51m) is fysiek onwaarschijnlijk voor een achtergrondvragenlijst. Vermoedelijke verklaring: een pilot-/testinvoer van de onderzoeker vóór aanvang van de dataverzameling (dezelfde dag als de eerste echte respons, `p2`). Deze rij is daarom als 4e niet-participant-rij uitgesloten, **niet** toegeschreven aan p1 of enige andere deelnemer.

**Gevolg: p1 ontbreekt volledig in de Qualtrics-data.** Van de verwachte 76 deelnemers (p1–p76) zijn er **75** terug te vinden (p2–p76); voor p1 is geen enkele Qualtrics-respons betrouwbaar te koppelen. `pss_demographics_cleaned.csv` bevat daarom bewust **geen rij voor p1**. Dit moet bij het samenvoegen met de rest van de dataset met een **LEFT/OUTER join** tegen de volledige p1–p76 participantenlijst gebeuren, zodat p1 als `NaN` verschijnt in age/gender/pss_total_score/etc., in plaats van stilzwijgend te worden overgeslagen of ingevuld.

Consistentiecheck: na uitsluiting van deze 4e rij is de gender-verdeling **49 vrouw / 26 man** (in plaats van de eerder gerapporteerde 50/26), precies 1 vrouw minder dan de proposal's bekende steekproefsamenstelling (50 vrouw/26 man). Dit is intern consistent met p1 ontbrekend (als p1 vrouw is, wat niet elders bevestigd is, verklaart dit exact het verschil) en wordt hier gemeld als bevestiging, niet als nieuw probleem.

### Participant-ID-normalisatie
Ruwe `participantNumber`-waarden waren inconsistent: hoofdletter-varianten (`P6`, `P11`, ...), losse cijferreeksen zonder `p`-prefix (`14`, `15`, `34`, `36`, `59`), typo's met verkeerde letter (`b53`→p53, `B58`→p58, `u76`→p76), en een trailing newline (`p50\n`). Normalisatie: strip whitespace, lowercase, dan `"p" + laatste cijferreeks in de string` (regex `(\d+)$`). Dit lost alle bovenstaande gevallen eenduidig op zonder een hardcoded mapping per typo.

### PSS-10 offset-fout
Alle 10 `Perceived_stress_1`–`_10`-kolommen zijn intern gecodeerd als **1–5**, terwijl de vraagtekst het antwoordschaal-label 0–4 (Never–Very Often) toont. Bevestigd doordat elk van de 10 kolommen exact min=1/max=5 heeft, zonder uitzondering (dus een systematische Qualtrics-coderingsfout, geen toevallig ontbrekende "0"-respons). Correctie: **eerst 1 aftrekken van elk item**, dan pas de standaard PSS-10-scoring toepassen: items 4, 5, 7, 8 (1-indexed) zijn reverse-scored (`4 - gecorrigeerde_waarde`), overige items ongewijzigd, som van alle 10 = `pss_total_score` (geldig bereik 0–40; waargenomen bereik in de data: 6–34).

### Gender-hercodering
`Gender` is intern numeriek gecodeerd zonder label in de header. Bevestigd via aantal-matching tegen de proposal (50 vrouw/26 man; zie hierboven voor de nuance na uitsluiting van de 4e niet-participant-rij): **1.0 = female, 2.0 = male**.

### Output
`data/processed/qualtrics/pss_demographics_cleaned.csv`: 75 rijen, kolommen: `participant` (genormaliseerd, p2–p76), `age`, `gender` (female/male), `pss_total_score`, plus overige demografische/gezondheidskolommen (`ed_level`, `handedness`, `chronic_disease`, `taking_medication`, `caffeinated_drinks`, `smoking`, `drinking`, `vision`, `sleep`, `physical_act`). Deze laatste zijn nog **numeriek/ongelabeld** overgenomen uit Qualtrics (geen antwoordlabels beschikbaar in de header) en moeten vóór analyse/rapportage nog gedecodeerd worden.

## Video Processing, DECIDED: session-wide OpenFace processing

OpenFace now runs once per participant on the full session video, instead
of three separately-cut task clips, then the task windows are sliced out
of that single output afterward.

Reason: OpenFace's normalization is sensitive to how much surrounding
context it sees. On identical frames, session-wide versus per-clip
processing gave meaningfully different AU values and a substantially
different blink count. Since blink detection drives the exclusion filter,
that changes which frames get excluded, not just the raw numbers.

Cost: slower overall, since OpenFace also processes the footage between
tasks. Accepted as the price of a consistent normalization context.
Output is cached per participant so this cost is paid once.

Blink-threshold pooling stays restricted to the 3 task windows, not the
full session, to keep this a controlled comparison.

Result: the adaptive blink threshold dropped noticeably under the new
approach despite a similar pooled event count, the underlying duration
distribution changed shape, not expected going in. The core finding this
was testing for was unaffected. Feature validation improved on balance,
though not for every feature.

Not yet done: only tested on one participant, reprocessing the rest of
the already-processed participants was a separate, substantial batch job.

## Video Processing, DECIDED: explicit camera calibration

Every OpenFace run now passes explicit camera intrinsics for the webcam
used to record all participants, instead of letting OpenFace estimate
focal length from image size, the same camera and zoom for everyone, so
one constant applies to all.

Reason: a calibrated-versus-uncalibrated comparison found a systematic,
same-direction correction to absolute head pitch and vertical gaze across
all 3 tasks, a real geometric bias, not noise.

What it does not fix: blink-related contamination. A contaminated
feature's contamination ratio was identical to many decimal places with
or without calibration, camera geometry and blink-related jitter are
unrelated problems.

For within-participant, task-vs-task comparisons (this project's main
use), the bias should largely cancel out since it's similar across tasks.
Calibration mainly matters for absolute-value correctness.

## Video Processing, DECIDED: exclude head pitch and one gaze column

Two features are dropped from the aggregated feature table going forward
(they remain in the raw cached output). Both stayed flagged as
contaminated after two independent, unrelated fixes were tried (adaptive
blink filtering, camera calibration), neither moved them, reasonably
strong evidence the cause is something more structural, most likely
residual landmark jitter right around eye closure. Rather than keep
publishing known-unreliable values, they're excluded outright. The rest
of the pose and gaze columns are unaffected.

## Full-batch reprocessing with the final pipeline configuration

The final configuration (session-wide processing, mandatory calibration,
the two exclusions) is being applied to all participants with video data,
p1 excluded (consistent with p1's PPG also being unusable).

Two bugs found and fixed while spot-checking a few participants:
1. The feature-exclusion list wasn't applied in the validation script, it
   had its own separate, hardcoded column lists. Fixed by moving the
   shared list into one module both scripts import, so they can't diverge
   again.
2. The orchestrator's validation call was pointed at the wrong (old) cache
   directory for several participants, producing validation numbers
   computed against outdated data. Fixed by passing the correct directory
   explicitly.

Generalization finding: after fixing both bugs, re-validating the
original test case against a few other participants on a like-for-like
basis showed a stark gap in how many features come out flagged, too large
to be ordinary individual variance. Likely cause: the blink-filter margin
and the calibration decision were both derived and validated on one
participant alone, a risk already flagged before more participants were
available. The parameters don't carry over well. This needs follow-up
before the feature set is treated as final; not addressed yet, the batch
is proceeding in the meantime.

## Video Processing, explored and rejected for now: OpenFace 3.0

Tested OpenFace 3.0 (Hu et al., 2025) in an isolated venv, never touched the production pipeline. It
promised roughly 2x CPU speedup and a simpler pip install, so it seemed worth checking as a replacement
for OpenFace 2.0.

Not adopted because of three problems:
1. No head pose output at all, and nothing in the project's code or GitHub issues suggests it's planned.
2. Only 8 unlabeled AU channels (vs. 17 named ones from OpenFace 2.0). Which blocks the AU04/glasses
   bias comparison I wanted this for in the first place.
3. The detect-video CLI command is broken.

OpenFace 2.0 stays the better choice. OpenFace 3.0 is worth mentioning as future work in the thesis
(still check it once again).

## GSR/EDA literature update

Boucsein (2012) is still the standard reference for EDA terminology (SCL/SCR, orienting response). Added
three more recent sources:

- Stržinar et al. (2023, Sensors), frequency spectrum EDA features for stress classification (WESAD
  dataset). Relevant as a possible additional feature family beyond SCL/SCR morphology, not implemented
  here.
- Pataca et al. (2025, Computers in Biology and Medicine), systematic review confirming EDA, HRV, and
  PPG as the most used signals for wearable stress detection, and reporting small sample sizes and lack
  of standard protocol as recurring limitations, which matches this thesis's own constraints.
- Tsirmpas et al. (2025, Sensors), quantifies how cvxEDA and Ledalab both tend to fit false phasic peaks
  on real world wearable EDA, especially around abrupt tonic level changes. This grounds the SCR
  over-detection check in the GSR pipeline, and is the EDA equivalent of the HeartPy dicrotic notch
  over-counting issue found earlier in the PPG pipeline.

p10, p12, and p60 are confirmed excluded from GSR analysis.
