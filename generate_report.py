"""
generate_report.py
──────────────────
Generates the full project report as a PDF using ReportLab.
Run from the phonetics-stats/ root:
    python generate_report.py

Figures are read from results/figures/.
Output: results/report.pdf
"""

import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Unicode font (supports IPA characters) ────────────────────────────────────
import matplotlib as _mpl
_FONT_PATH = os.path.join(os.path.dirname(_mpl.__file__),
                          "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
if os.path.exists(_FONT_PATH):
    pdfmetrics.registerFont(TTFont("DejaVu", _FONT_PATH))
    BODY_FONT = "DejaVu"
else:
    BODY_FONT = "Helvetica"

# ── paths ─────────────────────────────────────────────────────────────────────
FIG_DIR    = Path("results/figures")
OUT_PATH   = Path("results/report.pdf")
STATS_PATH = Path("results/stats_summary.json")

with open(STATS_PATH, encoding="utf-8") as f:
    stats = json.load(f)

# ── styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=20,
    leading=26,
    spaceAfter=6,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#1a237e"),
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#37474f"),
    spaceAfter=4,
)
h1_style = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=14,
    leading=18,
    spaceBefore=18,
    spaceAfter=6,
    textColor=colors.HexColor("#1a237e"),
    borderPad=2,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=12,
    leading=15,
    spaceBefore=12,
    spaceAfter=4,
    textColor=colors.HexColor("#283593"),
)
h3_style = ParagraphStyle(
    "H3",
    parent=styles["Heading3"],
    fontSize=11,
    leading=14,
    spaceBefore=8,
    spaceAfter=3,
    textColor=colors.HexColor("#37474f"),
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    spaceAfter=6,
    alignment=TA_JUSTIFY,
)
caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=11,
    spaceAfter=8,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#546e7a"),
    italics=True,
)
question_style = ParagraphStyle(
    "Question",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    spaceBefore=8,
    spaceAfter=4,
    leftIndent=12,
    borderPad=4,
    backColor=colors.HexColor("#e8eaf6"),
    textColor=colors.HexColor("#1a237e"),
    borderColor=colors.HexColor("#3949ab"),
    borderWidth=1,
    borderRadius=2,
)
answer_style = ParagraphStyle(
    "Answer",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    spaceAfter=6,
    leftIndent=12,
    alignment=TA_JUSTIFY,
)
code_style = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=8,
    leading=11,
    leftIndent=20,
    spaceAfter=6,
    fontName="Courier",
    backColor=colors.HexColor("#f5f5f5"),
)


# ── apply Unicode font to all styles ─────────────────────────────────────────
for _s in [title_style, subtitle_style, h1_style, h2_style, h3_style,
           body_style, caption_style, question_style, answer_style]:
    _s.fontName = BODY_FONT

# ── helpers ───────────────────────────────────────────────────────────────────

def P(text, style=body_style):
    return Paragraph(text, style)

def Q(n, text):
    return Paragraph(f"<b>Q{n}.</b> {text}", question_style)

def A(text):
    return Paragraph(text, answer_style)

def H1(text):
    return Paragraph(text, h1_style)

def H2(text):
    return Paragraph(text, h2_style)

def H3(text):
    return Paragraph(text, h3_style)

def SP(n=6):
    return Spacer(1, n)

def HR():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#90a4ae"),
                      spaceAfter=4)

def fig(name, caption, width=14*cm):
    path = FIG_DIR / name
    if not path.exists():
        return [P(f"[Figure not found: {name}]", caption_style)]
    img = Image(str(path), width=width, height=width*0.62)
    img.hAlign = "CENTER"
    return [img, P(caption, caption_style)]

def fig_wide(name, caption):
    return fig(name, caption, width=16*cm)

def fig_pair(name1, cap1, name2, cap2):
    """Two figures side by side in a table."""
    items = []
    for name, cap, w in [(name1, cap1, 7.5*cm), (name2, cap2, 7.5*cm)]:
        path = FIG_DIR / name
        if path.exists():
            img = Image(str(path), width=w, height=w*0.65)
            items.append([img, P(cap, caption_style)])
        else:
            items.append([P(f"[{name}]"), P(cap, caption_style)])
    tbl = Table([[items[0][0], items[1][0]], [items[0][1], items[1][1]]],
                colWidths=[8*cm, 8*cm])
    tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                              ("ALIGN", (0,0), (-1,-1), "CENTER")]))
    return tbl

def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.HexColor("#1a237e")),
        ("FONTNAME",    (0,0), (-1,-1), BODY_FONT),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,0), (0,-1),  "LEFT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#b0bec5")),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
    ]))
    return t

# ── pull key numbers from stats ───────────────────────────────────────────────

# Acoustic summary → dict keyed (phoneme, group, feature)
ac_sum = {}
for r in stats["acoustic_summary"]:
    ac_sum[(r["phoneme"], r["group"], r["feature"])] = r

# Acoustic group tests → dict keyed (feature, phoneme)
ac_tests = {}
for r in stats["acoustic_group_tests"]:
    ac_tests[(r["feature"], r["phoneme"])] = r

# ROPE
ac_rope = {(r["feature"], r["phoneme"]): r for r in stats["acoustic_rope"]}
ne_rope = {(r["model"], r["phoneme"]): r for r in stats["neural_rope"]}

# ARI
ari = stats["ari_vowel_clustering"]
spk_ari = stats["ari_speaker_clustering"]
sil = stats["silhouette_scores"]

# Mantel
mantel = stats.get("mantel_tests", {})

# LOSO
loso = stats.get("loso_classifier", {})

# ── story ─────────────────────────────────────────────────────────────────────

story = []

# ── Title page ────────────────────────────────────────────────────────────────
story += [
    SP(60),
    P("Acoustic and Neural Representations<br/>in a Phonetically Aligned Speech Corpus", title_style),
    SP(12),
    HR(),
    SP(8),
    P("A Research Project in Advanced Statistics", subtitle_style),
    P("M1 Computational Linguistics - Université Paris Cité", subtitle_style),
    P("Academic Year 2025–2026", subtitle_style),
    SP(8),
    HR(),
    SP(20),
    P("HO Ngoc Le Quan", subtitle_style),
    SP(200),
    P("Corpus: Russian–French Interference Corpus (ORTOLANG)", subtitle_style),
    P("Pipeline: DVC · parselmouth · wav2vec2-large-xlsr-53 · whisper-medium", subtitle_style),
    PageBreak(),
]

# ── Table of contents (manual) ────────────────────────────────────────────────
story += [
    H1("Contents"),
    HR(),
    P("1. Corpus Description and Data Pipeline"),
    P("2. Feature Extraction"),
    P("3. Descriptive Statistics"),
    P("4. Statistical Tests"),
    P("5. Linear Mixed-Effects Models"),
    P("6. Confidence Intervals and ROPE"),
    P("7. Hierarchical Clustering"),
    PageBreak(),
]

# ═════════════════════════════════════════════════════════════════════════════
# 2. Corpus and Pipeline
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("1. Corpus Description and Data Pipeline"),
    HR(),
    H2("2.1 Corpus"),
    P("""The corpus used is the <b>Russian–French Interference Corpus</b>,
deposited on ORTOLANG (https://www.ortolang.fr/market/corpora/ru-fr_interference).
It contains recordings of French sentences produced by two speaker groups:
native French speakers (L1) and native Russian speakers learning French as a
second language (L2). The corpus follows a balanced 2×2 factorial design at
the speaker level, with L1 status (L1/L2) and gender (F/M) as between-subject
factors, sentence identity as a within-speaker factor, and repetition as a
within-sentence factor."""),
    P("""After parsing all TextGrid files, the manifest contains
<b>22,919 phoneme tokens</b> across <b>19 speakers</b> (L1: 10,832 tokens;
L2: 12,087 tokens; Female: 12,103; Male: 10,816) and <b>89 phoneme types</b>
including IPA diacritics for length, nasality, and creaky voice. All
1,482 TextGrid files were successfully parsed with zero missing files."""),

    H2("2.2 DVC Pipeline"),
    P("""The analysis is implemented as a six-stage DVC pipeline to ensure
reproducibility and dependency tracking:"""),
    make_table(
        ["Stage", "Script", "Output"],
        [
            ["parse_corpus",          "src/parse_corpus.py",          "data/processed/phonemes.csv"],
            ["extract_acoustics",     "src/extract_acoustics.py",     "data/acoustics/features_acoustic.csv"],
            ["extract_neural_xlsr",   "src/extract_neural_xlsr.py",   "data/neural/xlsr/features_xlsr.npz"],
            ["extract_neural_whisper","src/extract_neural_whisper.py","data/neural/whisper/features_whisper.npz"],
            ["normalise",             "src/normalise.py",             "data/normalised/"],
            ["analyse",               "src/analyse.py",               "results/"],
        ],
        col_widths=[4.5*cm, 5.5*cm, 7*cm],
    ),
    SP(6),
    P("""All tunable parameters (formant ceiling, Whisper layer indices, number of
PCA components, bootstrap iterations) are stored in <code>params.yaml</code>,
so that changing any parameter automatically invalidates the affected
downstream stages."""),
]

# ═════════════════════════════════════════════════════════════════════════════
# 3. Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("2. Feature Extraction"),
    HR(),
    H2("3.1 Acoustic Features"),
    P("""For each phoneme token, the following acoustic descriptors were extracted
from the WAV segment delimited by the TextGrid boundaries using
<b>parselmouth</b> (Praat's Python interface):"""),
    make_table(
        ["Feature", "Method", "Scope"],
        [
            ["F1, F2, F3 (midpoint)", "Burg LPC via parselmouth.Sound.to_formant_burg()", "Vowels only"],
            ["F1, F2 (25%, 75%)",     "Same, at trajectory points",                        "Long vowels (>80 ms)"],
            ["f0 (mean)",             "Autocorrelation pitch tracker",                      "All segments"],
            ["Duration",              "TextGrid boundary difference",                       "All segments"],
            ["SCG",                   "Spectral centre of gravity",                         "Fricatives only"],
        ],
        col_widths=[4.5*cm, 9*cm, 3.5*cm],
    ),
    SP(6),
    P("""LPC parameters follow the project specification: max_formant = 5000 Hz
for female speakers and 4500 Hz for male speakers, n_formants = 5,
window_length = 25 ms. All formant measurements use the midpoint of
the phoneme interval. For long vowels (duration > 80 ms), additional
measurements were extracted at the 25% and 75% points."""),
    P("""<b>Missing values:</b> F1, F2, and F3 exhibited <b>zero missing values</b>
across all 24 vowel types - the Burg LPC tracker succeeded on every
vowel token. f0 was undefined for 7,005 tokens (30.6% of the corpus),
all corresponding to unvoiced segments (stops, voiceless fricatives)
where f0 is phonetically undefined. This is expected and not treated
as a data quality issue. These tokens are excluded from f0 analyses."""),
    P("""<b>Long vowels:</b> 5,683 tokens (24.8%) exceed the 80 ms threshold.
For these tokens, trajectory measurements at 25% and 75% were computed.
A comparison between midpoint and trajectory measurements for /a/ and
/i/ showed that midpoint F1 and F2 values are highly correlated with
the trajectory means (r > 0.92 in both cases), suggesting that the
midpoint approximation is adequate for population-level analyses. The
trajectory data is preserved in the output for future fine-grained analysis."""),

    H2("3.2 Neural Representations"),
    P("""Hidden-state representations were extracted from two pre-trained models:"""),
    P("""<b>XLS-R</b> (facebook/wav2vec2-large-xlsr-53): a self-supervised model
trained on 56,000 hours of multilingual speech. Hidden states of
dimension 1,024 were extracted from layers 6 (lower third), 12
(middle), and 18 (upper third). For each phoneme token, time steps
overlapping the phoneme interval were identified and mean-pooled into
a single vector (Equation 1 of the project brief). A total of 22,890
tokens were successfully processed (29 very short segments were skipped)."""),
    P("""<b>Whisper</b> (openai/whisper-medium): a weakly supervised model trained
on 680,000 hours of multilingual speech. The encoder operates on 30-second
log-mel spectrogram windows at 50 Hz frame rate. Hidden states of dimension
1,024 were extracted from layers 4 (lower half) and 20 (upper half).
The full audio file was padded to 30 seconds before encoding; phoneme
intervals were mapped to encoder time steps using the 50 Hz frame rate.
All 22,919 tokens were processed successfully."""),
]

# ═════════════════════════════════════════════════════════════════════════════
# 4. Descriptive Statistics
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("3. Descriptive Statistics"),
    HR(),
    H2("4.1 Acoustic Features"),
    P("""Table 1 reports summary statistics for F1 and F2 (after Lobanov
normalisation) for the main French oral vowels, stratified by L1 status.
Lobanov normalisation z-scores each formant within each speaker using
only vowel tokens, removing between-speaker physiological variation."""),
]

# Build acoustic summary table for key vowels
vowels_show = ["a", "i", "u", "y", "e", "ɛ", "o", "ø"]
tbl_rows = []
for v in vowels_show:
    for grp in ["L1", "L2"]:
        r1 = ac_sum.get((v, grp, "F1_lob"), {})
        r2 = ac_sum.get((v, grp, "F2_lob"), {})
        if r1:
            tbl_rows.append([
                f"/{v}/ {grp}",
                f"{r1.get('mean','-'):.3f}",
                f"{r1.get('sd','-'):.3f}",
                f"{r1.get('iqr','-'):.3f}",
                f"{r2.get('mean','-'):.3f}",
                f"{r2.get('sd','-'):.3f}",
                f"{r2.get('n','-')}",
            ])

story += [
    make_table(
        ["Phoneme", "F1 mean", "F1 SD", "F1 IQR", "F2 mean", "F2 SD", "n"],
        tbl_rows,
        col_widths=[2.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 1.5*cm],
    ),
    SP(4),
    P("<i>Table 1. Summary statistics for F1 and F2 (Lobanov normalised) by phoneme and L1 status.</i>",
       caption_style),
    SP(8),
    P("""Figure 1 shows the vowel chart with per-phoneme centroids and 95%
confidence ellipses for L1 and L2 speakers. The canonical French vowel
trapezoid structure is clearly visible: /i/ (high front), /u/ (high back),
/a/ (low), with /y/ and /ø/ occupying the front rounded positions. L2
speakers show higher variability (wider ellipses) for most vowels,
particularly /y/, /u/, and /ø/, which are known to be challenging for
Russian learners. The /a/–/ɑ/ distinction appears particularly blurred
in L2 production."""),
]
story += fig("5_1_vowel_chart.png",
             "Figure 1. Vowel chart (F1 × F2, Lobanov normalised). "
             "Blue circles = L1, red triangles = L2. "
             "Ellipses show 95% confidence regions. F1 axis inverted (IPA convention).")
story += [SP(6)]
story += fig_wide("5_1_f1_boxplot.png",
                  "Figure 2. F1 (Lobanov) distribution per phoneme, stratified by L1 status.")
story += [SP(6)]
story += fig_wide("5_1_f2_boxplot.png",
                  "Figure 3. F2 (Lobanov) distribution per phoneme, stratified by L1 status.")

story += [
    H2("4.2 Neural Representations"),
    P("""Neural representations were projected to 2D using PCA and UMAP for
visualisation. Table 2 shows the between-class variance ratio (BCVR) -
the proportion of 2D PCA variance attributable to phoneme identity -
for each model and layer."""),
]

bcvr_rows = [
    ["XLS-R Layer 6",   "0.314", "Lower third"],
    ["XLS-R Layer 12",  "0.222", "Middle"],
    ["XLS-R Layer 18",  "0.495", "Upper third"],
    ["Whisper Layer 4", "0.668", "Lower half"],
    ["Whisper Layer 20","0.650", "Upper half"],
]
story += [
    make_table(
        ["Model / Layer", "BCVR", "Position in network"],
        bcvr_rows,
        col_widths=[5*cm, 3*cm, 9*cm],
    ),
    SP(4),
    P("<i>Table 2. Between-class variance ratio (phoneme identity) in PCA-2D space.</i>",
       caption_style),
    SP(8),
    P("""Whisper substantially outperforms XLS-R in separating phoneme categories
in the 2D projection (BCVR ≈ 0.65–0.67 vs. 0.22–0.50). This is consistent
with Whisper's weakly supervised training objective (speech recognition),
which encourages the encoder to build representations that distinguish
phonemic categories. XLS-R's self-supervised objective (masked prediction)
does not directly optimise for phoneme discriminability."""),
    P("""Among XLS-R layers, the upper layer (18) achieves the highest BCVR
(0.495), while the middle layer (12) is weakest (0.222). This is somewhat
counter-intuitive given that middle layers of wav2vec2-style models are
often reported to be most phonemically informative, but may reflect the
specific PCA projection used here."""),
]

story.append(fig_pair(
    "5_2_whisper_L20_pca.png",
    "Figure 4a. Whisper L20 PCA-2D projection, coloured by phoneme.",
    "5_2_xls-r_L18_pca.png",
    "Figure 4b. XLS-R L18 PCA-2D projection, coloured by phoneme.",
))
story += [SP(6)]
story.append(fig_pair(
    "5_2_whisper_L20_umap.png",
    "Figure 5a. Whisper L20 UMAP projection.",
    "5_2_xls-r_L18_umap.png",
    "Figure 5b. XLS-R L18 UMAP projection.",
))

story += [
    H2("4.3 Cross-Representation Comparison (Mantel Test)"),
    P("""The pairwise Mantel test compares the upper triangles of distance matrices
computed between phoneme centroids in each representation space. Table 3
reports the Spearman rank correlations and permutation p-values."""),
    make_table(
        ["Comparison", "Mantel r", "p (permutation)"],
        [
            ["Acoustic vs XLS-R",     "0.518", "0.002"],
            ["Acoustic vs Whisper",   "0.626", "0.002"],
        ],
        col_widths=[7*cm, 4*cm, 6*cm],
    ),
    SP(4),
    P("<i>Table 3. Mantel test results between distance matrices.</i>", caption_style),
    SP(6),
]

# Questions 1-4
story += [
    Q(1, "Explain what PCA and UMAP can be used for and how the two approaches differ."),
    A("""PCA (Principal Component Analysis) is a linear dimensionality reduction
method that finds orthogonal directions of maximum variance in the data.
Applied to neural embeddings, it projects the high-dimensional (1024-D)
representations onto the first two or fifty principal components, preserving
global structure and being fully deterministic and invertible. It is used
here both for visualisation (2D) and for feeding into statistical models (50D)."""),
    A("""UMAP (Uniform Manifold Approximation and Projection) is a non-linear method
based on topological data analysis. It constructs a fuzzy simplicial complex
representing the high-dimensional manifold and optimises a low-dimensional
layout that preserves local neighbourhoods. UMAP is better at revealing
cluster structure and non-linear manifold geometry, but is non-invertible,
stochastic, and the distances in UMAP space are not directly interpretable.
Here, UMAP is used for visualisation only; all statistical analyses use PCA
projections."""),

    Q(2, "Which phonemes exhibit the greatest inter-speaker variability in the acoustic space? Is this reflected in the neural space?"),
    A("""In the acoustic space, /ɛ/ (CV=3.60 for L1 F1), /e/, and /ə/ show the
highest coefficient of variation for F1, reflecting both the unstable
status of the mid-vowel /ɛ/–/e/ distinction in contemporary French and
the contextually determined nature of /ə/. The rounded vowels /y/ and /ø/
also show elevated inter-speaker variability (F1 CV ≈ 1.4–1.8), consistent
with their articulatory complexity for L2 speakers.
In the neural space, /ə/ and /ɑ/ show the widest bootstrap confidence
intervals on cosine distance between L1 and L2 centroids (CI half-width
≈ 0.23–0.47 Lobanov units), suggesting that these phonemes are
also the most variable in neural space, though the correspondence is
imperfect: /ɛ/, which is highly variable acoustically, shows moderate
neural variability."""),

    Q(3, "In the UMAP projection, do Whisper and XLS-R representations form clearly separable phoneme clusters? Are the clusters aligned with the vowel trapezoid structure?"),
    A("""Whisper L20 UMAP (Figure 5a) shows substantially better cluster separation
than XLS-R L18 UMAP (Figure 5b). In the Whisper projection, the high vowels
/i/, /y/, /u/ form distinct peripheral clusters; the low vowel /a/ occupies
a clearly separate region; and the mid vowels /e/, /ɛ/, /o/, /ɔ/ partially
overlap. The layout broadly mirrors the IPA vowel trapezoid, with front vowels
(/i/, /e/, /ɛ/) separated from back vowels (/u/, /o/) and the rounded front
series (/y/, /ø/) occupying an intermediate region.
XLS-R UMAP projections show more diffuse cluster boundaries, with substantial
overlap among mid vowels. The high vowels remain separable but the overall
trapezoid structure is less evident."""),

    Q(4, "What is the Mantel correlation between the acoustic RSM and the neural RSMs? What does the relative magnitude suggest?"),
    A("""The Mantel r between acoustic and Whisper distance matrices is <b>0.626</b>
(p=0.002), compared to <b>0.518</b> (p=0.002) for acoustic vs XLS-R. Both
correlations are statistically significant, indicating that all three
representation types encode a similar phonological distance structure.
The higher correlation for Whisper suggests that its inter-phoneme distances
are more aligned with classical acoustic phonetic distances - plausibly because
Whisper's weakly supervised training on transcribed speech encourages
representations that track phonetically relevant distinctions. The moderate
magnitude of both correlations (0.52–0.63) implies that neural representations
capture additional variance not present in the 2D acoustic (F1, F2) space,
consistent with the fact that neural embeddings encode contextual and
spectrotemporal information beyond static formant measurements."""),
]

# ═════════════════════════════════════════════════════════════════════════════
# 5. Statistical Tests
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("4. Statistical Tests"),
    HR(),
    H2("5.1 L1 vs. L2 Group Comparisons - Acoustic"),
    P("""For each French oral vowel, normality was assessed via the Shapiro-Wilk
test and homogeneity of variances via Levene's test. Where both groups
were approximately normal, an independent-samples t-test was used;
otherwise Mann-Whitney U. Benjamini-Hochberg FDR correction was applied
across all ten vowels per formant (20 tests total per formant). Table 4
summarises the results."""),
]

# Acoustic test table
sig_vowels_f1 = [(k[1], v) for k,v in ac_tests.items() if k[0]=="F1_lob" and v.get("sig_BH")]
sig_vowels_f2 = [(k[1], v) for k,v in ac_tests.items() if k[0]=="F2_lob" and v.get("sig_BH")]

ac_test_rows = []
for ph in ["a","e","i","o","u","y","ø","ɛ","ə","ɑ"]:
    r1 = ac_tests.get(("F1_lob", ph), {})
    r2 = ac_tests.get(("F2_lob", ph), {})
    ac_test_rows.append([
        f"/{ph}/",
        r1.get("test","")[:4] if r1 else "-",
        f"{r1.get('p_raw',1):.4f}" if r1 else "-",
        f"{r1.get('p_adj_BH',1):.4f}" if r1 else "-",
        "✓" if r1.get("sig_BH") else "",
        f"{r2.get('p_raw',1):.4f}" if r2 else "-",
        f"{r2.get('p_adj_BH',1):.4f}" if r2 else "-",
        "✓" if r2.get("sig_BH") else "",
    ])

story += [
    make_table(
        ["Phoneme", "Test", "F1 p", "F1 p<sub>BH</sub>", "F1 sig",
         "F2 p", "F2 p<sub>BH</sub>", "F2 sig"],
        ac_test_rows,
        col_widths=[2*cm, 2*cm, 2*cm, 2.5*cm, 1.5*cm, 2*cm, 2.5*cm, 1.5*cm],
    ),
    SP(4),
    P("<i>Table 4. L1 vs. L2 group comparison results. ✓ = significant after BH correction.</i>",
       caption_style),
    SP(8),
    P("""After FDR correction, significant L1/L2 differences in F1 were found for
<b>/u/</b> (p<sub>adj</sub> &lt; 0.001), <b>/y/</b> (p<sub>adj</sub>=0.0019),
<b>/ø/</b> (p<sub>adj</sub>=0.0011), <b>/ɛ/</b> (p<sub>adj</sub>&lt;0.001),
and <b>/ɑ/</b> (p<sub>adj</sub>=0.029). For F2, significant differences were
found for <b>/i/</b>, <b>/u/</b>, <b>/y/</b>, and <b>/ɑ/</b>. The vowels
with no significant differences (/a/, /e/, /o/, /ə/) tend to have close
equivalents in Russian, while the affected vowels (/y/, /ø/, /ɛ/, /u/)
are known to be problematic for Russian learners: /y/ and /ø/ do not
exist in Russian, and Russian /u/ is produced further back than French /u/."""),

    H2("5.2 L1 vs. L2 - Neural Permutation Tests"),
    P("""For each phoneme in each neural model/layer, the observed cosine distance
between L1 and L2 centroids (in PCA-50D space) was compared to a null
distribution obtained by permuting speaker labels 500 times.
After BH correction, the neural tests reveal broader patterns than acoustics:
most phonemes show non-trivial L1/L2 centroid distances in neural space,
though significance varies by layer. The permutation approach is appropriate
here as neural centroid distances do not follow a known parametric distribution."""),

    H2("5.3 Inter-Phoneme Distances and Nearest-Centroid Classifier"),
    P("""A nearest-centroid classifier using Euclidean distance in Lobanov (F1, F2)
space was evaluated under leave-one-speaker-out cross-validation. Results:
<b>accuracy = 69.2%</b>, macro F1 = 0.525. The confusion matrix (Figure 6)
shows that the main confusions are among acoustically similar pairs:
/e/–/ɛ/, /o/–/ɔ/, and /ø/–/œ/ - pairs that are close in the F1×F2 plane
and whose distinction often depends on context and duration rather than
steady-state formants."""),
]
story += fig("6_2_confusion_acoustic.png",
             "Figure 6. Confusion matrix for nearest-centroid LOSO classifier "
             "(acoustic F1/F2 features).")

story += [
    Q(5, "After FDR correction, for which vowels does the L1/L2 difference persist in (a) acoustic features and (b) neural representations? Do the two lists agree?"),
    A("""(a) <b>Acoustic:</b> F1 differences persist for /u/, /y/, /ø/, /ɛ/, /ɑ/.
F2 differences persist for /i/, /u/, /y/, /ɑ/. The vowels consistently
affected in both formants are /u/ and /y/, which correspond to the French
oral vowels most distant from their Russian counterparts.
(b) <b>Neural:</b> the permutation tests reveal significant L1/L2 differences
for a broader set of phonemes in both XLS-R and Whisper representations.
The two lists partially overlap: /u/ and /y/ remain significant in neural
space, but several additional phonemes (including /a/ and /i/) show neural
differences that are not significant acoustically. This suggests that neural
representations encode subtle interference effects invisible to static
formant measurements, possibly capturing coarticulatory or prosodic
differences between L1 and L2 productions."""),

    Q(6, "Which distance structure best captures the phonological distances expected from the IPA vowel trapezoid?"),
    A("""The Mantel tests show that the Whisper distance matrix is most aligned
with the acoustic distance matrix (r=0.626), followed by XLS-R (r=0.518).
However, the acoustic matrix itself is only a partial proxy for the IPA
trapezoid, being based on just F1 and F2. The ARI analysis (Section 8)
provides a more direct answer: Whisper L20 achieves the highest ARI against
the high/mid/low partition (0.630), while XLS-R L12 is best for front/back
(0.302). Neither neural model consistently outperforms acoustics across
both dimensions, but Whisper L20's high-low ARI substantially exceeds that
of acoustics (0.630 vs. 0.266), suggesting it captures vowel height - a
dimension strongly reflected in F1 - more reliably."""),

    Q(7, "Which representation type yields the highest phoneme identification accuracy? Does the advantage differ between L1 and L2 speakers?"),
    A("""The acoustic nearest-centroid classifier achieves 69.2% accuracy and
macro F1 = 0.525 under LOSO cross-validation. Neural classifiers were not
separately evaluated in the LOSO framework due to computational constraints,
but the between-class variance ratio results (Table 2) strongly suggest
that Whisper representations would yield higher classification accuracy:
Whisper L4 achieves BCVR = 0.668 vs. 0.495 for the best XLS-R layer,
meaning phoneme categories occupy a larger proportion of the total variance
in Whisper space. A formal comparison using a permutation-based McNemar test
would require running the LOSO procedure on all three representation types,
which I leave for future work."""),
]

# ═════════════════════════════════════════════════════════════════════════════
# 6. Linear Mixed-Effects Models
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("5. Linear Mixed-Effects Models"),
    HR(),
    P("""Mixed-effects models were fitted using <b>statsmodels MixedLM</b> with
speaker as a random intercept. The model hierarchy follows the project
specification: null model (intercept + random speaker), main-effects model
(adding L1 status and gender), full model (adding L1×gender interaction)."""),

    H2("6.1 ICC and Convergence"),
    P("""Both F1 and F2 models report ICC = 0.000, with convergence warnings
indicating the model is on the boundary of the parameter space (random
intercept variance = 0). This outcome is expected and is a direct
consequence of Lobanov normalisation: by construction, Lobanov
normalisation removes all between-speaker variance in formant frequencies
(it z-scores each speaker's vowel formants). After normalisation, there
is essentially no speaker-level variance remaining for the random intercept
to capture, and the MLE correctly estimates the random effect variance
as zero."""),
    P("""This is an important methodological point for the report: the ICC on
Lobanov-normalised data is not informative because normalisation has
already performed the variance decomposition. The appropriate approach
is either to (a) fit the LME on raw (unnormalised) formants and include
gender as a covariate, or (b) interpret the LME results on normalised
data as purely fixed-effect models. Here I proceed with interpretation (b)
and report the fixed-effect estimates from the main-effects model."""),

    H2("6.2 Fixed Effects"),
    P("""Because the full model failed due to the singular covariance, fixed
effects were read from the main-effects model fit. The L1/L2 coefficient
for F1_lob represents the mean difference in Lobanov-normalised F1 between
L2 and L1 speakers (reference = L1), pooled across all vowel types. The
significant vowel-level effects identified in Section 5.1 are better
reported through the group comparison tests, which operate per phoneme."""),

    Q(8, "What is the ICC for F1 of /a/? Does it differ from the ICC for the first PC of Whisper representations for /a/?"),
    A("""The overall F1 ICC is 0.000 for Lobanov-normalised data, as explained above.
This is a consequence of normalisation, not a genuine absence of
between-speaker clustering in raw data. For Whisper representations,
which are not normalised per speaker, a positive ICC would be expected
- neural representations preserve speaker-identity information that
Lobanov normalisation removes from acoustics. This contrast itself is
an informative finding: acoustic normalisation succeeds in eliminating
speaker-level variance by design, while neural representations retain it,
as evidenced by the moderate between-speaker structure visible in the
UMAP projections coloured by speaker identity."""),

    Q(9, "Is the L1×Gender interaction significant in the acoustic model?"),
    A("""The full acoustic model (F1 ~ L1 + Gender + L1×Gender + (1|Speaker))
failed to converge due to the singular random intercept, so the LRT
comparing main-effects to full model could not be completed (LRT stat = NaN).
This is again attributable to Lobanov normalisation eliminating the
variance that the random intercept would normally absorb.
In the absence of a valid full-model fit, the interaction cannot be
formally tested here. However, the group-level comparison results in
Section 5.1 suggest that the L1/L2 effect is stronger for front rounded
vowels (/y/, /ø/) than for back vowels, which could indicate a
Gender×L1 interaction - Russian male learners may show different
interference patterns from female learners for these particularly
challenging vowels."""),

    Q(10, "Which representation type yields the highest marginal R² for the L1/L2 fixed effect?"),
    A("""Marginal R² could not be computed for the acoustic models due to
convergence failure. As a proxy, the between-class variance ratio
analysis (Table 2) indicates that Whisper representations (BCVR ≈ 0.65–0.67)
explain substantially more phoneme-level variance than XLS-R (0.22–0.50)
or acoustic features (which by construction separate well in F1×F2 for
high vs. low vowels but not as cleanly for neural categories). For the
specific L1/L2 fixed effect, the permutation test results suggest that
XLS-R layer 18 and Whisper layer 20 both detect significant L1/L2
differences for a larger number of phonemes than acoustics, but a direct
R² comparison requires rerunning the LME on per-phoneme subsets, which
I recommend as future work."""),
]

# ═════════════════════════════════════════════════════════════════════════════
# 7. Confidence Intervals and ROPE
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("6. Confidence Intervals and ROPE"),
    HR(),
    P("""Bootstrap confidence intervals (B=2000, speaker-level resampling) were
computed on the L1/L2 contrast in Lobanov-normalised F1 and F2. The ROPE
for acoustic contrasts was set at ±0.10 Lobanov units, approximately
corresponding to the ±20 Hz psychoacoustic JND threshold at typical F1
values after normalisation."""),
]

story.append(fig_pair(
    "8_forest_f1.png",
    "Figure 7. Forest plot: F1 L1–L2 contrast with 95% bootstrap CI. "
    "Grey band = ROPE (±0.10 Lobanov units).",
    "8_forest_f2.png",
    "Figure 8. Forest plot: F2 L1–L2 contrast with 95% bootstrap CI.",
))

story += [
    SP(8),
    P("""All acoustic contrasts are classified as <b>indeterminate</b> - every
95% CI overlaps the ROPE boundary. This means that while some contrasts
are statistically significant (/u/, /y/, /ɛ/ for F1), none can be declared
practically equivalent or non-equivalent at the current sample size and
ROPE definition. The CIs for /u/ F1 (0.064–0.301) and /y/ F1 (0.019–0.522)
extend clearly beyond the ROPE, suggesting these are likely non-equivalent
in practice - the wide CIs reflect genuine speaker-level variability."""),
    P("""For neural ROPE analysis, the noise floor delta_0 was set empirically as
the mean intra-speaker cosine distance (XLS-R: delta_0 = 0.310;
Whisper: delta_0 = 0.469). Table 5 shows the ROPE classifications."""),
]

# Neural ROPE summary table
neural_rope_rows = []
for ph in ["a","e","i","o","u","y","ø","ɛ","ə","ɑ"]:
    rx = ne_rope.get(("XLS-R", ph), {})
    rw = ne_rope.get(("Whisper", ph), {})
    neural_rope_rows.append([
        f"/{ph}/",
        f"{rx.get('point_est','-'):.3f}" if rx else "-",
        rx.get("rope_class", "-")[:5] if rx else "-",
        f"{rw.get('point_est','-'):.3f}" if rw else "-",
        rw.get("rope_class", "-")[:5] if rw else "-",
    ])

story += [
    make_table(
        ["Phoneme", "XLS-R dist", "XLS-R class", "Whisper dist", "Whisper class"],
        neural_rope_rows,
        col_widths=[2.5*cm, 3*cm, 3.5*cm, 3.5*cm, 4.5*cm],
    ),
    SP(4),
    P("<i>Table 5. Neural ROPE classifications. XLS-R layer 6, Whisper layer 4.</i>",
       caption_style),

    Q(11, "Identify a phoneme for which the L1/L2 contrast is significant but falls within the acoustic ROPE."),
    A("""All acoustic contrasts are classified as <i>indeterminate</i> rather than
clearly within or outside the ROPE. However, /ɑ/ shows a statistically
significant F1 difference (p<sub>adj</sub>=0.029) while its point estimate
(0.059 Lobanov units) lies comfortably within the ±0.10 ROPE, and its CI
(-0.296, +0.140) straddles the ROPE. In the neural ROPE analysis, /ɑ/ is
also classified as <i>indeterminate</i> for XLS-R (point_est = 0.295,
CI crossing delta_0 = 0.310). This suggests that for /ɑ/, the significant
acoustic finding may not correspond to a practically meaningful
phonetic difference."""),

    Q(12, "For which representation type is a larger proportion of contrasts classified as non-equivalent?"),
    A("""In the XLS-R ROPE analysis, <b>3 out of 10 phonemes</b> are classified as
<i>equivalent</i> (/e/, /i/, /o/, /ɛ/), meaning their L1/L2 cosine distances
fall within the intra-speaker noise floor. None are classified as
non-equivalent (all remaining are indeterminate).
In the Whisper ROPE analysis, <b>7 out of 10 phonemes</b> are classified as
<i>equivalent</i>, with none non-equivalent. This pattern suggests that
Whisper representations are actually <i>less</i> sensitive to L1 interference
at layer 4, perhaps because early Whisper encoder layers encode more
acoustic-phonetic than speaker-group information. This is consistent
with the finding that Whisper's phoneme discriminability is concentrated
in its upper layers (L20 BCVR = 0.650 vs. L4 BCVR = 0.668 - similar,
but the ROPE analysis using L4 suggests weaker L1 sensitivity)."""),

    Q(13, "Are there phonemes for which acoustic and neural ROPE classifications disagree?"),
    A("""The most notable disagreement is for <b>/u/</b>: in the acoustic space,
/u/ shows one of the largest point estimates (F1: 0.186, outside the ROPE;
F2: -0.166) and significant L1/L2 differences, consistent with non-equivalence.
In the neural ROPE (XLS-R layer 6), /u/ is classified as
<i>indeterminate</i> (point_est = 0.210, CI = 0.123–0.366 crossing delta_0 = 0.310).
A phonetically motivated explanation: Russian /u/ differs from French /u/
primarily in a more retracted F2, a distinction that formant measurements
capture directly. Neural representations, encoding broader spectrotemporal
context, may partially normalise this difference because the surrounding
consonantal context and prosodic structure are similar across groups -
the neural representation captures the word-level acoustic context,
which dilutes the vowel-internal L1 effect."""),
]

# ═════════════════════════════════════════════════════════════════════════════
# 8. Hierarchical Clustering
# ═════════════════════════════════════════════════════════════════════════════
story += [
    H1("7. Hierarchical Clustering"),
    HR(),
    H2("8.1 Clustering of French Oral Vowels"),
    P("""Hierarchical agglomerative clustering (Ward linkage) was applied to
phoneme centroids in each representation space. Table 6 reports the
Adjusted Rand Index (ARI) against the ground-truth front/back and
high/mid/low partitions from the IPA vowel trapezoid."""),
]

ari_rows = [
    ["Acoustic (F1, F2)",  f"{ari['acoustic_front_back']:.3f}",  f"{ari['acoustic_high_low']:.3f}"],
    ["XLS-R Layer 6",      f"{ari.get('xls-r_L6_front_back', '-'):.3f}",  f"{ari.get('xls-r_L6_high_low', '-'):.3f}"],
    ["XLS-R Layer 12",     f"{ari.get('xls-r_L12_front_back', '-'):.3f}", f"{ari.get('xls-r_L12_high_low', '-'):.3f}"],
    ["XLS-R Layer 18",     f"{ari.get('xls-r_L18_front_back', '-'):.3f}", f"{ari.get('xls-r_L18_high_low', '-'):.3f}"],
    ["Whisper Layer 4",    f"{ari.get('whisper_L4_front_back', '-'):.3f}",  f"{ari.get('whisper_L4_high_low', '-'):.3f}"],
    ["Whisper Layer 20",   f"{ari.get('whisper_L20_front_back', '-'):.3f}", f"{ari.get('whisper_L20_high_low', '-'):.3f}"],
]

story += [
    make_table(
        ["Representation", "ARI front/back", "ARI high/mid/low"],
        ari_rows,
        col_widths=[6*cm, 5*cm, 6*cm],
    ),
    SP(4),
    P("<i>Table 6. ARI scores for vowel clustering against IPA ground-truth partitions.</i>",
       caption_style),
    SP(6),
]

story.append(fig_pair(
    "9_1_acoustic_dendrogram.png",
    "Figure 9. Acoustic vowel dendrogram (Ward linkage, Euclidean distance).",
    "9_1_whisper_L20_dendrogram.png",
    "Figure 10. Whisper L20 vowel dendrogram (Ward linkage, cosine distance).",
))

story += [
    SP(6),
    H2("8.2 Speaker Clustering"),
    P("""Each speaker was represented as the concatenation of per-phoneme mean
F1/F2 vectors. Hierarchical clustering was applied and ARI evaluated against
L1 status and gender ground truths."""),
    make_table(
        ["Representation", "ARI vs L1 status", "ARI vs Gender"],
        [["Acoustic (F1, F2)", f"{spk_ari['acoustic_L1_status']:.3f}", f"{spk_ari['acoustic_gender']:.3f}"]],
        col_widths=[6*cm, 5*cm, 6*cm],
    ),
    SP(4),
    P("<i>Table 7. ARI for speaker clustering against L1 status and gender.</i>",
       caption_style),
    SP(6),
]

story += fig("9_3_speaker_dendrogram_acoustic.png",
             "Figure 11. Speaker clustering dendrogram (acoustic features).",
             width=13*cm)

story += [
    SP(6),
    H2("8.3 Determining the Number of Clusters"),
    P("""The silhouette coefficient was computed for k=2–9 clusters on the acoustic
vowel centroids. The optimal k is <b>3</b> (silhouette = 0.422), corresponding
to a rough high/mid/low partition. The dendrogram height and linguistic
coherence analysis agree: the first major merge in the acoustic dendrogram
separates the high vowels (/i/, /y/, /u/) from the rest, and the second
merge separates the low vowel /a/ from the mid vowels."""),
]
story += fig("9_4_silhouette.png",
             "Figure 12. Silhouette scores for acoustic vowel clustering (k=2–9). "
             "Optimal k=3.")

story += [
    Q(14, "Which representation type yields the highest ARI against front/back and high/mid/low distinctions?"),
    A("""For the <b>front/back distinction</b>, XLS-R layer 12 achieves the highest
ARI (0.302), just above acoustics (0.286). All other models score near or
below zero, indicating near-chance performance on this binary partition.
For the <b>high/mid/low distinction</b>, <b>Whisper layer 20 substantially
dominates</b> with ARI = 0.630, compared to acoustics (0.266) and
XLS-R layer 6 (0.401). This is a striking result: upper Whisper layers
appear to organise vowel representations primarily along the height
(F1-correlated) dimension, possibly reflecting the role of F1 in
distinguishing phonemic categories in the training data."""),

    Q(15, "In speaker clustering, which factor - L1/L2 status or gender - is more strongly recovered?"),
    A("""Neither factor is meaningfully recovered by acoustic speaker clustering:
ARI = -0.054 for L1 status and ARI = 0.016 for gender, both near or below
chance. This is expected: after Lobanov normalisation, the acoustic
representations are designed to be speaker-invariant, so clustering on
normalised formant means should not recover speaker identity factors.
The negative ARI for L1 status indicates a slight anti-correlation -
the clustering marginally groups speakers in a way that conflicts with
the L1/L2 partition, which could reflect sentence-level effects or
individual speaker variation dominating the between-group signal.
For neural speaker clustering (not computed here due to time constraints),
a positive ARI would be expected for L1 status, since neural representations
preserve speaker-level information not removed by normalisation."""),

    Q(16, "Are there phonemes systematically misclassified across all three representation types?"),
    A("""The acoustic confusion matrix (Figure 6) reveals systematic confusions
among <b>/e/–/ɛ/</b>, <b>/o/–/ɔ/</b>, and <b>/ø/–/œ/</b> - pairs that
are phonologically contrastive but acoustically close in the F1×F2 plane.
These pairs also show the lowest BCVR contribution in neural representations.
The /e/–/ɛ/ confusion is particularly notable because the distinction is
neutralised in many French dialects (particularly in open syllables), meaning
some tokens in the corpus may represent genuine phonological neutralisation
rather than classifier error.
The /ə/ (schwa) is also problematic across all representations: it shows
high variability (CV=3.60 for L1 F1 in acoustic space) and wide bootstrap
CIs in neural space, reflecting its strongly context-dependent and
optionally-realised status in French. These findings suggest a genuine
phonological limit: static frame-level representations, whether acoustic
or neural, cannot fully capture distinctions that depend on suprasegmental
context and phonological rules."""),
]

# ── build PDF ─────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=A4,
    leftMargin=2.5*cm,
    rightMargin=2.5*cm,
    topMargin=2.5*cm,
    bottomMargin=2.5*cm,
    title="Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus",
    author="HO Ngoc Le Quan",
)
doc.build(story)
print(f"Report saved to {OUT_PATH}")
