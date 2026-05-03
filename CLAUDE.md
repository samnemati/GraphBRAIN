# GraphMind — Project Briefing for Claude Code

**Repository:** `samnemati/GraphMind`  
**PI / Researcher:** Samaneh Nemati (USC, Communication Sciences and Disorders)  
**Last updated:** April 2026

---

## Project Goal

Build a Graph Neural Network (GNN) regression pipeline to predict behavioral outcomes
(cognitive and hearing scores) from resting-state functional brain connectivity in a
healthy aging cohort (ages 20–80). Once the pipeline is working and results are
obtained, write a paper.

This extends prior classification work (PSA patients vs. healthy controls, SNL 2023
poster) into a regression task on a new dataset.

---

## Prior Work (Context)

**SNL 2023 Poster:** "Exploring Structural Connectivity Networks for Classification of
Post-Stroke Aphasia Patients and Healthy Controls using Graph Neural Networks"

- Dataset: PSA patients (N=50) vs. age-matched healthy controls (N=40)
- Graph: DTI structural connectivity → edges; rsfMRI time-series → node features
- Atlas: JHU
- Framework: DGL (Deep Graph Library, Python)
- Architecture: 6 × GCN layers + BatchNorm + ReLU + Linear classifier
- Results: precision=0.85, recall=0.77, F1=0.76
- GitHub (prior, may be incomplete): https://github.com/samnemati/GCNN_PABC

---

## Dataset — ABC Healthy Aging Study

### Brain Data
- **Location:** `/Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/Brain_Data/`
- **Session 1:** 306 .mat files — `ABC_MATfiles_session1/` — PRIMARY dataset
- **Session 2:** 43 .mat files — `ABC_MATfiles_session2/` — longitudinal (5 yrs later)
- **Use Session 1 as the main dataset.** Session 2 is too small to train on alone.
- **Modality summary CSV:** `ABC_MATfiles_session1/ABC_modality_summary.csv`

### Behavioral Data
- **Location:** `/Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/Behavioral_Data/ABC_PTT_WIN_MoCA.xlsx`
- **305 subjects** have both brain + behavioral data (one subject has brain data but no behavioral record)
- **Columns:**

| Column | Description | Range | Mean ± Std | Missing |
|--------|-------------|-------|------------|---------|
| Study ID | Subject ID (e.g. ABC1023) | — | — | 0 |
| Age | Age in years | 20–80 | 47.0 ± 19.2 | 0 |
| Sex (1=1, 0=0) | Biological sex | 0/1 | — | 0 |
| Highest Level of Education Completed | Education category | — | — | 18 |
| MoCA_T | Montreal Cognitive Assessment total score | 14–30 | 27.2 ± 2.6 | 0 |
| WIN_Threshold_R | Words in Noise threshold, right ear | — | — | 32 |
| WIN_Threshold_L | Words in Noise threshold, left ear | — | — | 32 |
| WIN_Threshold_Average | Average WIN threshold | -2.00–24.80 | 7.05 ± 3.71 | 32 |
| PTT_L | Pure Tone Threshold, left ear | — | — | 28 |
| PTT_R | Pure Tone Threshold, right ear | — | — | 28 |
| PTT_Average | Average pure tone threshold (hearing) | -3.75–75.62 | 14.55 ± 13.29 | 28 |

**Primary regression targets:** `MoCA_T` and `PTT_Average`
**Later targets:** `WIN_Threshold_Average`, `Age` (as sanity check)

### .mat File Structure

Each subject file is ~60MB, MATLAB 5.0 format (zlib-compressed blocks). Load with
`scipy.io.loadmat()`. Each file contains ~100 data blocks. Key variables for the
**JHU atlas** (the atlas we are using):

**Connectivity matrices (candidate edges):**
```
rest_jhu        → resting-state FUNCTIONAL connectivity (189×189), fields: 'r', 'label'
                  'r'     : connectivity strength matrix (Pearson correlation)
                  'label' : 189 ROI names from JHU atlas
dti_jhu         → DTI structural connectivity (189×189)
dtifc_jhu       → DTI fiber count connectivity (189×189)
dtimx_jhu       → DTI max connectivity
dtimn_jhu       → DTI min connectivity
```

**Per-node features — already aggregated per JHU region (ready to use as node features):**
```
fa_jhu          → Fractional Anisotropy per region       (shape: 1×189)
md_jhu          → Mean Diffusivity per region            (shape: 1×189)
vbm_gm_jhu      → Gray matter volume per region         (shape: 1×189)
vbm_wm_jhu      → White matter volume per region        (shape: 1×189)
fmri_jhu        → Mean fMRI activation per region       (shape: 1×189)
palf_jhu        → Amplitude low-freq fluctuations        (shape: 1×189)
alf_jhu         → ALF per region                        (shape: 1×189)
i3mT1_jhu       → T1 intensity per region               (shape: 1×189)
```

**Scalar values (subject-level, not per-region):**
```
VBM_volume_Total, VBM_volume_GM, VBM_volume_WM, VBM_volume_CSF, VBM_volume_WMH
```

Other atlases also available in each file: AICHA, AAL, aalcat, bro, cat, fox,
yourcustomatlas — but we are using JHU exclusively.

---

## Design Decisions (agreed in initial planning session)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Atlas | JHU (189 regions) | Consistent with prior work; white matter emphasis |
| Edge type | `rest_jhu` → 'r' matrix | Functional connectivity more sensitive to behavioral outcomes |
| Node features | `fa_jhu`, `md_jhu` | DTI white-matter microstructure measures paired with the white-matter atlas; functional info is captured in edges |
| Framework | PyTorch + PyTorch Geometric (PyG) | More modern than DGL, better docs, same task |
| Primary targets | `MoCA_T`, `PTT_Average` | Cognitive + hearing; start simple |
| Session | Session 1 only (N=305) | Session 2 too small (N=43) |
| Validation | 5-fold cross-validation | Dataset too small for stable held-out test set |
| Loss | MSE (with MAE, R², Pearson r as metrics) | Standard for regression |
| Baseline | Ridge regression on upper triangle of connectivity matrix | Required to address critical 2025 paper questioning GNN advantage |
| Interpretability | GAT attention weights + GNNExplainer | Identify which regions/connections predict MoCA and PTT |

---

## Pipeline Architecture

### Graph Construction (per subject)
- **Nodes:** 189 JHU brain regions
- **Edges:** Thresholded `rest_jhu['r']` matrix — keep top 15–20% strongest connections (proportional threshold, standard in connectomics)
- **Edge weights:** Connectivity strength values
- **Node feature matrix X:** Stack `[fa, md]` → shape [189 × 2]
- **Label y:** Continuous behavioral score (MoCA_T or PTT_Average)

### Model Architecture
```
Input: Node features X [189 × num_features]
       Edge index + edge weights

→ GCN or GAT layers (2–3 layers)
  - ReLU activation
  - Dropout (p=0.3–0.5)
  - BatchNorm

→ Global mean pooling
  [189 × hidden_dim] → [hidden_dim]

→ Fully connected regression head
  → scalar prediction

Loss: MSE
```

Start with GCN. Add GAT variant for interpretability (attention weights = edge importance).

### Project File Structure
```
GraphMind/
├── CLAUDE.md               ← this file
├── README.md
├── requirements.txt
├── config.py               ← paths, hyperparameters, toggles
├── data/
│   ├── __init__.py
│   ├── loader.py           ← load .mat files, extract rest_jhu + node features
│   ├── dataset.py          ← PyG Dataset class (one graph per subject)
│   └── graph_builder.py    ← connectivity matrix → PyG Data object
├── models/
│   ├── __init__.py
│   ├── gcn_regression.py   ← GCN with regression head
│   └── gat_regression.py   ← GAT variant (interpretable)
├── training/
│   ├── __init__.py
│   ├── train.py            ← training loop, cross-validation
│   └── evaluate.py         ← MAE, R², Pearson r, plots
├── interpretation/
│   ├── __init__.py
│   └── explain.py          ← GNNExplainer, attention weight extraction
├── baselines/
│   ├── __init__.py
│   └── ridge_regression.py ← baseline comparison
├── notebooks/
│   └── exploration.ipynb   ← data exploration and visualization
├── results/
│   └── .gitkeep
└── main.py                 ← entry point
```

---

## Key Technical Notes

- **scipy not available in the Cowork sandbox** — use your local Python environment for running code
- **Loading .mat files:** Use `scipy.io.loadmat(filepath)` — the files are MATLAB 5.0 format. Access connectivity as `mat['rest_jhu']['r'][0,0]` and labels as `mat['rest_jhu']['label'][0,0]`
- **Subject ID matching:** .mat filenames (e.g. `ABC1023.mat`) match the `Study ID` column in the behavioral Excel exactly (no quotes, no padding)
- **305 overlapping subjects** between brain data and behavioral data (verified)
- **Missing behavioral data:** PTT_Average missing for 28 subjects, WIN missing for 32 — handle by dropping those subjects for the relevant target, not imputing
- **Connectivity matrix symmetry:** `rest_jhu['r']` is symmetric; use upper triangle for edge construction
- **Thresholding:** Proportional threshold recommended (keep top k% connections per subject) so graph density is consistent across subjects

---

## Literature Review & Key References

### Most Directly Relevant Papers

**1. RegGNN — Regression GNN for cognitive score prediction (MOST RELEVANT)**
- Demirtaş et al. (2022). "Predicting cognitive scores with graph neural networks through
  sample selection learning." *Brain Imaging and Behavior*.
- The first GNN specialized for regressing brain connectomes to a cognitive score.
  Uses functional connectivity (rsfMRI, 116 ROIs, AAL atlas), 2 GCN layers + FC head,
  predicts IQ scores. Implemented in **PyTorch Geometric**. Also proposes a
  sample-selection method to handle small datasets.
- Paper: https://link.springer.com/article/10.1007/s11682-021-00585-7
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9107424/
- GitHub: https://github.com/basiralab/RegGNN

**2. BrainGNN — Interpretable GNN for fMRI (MOST RELEVANT FOR INTERPRETABILITY)**
- Li et al. (2021). "BrainGNN: Interpretable Brain Graph Neural Network for fMRI
  Analysis." *Medical Image Analysis*.
- ROI-aware graph convolutional layers + ROI-selection pooling layers that explicitly
  identify which brain regions are most important for prediction. Applied to fMRI for
  ASD (autism) and HCP tasks. The key reference for the interpretability component.
- Paper: https://www.sciencedirect.com/science/article/abs/pii/S1361841521002784
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9916535/
- PDF: https://www.biorxiv.org/content/10.1101/2020.05.16.100057v4.full.pdf

**3. Integrated fMRI + DTI + sMRI with interpretable GNNs**
- (2024/2025). "Integrated Brain Connectivity Analysis with fMRI, DTI, and sMRI
  Powered by Interpretable Graph Neural Networks." *Medical Image Analysis*.
- Multimodal GNN combining all three modalities available in our .mat files.
  MaskGNN (FC + SC + anatomical) achieves best regression performance. Relevant
  for the multimodal extension phase of the project.
- Paper: https://www.sciencedirect.com/science/article/pii/S1361841525001173
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11383444/

**4. GAT-LI — Graph Attention Network with interpretability**
- (2021). "GAT-LI: a graph attention network based learning and interpreting method
  for functional brain network classification."
- Shows that GNNExplainer outperforms saliency maps for interpreting GAT models.
  Attention weights on edges identify important connections for a prediction.
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8296748/

**5. GNNExplainer (post-hoc interpretability)**
- Ying et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks."
  NeurIPS 2019.
- The standard post-hoc explanation method for GNNs. Identifies the minimal subgraph
  and node features most important for a given prediction.
- PDF: https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf
- Available as `torch_geometric.explain.GNNExplainer`

### Critical / Cautionary Paper — READ BEFORE WRITING THE PAPER

**6. Rethinking functional brain connectome analysis (IMPORTANT)**
- (2025). "Rethinking functional brain connectome analysis: do graph deep learning
  models Help?" *npj Artificial Intelligence*.
- KEY FINDING: The message aggregation mechanism of GNNs does not consistently
  improve behavioral prediction and sometimes degrades it vs. simpler linear models.
  Recommends hybrid approaches (linear + GAT).
- IMPLICATION: We MUST include a ridge regression baseline. If GNNs outperform it,
  great. If not, that is itself a publishable honest contribution.
- Paper: https://www.nature.com/articles/s44387-025-00067-x

### Reviews & Benchmarks

**7. GNN in Brain Connectivity — 2025 Review**
- (2025). "Graph Neural Networks in Brain Connectivity Studies: Methods, Challenges,
  and Future Directions." PMC.
- Comprehensive review of the field. Good source for Introduction section of the paper.
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11763835/

**8. NeuroGraph — Benchmark for GNNs in brain connectomics**
- (2023). "NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics."
- Benchmark datasets including regression tasks (fluid intelligence, working memory).
  Useful for comparing methodology and metrics to community standard.
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10690301/

**9. Survey: GNNs for Brain Graph Learning — IJCAI 2024**
- (2024). "Graph Neural Networks for Brain Graph Learning: A Survey." IJCAI 2024.
- PDF: https://www.ijcai.org/proceedings/2024/0903.pdf
- ArXiv: https://arxiv.org/html/2406.02594v1

**10. BrainGB — Benchmark for brain network analysis with GNNs**
- (2023). "BrainGB: A Benchmark for Brain Network Analysis with Graph Neural Networks."
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10079627/

### Brain Age Prediction (analogous regression task)

**11. GNN for brain age from rs-fMRI**
- (2023). "Brain age prediction using the graph neural network based on resting-state
  functional MRI in Alzheimer's disease." *Frontiers in Neuroscience*.
- Very similar regression task. Predicting a continuous variable from rs-fMRI
  connectivity. Validates our approach for the Age prediction subtask.
- Paper: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1222751/full

**12. Feature attention GNN for brain age + important connections**
- (2024). "Feature attention graph neural network for estimating brain age and
  identifying important neural connections." *Imaging Neuroscience* (MIT Press).
- Combines age estimation with identification of important neural connections —
  directly aligned with our interpretability goal.
- Paper: https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00245/123638/

**13. Structural connectome GNN for age and dementia prediction**
- (2025). "Structural Connectome Analysis using a Graph-based Deep Model for Age
  and Dementia Prediction."
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11952334/

### Related Disease / Disorder Papers (context for Introduction)

**14. GNN review for neurological disorders**
- (2023). "The Combination of a Graph Neural Network Technique and Brain Imaging to
  Diagnose Neurological Disorders: A Review and Outlook."
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10605282/

**15. Explainable GAT for gait impairment from functional networks**
- (2023). "An Explainable Geometric-Weighted Graph Attention Network for Identifying
  Functional Networks Associated with Gait Impairment." PMC.
- Similar regression design (continuous behavioral outcome from rs-fMRI GNN,
  with interpretability). Good methods reference.
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10402187/

### Hearing + Brain Connectivity (background for PTT/WIN prediction)

**16. Age-Related Hearing Decline and Resting-State Networks**
- (2025). "Age-Related Hearing Decline and Resting-State Networks."
  *American Journal of Audiology*.
- Directly relevant background for the hearing prediction task.
- Paper: https://pubs.asha.org/doi/10.1044/2025_AJA-25-00025

### Novelty of This Project

Applying GNNs to predict **hearing scores (PTT, WIN) from brain connectivity** in a
healthy aging cohort appears to be **genuinely novel** — no prior papers found on this
specific combination. Most GNN behavioral prediction papers focus on IQ, fluid
intelligence, or disease diagnosis (Alzheimer's, autism, schizophrenia).

---

## Paper Plan (once results are available)

**Proposed title:** "Predicting Cognitive and Auditory Behavioral Outcomes from
Functional Brain Connectivity using Graph Neural Networks in Healthy Aging"

**Target journals:** NeuroImage, Human Brain Mapping, Brain Connectivity,
Frontiers in Neuroscience (Neuroimaging)

**Proposed structure:**
1. Introduction — brain-behavior relationships, GNNs in neuroimaging, hearing-cognition link
2. Methods — dataset, graph construction, GCN/GAT architecture, cross-validation, baselines, interpretability
3. Results — regression performance (MAE, R², r) for MoCA and PTT; comparison to ridge baseline; important regions/connections
4. Discussion — neurobiological interpretation, comparison to literature, limitations
5. Conclusion

---

## Working Instructions for Claude Code

- Default output for scripts: `.py`; for reports: `.docx`
- Never delete files without approval
- Always show a brief plan before taking action
- Ask clarifying questions before executing non-trivial tasks
- The behavioral data Excel has a lock file (`~$ABC_PTT_WIN_MoCA.xlsx`) — ignore it
- When referencing data paths, use absolute paths based on the user's local machine
- scipy IS available in the local Python environment (just not in the Cowork sandbox)
- Keep commits small and descriptive — this will be used for the paper methods section
