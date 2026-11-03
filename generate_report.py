"""
generate_report.py - Generates a PDF project report for the
Pangenome-Based Variant Pathogenicity Prediction project.

Usage:
    pip install fpdf2
    python generate_report.py
"""

from fpdf import FPDF


class ProjectReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "Pangenome-Based Variant Pathogenicity Prediction - Project Report", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def bullet_point(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, "  - " + text)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        for line in text.split("\n"):
            self.cell(5)
            self.cell(0, 5, line, ln=True, fill=True)
        self.ln(3)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(200, 220, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True, align="C")
        self.ln()

    def table_row(self, cols, widths):
        self.set_font("Helvetica", "", 9)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, align="C")
        self.ln()


def main():
    pdf = ProjectReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===== TITLE PAGE =====
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "Pangenome-Based Variant Pathogenicity", align="C", ln=True)
    pdf.cell(0, 12, "Prediction for Alzheimer's Genes", align="C", ln=True)
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "M.Tech Project Report", align="C", ln=True)
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Technologies: Python, DNABERT, scikit-learn, NetworkX, PyTorch", align="C", ln=True)
    pdf.cell(0, 8, "Genes: APP, PSEN1, PSEN2 (Alzheimer's Disease)", align="C", ln=True)
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Two pipelines: Deep Learning (DNABERT) + Classical ML (RandomForest, SVM, etc.)", align="C", ln=True)

    # ===== 1. PROJECT OVERVIEW =====
    pdf.add_page()
    pdf.chapter_title("1. Project Overview")
    pdf.body_text(
        "This project implements a variant pathogenicity prediction system for Alzheimer's disease-related "
        "genes (APP, PSEN1, PSEN2) using pangenome graph representations and machine learning. "
        "The system classifies DNA sequence variants as Healthy/Benign or Unhealthy/Pathogenic."
    )
    pdf.body_text(
        "The project consists of two independent ML pipelines:\n\n"
        "Pipeline A (main.py): Pangenome Graph -> DNABERT deep learning inference\n"
        "Pipeline B (gene_classifier.py): FASTA files -> k-mer frequency -> classical ML classifiers"
    )

    pdf.section_title("Objectives")
    pdf.bullet_point("Parse pangenome graphs (GFA format) representing genetic variation")
    pdf.bullet_point("Extract linear paths from graph structures for sequence analysis")
    pdf.bullet_point("Apply k-mer tokenization for DNA sequence representation")
    pdf.bullet_point("Classify variants using pretrained DNABERT (transformer-based)")
    pdf.bullet_point("Compare with classical ML baselines (RF, SVM, LR, Gradient Boosting)")
    pdf.bullet_point("Evaluate on real Alzheimer's gene mutations from ClinVar database")

    # ===== 2. DATASET =====
    pdf.add_page()
    pdf.chapter_title("2. Dataset Description")

    pdf.section_title("Dataset Structure")
    pdf.body_text(
        "The dataset contains DNA sequences from three Alzheimer's-related genes, each with "
        "healthy (wild-type) and unhealthy (mutated) variants generated from ClinVar annotations."
    )

    widths = [30, 35, 35, 35, 55]
    pdf.table_header(["Gene", "Healthy", "Unhealthy", "Total", "Seq Length Range"], widths)
    pdf.table_row(["APP", "23", "23", "46", "~290K bp"], widths)
    pdf.table_row(["PSEN1", "78", "78", "156", "~44K bp"], widths)
    pdf.table_row(["PSEN2", "9", "9", "18", "~44K bp"], widths)
    pdf.table_row(["TOTAL", "110", "110", "220", "43K - 290K bp"], widths)
    pdf.ln(5)

    pdf.section_title("Dataset Statistics")
    pdf.bullet_point("Total sequences: 220 (perfectly balanced)")
    pdf.bullet_point("Class distribution: 110 Healthy (label 0) + 110 Unhealthy (label 1)")
    pdf.bullet_point("Sequence lengths: Min=43,831 bp, Max=290,579 bp, Mean=127,268 bp")
    pdf.bullet_point("Mutation types: SNV, Deletion, Insertion, Duplication, Inversion, Delins")

    pdf.section_title("Data Generation Process")
    pdf.body_text(
        "Mutation scripts (mutation_app.py, mutation.py) apply known ClinVar variants to reference "
        "genomes. Each script:\n"
        "1. Reads the reference FASTA file for the gene\n"
        "2. Reads a details file listing mutations (type, position, ref, alt)\n"
        "3. Applies each mutation independently to generate separate FASTA files\n"
        "4. Handles: SNV, deletion, insertion, duplication, delins, and inversion"
    )

    # ===== 3. PIPELINE A =====
    pdf.add_page()
    pdf.chapter_title("3. Pipeline A: Pangenome -> DNABERT")

    pdf.section_title("Architecture Overview")
    pdf.body_text(
        "This pipeline processes pangenome variation graphs (GFA format) through a series of "
        "transformations to produce input suitable for the DNABERT transformer model."
    )

    pdf.section_title("Stage 1-2: Graph Processing (graph_processing.py)")
    pdf.bullet_point("Parses GFA v1/rGFA files (S=segments, L=links, P=paths)")
    pdf.bullet_point("Builds a NetworkX directed graph with DNA sequences on nodes")
    pdf.bullet_point("Identifies Strongly Connected Components (SCCs)")
    pdf.bullet_point("Contracts SCCs into super-nodes to produce a true DAG")
    pdf.bullet_point("Result: DAG with 9 nodes, 12 edges (from 12 segments, 17 links)")

    pdf.section_title("Stage 3: Path Extraction (path_extraction.py)")
    pdf.bullet_point("Strategy 1: Topological DFS from all source nodes to all sink nodes")
    pdf.bullet_point("Strategy 2: Random-walk sampling for additional path diversity")
    pdf.bullet_point("Deduplication of identical paths")
    pdf.bullet_point("Result: 8 unique linear paths through the pangenome")

    pdf.section_title("Stage 4: Sequence Linearization (sequence_processing.py)")
    pdf.bullet_point("Concatenates DNA sequences along each path")
    pdf.bullet_point("Each path becomes a single linear DNA string (96-120 bp)")
    pdf.bullet_point("Represents one possible haplotype through the pangenome")

    pdf.section_title("Stage 5: K-mer Tokenization (tokenizer_module.py)")
    pdf.bullet_point("Sliding window of size k (default k=3) with stride 1")
    pdf.bullet_point("Example: ATCGATCG -> ATC, TCG, CGA, GAT, ATC, TCG, CGA")
    pdf.bullet_point("Vocabulary: 5 special tokens + 4^k k-mers = 69 tokens (for k=3)")
    pdf.bullet_point("Special tokens: [CLS], [SEP], [PAD], [UNK], [MASK]")

    pdf.section_title("Stage 6: Sequence Encoding (sequence_processing.py)")
    pdf.bullet_point("Format: [CLS] + k-mer tokens + [SEP] + [PAD]...")
    pdf.bullet_point("Fixed length: 128 tokens (pad or truncate)")
    pdf.bullet_point("Generates: token_ids, position_ids, attention_mask")
    pdf.bullet_point("Attention mask: 1 for real tokens, 0 for padding")

    pdf.section_title("Stage 7-8: DNABERT Inference (dnabert_model.py)")
    pdf.bullet_point("Model: zhihan1996/DNA_bert_3 (DNABERT with k=3)")
    pdf.bullet_point("Architecture: BERT-base (86M parameters) + classification head")
    pdf.bullet_point("Output: Softmax probabilities for [BENIGN, PATHOGENIC]")
    pdf.bullet_point("Supports fine-tuning with AdamW optimizer + weighted CrossEntropyLoss")
    pdf.bullet_point("Threshold tuning for class-imbalanced predictions")

    # ===== 4. PIPELINE B =====
    pdf.add_page()
    pdf.chapter_title("4. Pipeline B: K-mer Frequency + Classical ML")

    pdf.section_title("Feature Extraction")
    pdf.body_text(
        "Each DNA sequence is converted to a k-mer frequency vector:\n"
        "1. Build vocabulary of all possible 4-mers (4^4 = 256 entries)\n"
        "2. Slide a window of size 4 across the sequence\n"
        "3. Count occurrences of each k-mer\n"
        "4. Normalize to frequencies (proportions)\n"
        "Result: Each sequence becomes a 256-dimensional feature vector"
    )

    pdf.section_title("Models Compared")
    pdf.bullet_point("Random Forest: 200 trees, max_depth=20, class-weight balanced")
    pdf.bullet_point("SVM (RBF kernel): C=10, gamma=scale, class-weight balanced")
    pdf.bullet_point("Logistic Regression: C=1.0, max_iter=2000, class-weight balanced")
    pdf.bullet_point("Gradient Boosting: 150 trees, max_depth=5, learning_rate=0.1")

    pdf.section_title("Evaluation Strategy")
    pdf.bullet_point("Stratified train/test split: 80% train (176), 20% test (44)")
    pdf.bullet_point("5-fold stratified cross-validation on training set")
    pdf.bullet_point("Metrics: Accuracy, Precision, Recall, F1, ROC-AUC")
    pdf.bullet_point("Confusion matrix visualization")
    pdf.bullet_point("Feature importance analysis (top k-mers)")

    # ===== 5. RESULTS =====
    pdf.add_page()
    pdf.chapter_title("5. Results & Performance")

    pdf.section_title("Pipeline B: Gene Classifier Results (Test Set)")
    widths = [35, 25, 25, 22, 22, 25, 36]
    pdf.table_header(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1 (5-fold)"], widths)
    pdf.table_row(["Grad. Boost", "0.6818", "0.6818", "0.6818", "0.6818", "0.6932", "0.6024 +/- 0.09"], widths)
    pdf.table_row(["Random Forest", "0.6136", "0.6316", "0.5455", "0.5854", "0.6921", "0.4751 +/- 0.12"], widths)
    pdf.table_row(["Logistic Reg.", "0.5455", "0.5625", "0.4091", "0.4737", "0.5806", "0.3474 +/- 0.10"], widths)
    pdf.table_row(["SVM (RBF)", "0.4773", "0.4444", "0.1818", "0.2581", "0.3884", "0.2425 +/- 0.13"], widths)
    pdf.ln(5)

    pdf.section_title("Best Model: Gradient Boosting")
    pdf.body_text("Confusion Matrix (44 test samples):")
    widths2 = [50, 40, 40]
    pdf.table_header(["", "Pred Healthy", "Pred Unhealthy"], widths2)
    pdf.table_row(["Actual Healthy", "15 (TN)", "7 (FP)"], widths2)
    pdf.table_row(["Actual Unhealthy", "7 (FN)", "15 (TP)"], widths2)
    pdf.ln(5)

    pdf.section_title("Top Discriminative K-mers (Gradient Boosting)")
    widths3 = [15, 25, 35]
    pdf.table_header(["Rank", "K-mer", "Importance"], widths3)
    kmers = [("1", "CATG", "0.05488"), ("2", "TCTG", "0.05202"), ("3", "TGTT", "0.04011"),
             ("4", "CTGT", "0.03172"), ("5", "CTCT", "0.03136"), ("6", "TTTT", "0.02893"),
             ("7", "TGGG", "0.02385"), ("8", "AGCA", "0.02322"), ("9", "CCAT", "0.02175"),
             ("10", "GATC", "0.02110")]
    for row in kmers:
        pdf.table_row(list(row), widths3)
    pdf.ln(5)

    pdf.section_title("Pipeline A: DNABERT Results (Without Fine-tuning)")
    widths4 = [40, 30]
    pdf.table_header(["Metric", "Value"], widths4)
    pdf.table_row(["Accuracy", "0.5000"], widths4)
    pdf.table_row(["F1 Score", "0.6667"], widths4)
    pdf.table_row(["Precision", "0.5000"], widths4)
    pdf.table_row(["Recall", "1.0000"], widths4)
    pdf.ln(5)
    pdf.body_text(
        "Note: The pretrained DNABERT model without fine-tuning predicts all sequences as PATHOGENIC. "
        "The classification head is randomly initialized and requires fine-tuning on labelled data "
        "to learn meaningful distinctions between benign and pathogenic variants."
    )

    # ===== 6. ANALYSIS =====
    pdf.add_page()
    pdf.chapter_title("6. Analysis & Observations")

    pdf.section_title("Key Findings")
    pdf.bullet_point("Gradient Boosting achieves best baseline: 68.18% accuracy with balanced predictions")
    pdf.bullet_point("Random Forest is second-best (61.36%) but has lower recall (misses unhealthy samples)")
    pdf.bullet_point("SVM and Logistic Regression underperform - k-mer frequency differences are subtle")
    pdf.bullet_point("DNABERT without fine-tuning cannot distinguish classes (random classification head)")
    pdf.bullet_point("Cross-validation shows moderate instability (std 0.09-0.13) suggesting limited data")

    pdf.section_title("Why Performance is Moderate")
    pdf.bullet_point("Small dataset: 220 samples is limited for ML, especially for 256 features")
    pdf.bullet_point("Single-nucleotide variants cause minimal k-mer frequency changes in long sequences")
    pdf.bullet_point("K-mer frequency loses positional information (mutation location is lost)")
    pdf.bullet_point("Gene-length imbalance: APP (~290K bp) vs PSEN1/PSEN2 (~44K bp)")

    pdf.section_title("Potential Improvements")
    pdf.bullet_point("Fine-tune DNABERT on the 220-sample dataset with --train flag")
    pdf.bullet_point("Use position-aware features (mutation-window k-mers instead of full-sequence)")
    pdf.bullet_point("Increase dataset size by including more ClinVar variants")
    pdf.bullet_point("Try k=5 or k=6 for more discriminative features (1024 or 4096 dimensions)")
    pdf.bullet_point("Apply data augmentation (reverse complement, random subsequences)")
    pdf.bullet_point("Use graph-aware embeddings from the pangenome structure")

    # ===== 7. FILE STRUCTURE =====
    pdf.add_page()
    pdf.chapter_title("7. Project File Structure")

    pdf.section_title("Core Pipeline Files")
    files = [
        ("main.py", "Master orchestrator for Pipeline A (GFA -> DNABERT)"),
        ("gene_classifier.py", "Complete Pipeline B (FASTA -> k-mer -> sklearn classifiers)"),
        ("graph_processing.py", "GFA parsing, directed graph construction, SCC -> DAG"),
        ("path_extraction.py", "Topological + random-walk path enumeration from DAG"),
        ("sequence_processing.py", "Path linearization + token encoding with special tokens"),
        ("tokenizer_module.py", "Sliding-window k-mer tokenizer with extensible interface"),
        ("annotation_loader.py", "CSV/TSV/VCF label loading with format auto-detection"),
        ("dnabert_model.py", "DNABERT wrapper: predict, fine-tune, evaluate, save"),
        ("utils.py", "PipelineConfig dataclass, logging, validation helpers"),
        ("download_dataset.py", "ClinVar download + pangenome GFA builder"),
    ]
    widths5 = [55, 130]
    pdf.table_header(["File", "Purpose"], widths5)
    for fname, purpose in files:
        pdf.table_row([fname, purpose], widths5)
    pdf.ln(5)

    pdf.section_title("Data Files")
    data_files = [
        ("sample.gfa", "Toy pangenome graph (12 segments, 17 links) for testing"),
        ("sample_labels.csv", "8 test labels for sample.gfa evaluation"),
        ("clinvar_pangenome.gfa", "80-segment pangenome from 20 real ClinVar variants"),
        ("clinvar_labels.csv", "40 labels for ClinVar pangenome paths"),
        ("Dataset/Dataset/", "220 FASTA files across APP, PSEN1, PSEN2 genes"),
        ("trained_model/", "Saved model weights (model.safetensors + config.json)"),
    ]
    pdf.table_header(["File/Folder", "Purpose"], widths5)
    for fname, purpose in data_files:
        pdf.table_row([fname, purpose], widths5)
    pdf.ln(5)

    pdf.section_title("Data Generation Scripts")
    gen_files = [
        ("Dataset/.../APP/mutation_app.py", "Applies ClinVar mutations to APP reference genome"),
        ("Dataset/.../PSEN1/mutation.py", "Applies mutations to PSEN1 reference genome"),
        ("Dataset/.../PSEN2/mutation.py", "Applies mutations to PSEN2 reference genome"),
    ]
    pdf.table_header(["File", "Purpose"], widths5)
    for fname, purpose in gen_files:
        pdf.table_row([fname, purpose], widths5)

    # ===== 8. HOW TO RUN =====
    pdf.add_page()
    pdf.chapter_title("8. How to Run")

    pdf.section_title("Installation")
    pdf.code_block("pip install -r requirements.txt")

    pdf.section_title("Pipeline B: Gene Classifier")
    pdf.code_block(
        "python gene_classifier.py --model rf --kmer 4\n"
        "python gene_classifier.py --model svm --kmer 4\n"
        "python gene_classifier.py --model lr --kmer 4\n"
        "python gene_classifier.py --model gb --kmer 4"
    )

    pdf.section_title("Pipeline A: DNABERT (Inference Only)")
    pdf.code_block(
        "python main.py --gfa sample.gfa --labels sample_labels.csv"
    )

    pdf.section_title("Pipeline A: DNABERT (Fine-tune + Evaluate)")
    pdf.code_block(
        "python main.py --gfa sample.gfa --labels sample_labels.csv --train --epochs 20"
    )

    pdf.section_title("Pipeline A: With ClinVar Data")
    pdf.code_block(
        "python main.py --gfa clinvar_pangenome.gfa --labels clinvar_labels.csv --train"
    )

    # ===== 9. TECHNOLOGIES =====
    pdf.add_page()
    pdf.chapter_title("9. Technologies & Libraries")

    widths6 = [45, 50, 95]
    pdf.table_header(["Library", "Version", "Usage"], widths6)
    techs = [
        ("Python", "3.10+", "Core programming language"),
        ("PyTorch", ">=2.0", "Deep learning framework (DNABERT backbone)"),
        ("Transformers", ">=4.30", "HuggingFace BERT model loading & fine-tuning"),
        ("scikit-learn", ">=1.3", "Classical ML classifiers & evaluation metrics"),
        ("NetworkX", ">=3.1", "Graph data structures & algorithms (DAG, SCC)"),
        ("NumPy", ">=1.24", "Numerical computation & feature vectors"),
        ("tokenizers", ">=0.13", "BPE tokenizer support (extensibility)"),
    ]
    for row in techs:
        pdf.table_row(list(row), widths6)

    # ===== SAVE =====
    output_path = "Project_Report.pdf"
    pdf.output(output_path)
    print(f"\n  PDF report generated: {output_path}")
    print(f"  Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
