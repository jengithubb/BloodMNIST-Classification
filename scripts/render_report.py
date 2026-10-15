"""Regenerate portfolio figures/PDF from archived aggregates; no model fitting."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
GREEN="#286A59"
INK="#203E35"
MUTED="#66756E"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
    "axes.spines.top":False,"axes.spines.right":False})
def rows(name):
    with (ROOT/"results/reference"/name).open() as f:
        return list(csv.DictReader(f))

styles=getSampleStyleSheet()
for name,size,leading,color,font in [
    ("K",9,13,GREEN,"Helvetica-Bold"),("T",28,32,INK,"Helvetica-Bold"),
    ("H",14,19,INK,"Helvetica-Bold"),("B",10.2,15,INK,"Helvetica"),
    ("S",8.6,12,MUTED,"Helvetica"),("C",9.5,13,INK,"Helvetica")]:
    styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,
        textColor=HexColor(color),spaceAfter=10,spaceBefore=8 if name=="H" else 0))
def p(text,kind="B"): return Paragraph(text,styles[kind])
def table(data,widths):
    t=Table([[p(str(v),"C") for v in r] for r in data],colWidths=widths,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HexColor("#EEF4F0")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),9),
        ("LINEBELOW",(0,0),(-1,-1),.35,HexColor("#D6E2DA"))]))
    return t
def footer(c,d):
    c.setStrokeColor(HexColor("#D6E2DA"));c.line(48,43,564,43)
    c.setFont("Helvetica",8);c.setFillColor(HexColor(MUTED))
    c.drawString(48,29,FOOTER);c.drawRightString(564,29,str(d.page))

FOOTER="BLOODMNIST  /  COURSE PROJECT  /  PORTFOLIO EDITION"
metrics=rows("model_metrics.csv")
fig,ax=plt.subplots(figsize=(9,3.4),layout="constrained")
labels=["Accuracy","Macro F1","Macro AUC"]
for i,r in enumerate(metrics):
    vals=[float(r[k]) for k in ["accuracy","macro_f1","macro_auc"]]
    x=[j+(-.18 if i==0 else .18) for j in range(3)]
    bars=ax.bar(x,vals,width=.32,label=r["model"],color=["#9EB7AA",GREEN][i])
    ax.bar_label(bars,labels=[f"{v:.4f}" for v in vals],padding=4,fontsize=10)
ax.set_xticks(range(3),labels);ax.set_ylim(0,1.25);ax.set_ylabel("Score")
ax.legend(loc="upper center",ncol=2,frameon=False,fontsize=10);ax.set_title("Saved test-set performance",loc="left",pad=15,fontweight="bold")
fig.savefig(FIG/"model-comparison.png",dpi=190,facecolor="white");plt.close(fig)
classes=rows("cnn_class_metrics.csv")
fig,ax=plt.subplots(figsize=(9,3.5),layout="constrained")
ax.barh([r["class"] for r in classes],[float(r["f1"]) for r in classes],color=GREEN)
ax.invert_yaxis();ax.set_xlim(0,1.08);ax.set_xlabel("F1 score")
ax.set_title("CNN performance varied across cell types",loc="left",pad=15,fontweight="bold")
fig.savefig(FIG/"cnn-class-f1.png",dpi=190,facecolor="white");plt.close(fig)
story=[p("MACHINE LEARNING  /  MEDICAL IMAGE BENCHMARK","K"),
    p("Blood cell image<br/>classification","T"),
    p("Comparing a Random Forest baseline with a compact convolutional neural network."),
    p("Course project | Cleaned edition of the original BloodMNIST notebooks","S"),
    table([["8 cell classes","11,959 train","3,421 test"],
        ["28 x 28 RGB images","Official training split","Official test split"]],[172,172,172]),
    p("Research question","H"),
    p("Can a small convolutional network improve on a pixel-based Random Forest for classifying eight types of blood cells? The project compares overall discrimination and class-level performance on BloodMNIST."),
    p("Model design","H"),
    table([["Random Forest","Convolutional neural network"],
        ["Flattened RGB inputs in [0, 1]; 100 trees; random_state=42.",
         "Two convolution/pooling blocks (32 and 64 channels), a 128-unit hidden layer and eight logits."],
        ["Classification from raw pixel features.",
         "Adam, learning rate 0.001; batch 128; ten epochs; unweighted cross-entropy; RGB normalization."]],[258,258]),
    p("What the saved run showed","H"),
    p("The CNN achieved 90.76% test accuracy versus 83.10% for the Random Forest, a difference of 7.66 percentage points. Macro F1 increased from 0.8004 to 0.8923. No uncertainty interval or external validation was available."),
    p("Provenance: values come from saved outputs in the original notebooks. Cleaned notebooks have cleared outputs and have not been retrained in this preparation environment.","S"),
    PageBreak(),p("01  /  RESULTS AND REPRODUCIBILITY","K"),
    p("Better overall scores;<br/>uneven class performance","T"),
    Image(str(FIG/"model-comparison.png"),width=516,height=195),
    p("Figure 1. Archived test metrics. AUC is the macro one-vs-rest score; bars are point estimates, without uncertainty intervals.","S"),
    table([["Model","Accuracy","Macro F1","Macro AUC"]]+[
        [r["model"],f'{float(r["accuracy"])*100:.2f}%',r["macro_f1"],r["macro_auc"]] for r in metrics],
        [186,110,110,110]),
    p("Class-level errors","H"),
    p("CNN F1 was 0.9989 for platelets and 0.9809 for eosinophils, compared with 0.7836 for monocytes and 0.7982 for immature granulocytes. These differences motivate class-level evaluation rather than reliance on accuracy alone."),
    p("Reproducibility and next steps","H"),
    p("The repository retains the original notebooks and adds clean entry points, dependency instructions and separate CSV exports for future runs. A CNN seed was added during cleanup, so exact recovery of the old stochastic run is not guaranteed. Use the official validation split for future tuning and keep the test split reserved for final evaluation."),
    p("Source: github.com/jengithubb/DermaMNIST-CNN-project, commit 243db78. Despite its old name, the source uses BloodMNIST. The README's weighted-loss claim and 90.1% score were corrected to match the training cell and saved output. Dataset: medmnist.com. This is a benchmark study, not a clinical decision tool.","S")]
pdf=ROOT/"reports/bloodmnist-report.pdf"
SimpleDocTemplate(str(pdf),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=45,bottomMargin=62,
    title="BloodMNIST Image Classification").build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)
