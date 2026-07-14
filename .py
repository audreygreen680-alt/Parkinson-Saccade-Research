Integrating CNN-Based Saccadic Architecture with Clinical
Biomarkers
Adriana Turcan 1 , Schaefer Grant 2 , Jason Aldred 1,2
1 Coeur d’ Alene Charter Academy, Post Falls, Idaho, 83854,USA
2 Inland Northwest Research, Selkirk Neurology PLLC, Spokane, Washington
3 Inland Northwest Research, Selkirk Neurology PLLC, Spokane, Washington
Student Authors
Adriana Turcan, Coeur d’ Alene Charter
KEYWORDS: Parkinson’s disease, Saccadic eye movements, Convolutional Neural Network,
Dimensionality reduction, Biomarkers

OVERVIEW: This research utilizes a PyTorch deep-learning model by using saccadic eye-
tracking to catch Parkinson’s disease early. We hypothesized that by using ocular metrics with
PPMI clinical datasets along with Psych Archives would help maximize model sensitivity. The
deep learning model isolates pathological eye movement from natural aging achieving a 97.06%
accuracy rate and validating this non-invasive biomarker.

SUMMARY
In 2030, it is estimated that 1.2 million patients in the USA would have Parkinson ’s disease
(PD).Patients lose 60-80% of their neurons before their symptoms are addressed in clinics. This
research identifies saccadic eye movement as an early biomarker assistant before irreversible
neurological damage occurs. This diagnostic gap is tackled in this research by creating a
Multimodal Fusion architecture (MMF) using PyTorch and ResNet 18 to help analyze gaze
coordinate graphs from saccadic hypometria as a PD biomarker. We hypothesized that merging
multiple clinical dataset lead to a significant improvement in model sensitivity compared to
ocular data alone. By merging multiple PPMI datasets along with Psych Archives (saccade
dataset) which then was processed through a Yellow Fusion Block to filter out age-related
biases. The fused dataset coordinates were mapped into latent space using UMAP to visualize
PD projection. The MMF achieved 97.06% accuracy and PR-AUC curve of 0.94 indicates that
saccades proved to be accurate biomarker. The Yellow Fusion Block isolates pathological
staircase patterns in ocular data rather than relying on age. The Density-Based Manifold
Analysis reveals a clear difference between healthy controls vs. PD and this effectively identifies

6
nerve loss through a Cognitive Decline scale. Saccadic Eye movement was supported as viable
biomarker that distinguished PD from natural aging.

INTRODUCTION
Parkinson’s disease (PD) continues to be a growing global health concern as one of the fastest-
growing neurodegenerative disorders on the planet. As the global population ages, the
prevalence of PD is projected to double by 2040, creating an unprecedented strain on
healthcare infrastructures (Dorsey &amp; Bloem, 2018). In the United States alone, the Parkinson’s
population is expanding rapidly, currently affecting 1.1 million people. The fundamental
challenge remains the diagnostic gap. Traditional clinical diagnosis relies on the Unified
Parkinson&#39;s Disease Rating Scale (UPDRS), which research indicates is inherently subjective
and prone to human error (Post et al., 2005). This study seeks to identify objective, early
biomarkers to address this critical delay. The unfortunate reality is by the time motor symptoms
are visible the substantia nigra has already accumulated a massive amount of damage (Sonne
et al., 2024).This research focuses on the loss of dopaminergic neurons and the aggregations of
misfolded alpha-synuclein protein into Lewy bodies, which disrupt cellular function. Because of
this progression, patients will suffer a loss of between 60% and 80% of dopaminergic neurons
before a single tremor is detected (Burré et al., 2024). To bridge this gap, this study will utilize
the oculomotor pathway as an immediate indication of basal ganglia health. The basal ganglia
serve as a neural gate, regulating the transmission of oculomotor commands (Young &amp; Reddy,
2023). Under normal circumstances, the substantia nigra pars reticulata maintains a tonic
inhibition on the superior colliculus to prevent involuntary eye movements (Basso &amp; Sommer,
2012). In a healthy brain, a burst of dopamine releases this hold when a person intends to shift
their gaze (Martin et al., 2011). However, in a brain affected by Parkinson&#39;s, the lack of
dopamine will cause over-inhibition (Ramesh &amp; Arachchige, 2023). This will choke the signal to
the eye muscles, causing patients to undershoot targets—a phenomenon called saccadic
hypometria (Pretegiani &amp; Optican, 2017). These saccadic errors occur in less than a second,
making them hard to identify without expensive equipment (Wang &amp; Kapoor, 2014). This
research proposes a multi-model fusion architecture that utilizes the PPMI dataset to validate a
low-cost alternative to traditional systems. Python scripts converts raw ocular time-series data
into two-dimensional recurrence plots (RP) to visualize hidden periodicities (Marwan et al.,
2007). Texture biomarkers are examined by a CNN implemented in PyTorch and combined with
clinical variables like the Benton Line Orientation (JLO) and Clock Drawing tests. This module
integrates ocular signals from a ResNet-18 architecture with longitudinal clinical data (He et al.,

7
2016). Finally, by employing UMAP, high-dimensional information will be reduced to a two-
dimensional projection to map a continuous cognitive decline gradient (McInnes &amp; Healy, 2018).
This research will prove that open-source information can develop an affordable, objective
diagnostic

RESULTS
The research project will utilize a multi-modal statistical framework to validate the diagnostic
architecture&#39;s ability to distinguish Parkinsonian neurodegeneration from natural aging. The
model’s diagnostic efficacy will be quantified using a confusion matrix to track the classification
of 4,037 total subjects, consisting of 2,453 healthy controls and 1,584 Parkinson’s disease
patients. From these raw counts, the overall accuracy will be calculated using the formula
which currently achieves a benchmark of 97.06%. Sensitivity, or recall,
will be measured to determine the success rate in identifying true Parkinson&#39;s cases using the
formula   targeting a rate of 96.17%. Specificity will be calculated to measure the
ability to correctly identify healthy individuals via the formula at a target of
97.65%. To ensure the reliability of positive results, precision will be determined by the formula
aiming for 96.41%. Ocular time-series data will transition from one-dimensional
signals into two-dimensional texture biomarkers to isolate specific disease indicators. Raw x,y
gaze coordinates will be plotted to compare linear healthy fluid motion against the Parkinson&#39;s
signal, which displays the staircase patterns characteristic of saccadic hypometria. Using Python
scripts, these one-dimensional signals will be transformed into two-dimensional recurrence plots
to enable the convolutional neural network to identify laminarity. The analysis will specifically
screen for staircase blocks within the recurrence plots, which represent the over-inhibition of the
superior colliculus caused by dopamine deficiency. The relationship between digital biomarkers
and clinical severity will be analyzed through feature correlation and dimensionality reduction
techniques. A clinical feature correlation cluster map will be generated to calculate the statistical
relationship between variables using the Pearson correlation coefficient formula   
  specifically examining the Roche Digital App. Score (QRSRESN) alongside
standardized clinical tests like the Benton Line Orientation (JLO_TOTRAW) and Clock

8
Drawing (CLCKTOT) scores. High-dimensional feature vectors will then be projected into a
two-dimensional space using Uniform Manifold Approximation and Projection (UMAP) to
visualize patient data distributions. The resulting manifold will be analyzed by comparing the
distinct, high-density clusters of Parkinson&#39;s patients against the broader spread of healthy
controls to map the severity of neurodegeneration. To confirm that the architecture successfully
filters age-related biases, a precision-recall curve will be generated and analyzed. The study will
focus on maintaining a near-perfect precision up to a recall threshold of 0.96 to demonstrate that
the multi-modal fusion effectively differentiates pathological saccadic hypometria from the
natural ocular decline associated with aging. Finally, a digital biomarker signature will be
visualized using violin plots to show the distribution of Roche Digital Scores across the healthy
control and Parkinson&#39;s groups, providing a clear indication of the model&#39;s ability to separate the
two populations. To validate the model&#39;s performance against disease progression, I performed a
longitudinal comparison between Baseline data and Year 2 follow-ups. I specifically optimized
the classification threshold to prioritize Sensitivity (96.17%) over Accuracy, ensuring the system
catches &#39;Drug Naive&#39; patients in the earliest possible phase. This satisfies the engineering goal of
creating a high-precision screening tool that differentiates pathological saccadic hypometria from
natural, age-related ocular changes.
The results are where you report the observations from your experiment(s). For each
experiment:
o Describe the rationale (why does this experiment make sense to do?)
o Briefly explain how the experiment was done
o This is to help your readers understand the data. Full details will be provided in
“Materials and Methods.”

o State what you observed in your experiment(s), making sure you reference any figures
or tables that contain your results
o Your findings should be as detailed as possible and state exact measurements.
For example, if your experiment involved measuring plant growth in inches, state
the observed growth in average inches instead of saying you saw an
increase/decrease in growth.
o Exact measurements and p-values should be reported in this section.

9
The Results can be divided into subsections with a header (in italics) summarizing the content
of that subsection. This is particularly helpful in keeping information organized if you have
different experiments.

Common mistakes we see are that present tense instead of past tense is used to describe
results and too much focus is placed on figures, statistics, and/or numbers without putting them
in the larger context of the experiment. We recommend that authors think about referencing
these items as they do when citing literature to support claims. An example is given below:

Incorrect:
In Figure 1, we can see that with a p-value of 0.01 seeds planted in soil containing 40 ppm
nitrogen grew significantly taller than those in soil with 20 ppm nitrogen.

Correct:
Seeds planted in soil containing 40 ppm nitrogen grew significantly taller than those in soil with
20 ppm nitrogen (1.7 ± 0.3 in vs. 1.0 ± 0.1 in; p = 0.01, two-sample t-test, Figure 1).

DISCUSSION
The discussion is often the longest section of a manuscript. This section is where you want to
discuss your results and interpret what these results mean.
In conclusion, the experimental results partially supported the hypothesis that a multi-model
architecture could effectively track Parkinson’s disease biomarkers. The findings demonstrate
that while integrating ocular and cognitive data yields a highly precise diagnostic tool, tracking
disease progression remains challenging to the nonlinear nature of neurodegeneration. While
the architecture successfully classified 2,453 healthy controls and 1,584 Parkinson’s patients,
feature-level fusion showed that the data points didn’t scale uniformly. By converting raw X,Y
coordinate X,Y coordinates time-series from Pysch Archives dataset into two-texture biomarker
features-such as laminarity and staircase blocks-we minimized diagnostic error,resulting in only
58 false positives and 63 false negatives. This success demonstrated that high-frequency ocular
signals can mathematically capture the downstream effects of basal ganglia deterioration-
specifically, the over-inhibition of superior colliculus-which remains visible during a standard

10
clinical eye exam. Furthermore, the Precision-Recall curve maintained near perfect precison
P1.0 up to a recall threshold of 0.85. Nevertheless, multi-modal inconsistencies during the
integration phase of the Yellow Fusion Block reveal data limitations that warrant deeper
analysis. While the high precison-recall balance highlights the strength of the system, several
inherent architectural and biological limitations must be addressed. A major factor influencing
these results is the difficulty of filtering natural age-related ocular decline from true Parkinsonian
saccadic hypometria. Though a confounding-filtering algorithm was integrated into the model
structure to mitigate age biases, the non-uniform scaling observed during early feature fusion
suggest that the raw ocular datasets possess variable patterns that require more aggressive
normalization. In addition, this study relies on pre-existing, open source datset from PPMI
database. While highly
The structural variations identified within the Yellow Fusion Block indicate that simply adding
more clinical documents may not linearly improve results; rather, the model would benefit from a
broader array of sequential longitudinal data to track how these specific biomarkers shift as
dopamine depletion worsens over multiple years. While the data demonstrates incredibly strong
correlation between multi-modal ocular-cognitive features and Parkinsonian states, correlation
doesn’t equal causation. These results don’t definitively diagnose an active patient, nor does
this software track the microscopic cellular pathway of α-synuclein misfolding in real time. The
system is designed as an automated screening tool to flag high-risk individuals. This system
can’t replace a clinical neurological evaluation. Several critical question remain regarding the
specificity of the architecture. Future research test the Yellow Block Fusion pipeline against
other neurodegenerative disorders-such as Alzheimer’s disease or progressive supranuclear
palsy-to confirm that the texture biomarkers like laminarity are strictly unique to Parkinsonian
saccadic hypometria rather than general cognitive decline. Additionally, future studies should
transition the pipeline from curated database files and evaluate its performance using low-
frequency data captured from standard consumer webcams. This will determine if the diagnostic
scaling holds true in unconstrained, real-world testing environments. In summary, this
investigation successfully designed and validated a multi-model fusion architecture capable of
identifying early-stage architecture capable of classifying 2,453 healthy subjects and 1,584
Parkinson’s patients. By converting messy eye tracking time series into 2D recurrence plots the
Pytorch-based CNN was able to successfully isolate sub-second ocular abnormalities hidden
from human eyes. When these ocular vectors were fused with normalized spatial-cognitive
metadata within the Yellow Fusion Block, diagnostic sensitivity significantly increased. Finally,
applying UMAP dimensionality reduction projected these high-dimensional features into a clear,

11
tractable Cognitive Decline Gradient. This research attempts to establish a accessible
automated screening system to break down finical barriers for early neurodegeneration

Your discussion should:
o Summarize the experimental results and draw conclusions from these results
o Results must be mentioned in the results section first.
o Discuss factors or limitations that could have influenced your results
o Human error is assumed, so it should not be the primary focus of this section.
o Think about whether your experiments should have had more replicates, different
controls, or if there were better ways to measure your data than what you did.

o Tell the readers about the significance/impact of your results
o Be careful not to overinterpret your results!
 Example of overinterpretation:
 Treating cells in vitro and seeing an effect (i.e., decrease in
glucose) does not indicate that you have found a new compound
that will work in animals or humans in vivo to treat a disease (i.e.,
diabetes).

o Also, remember that correlation does not equal causation.
 Your data may support a hypothesis, but they cannot definitively prove it.
Avoid words like “prove” in your interpretation.
o Discuss remaining scientific questions and potential future studies
o Conclude with a paragraph that briefly summarizes your results and touches on the
overall impacts of your study

Common mistakes seen:
 Short discussion sections. This section should be a significant portion of your manuscript
and not just 1–2 paragraphs.
 Including subsections/subheadings. This section should be in a narrative format with
thoughtful transition sentences between paragraphs. No subheadings should be
included.
 Focus is on limitations of study rather than the larger implications. While limitations
should be discussed, they should not take up the majority of the writing in this section.

12

MATERIALS AND METHODS
I started by requesting seven key clinical documents from the Michael J. Fox Foundation’s PPMI
database. These datasets—including the Benton Line test, Clock Drawing test, Roche App
markers, and enrollment data—provided the multidimensional patient perspective I needed.
After gathering the files, I merged them into a master table to sync the eye-tracking
measurements with their specific clinical scores. This step was essential to ensure all the data
was perfectly aligned for analysis. To turn messy eye-movement data into something a
computer could analyze, I used Python to convert one-dimensional signals into 2D Recurrence
Plots. These plots acted as a visual signature of the patient’s eye health. I then deployed a
ResNet-18 CNN on a GPU to scan these images for laminarity—the staircase-like patterns that
reveal a struggle to focus. This process effectively turned raw coordinates into high-level
saccade vectors, ready for the final fusion.To isolate early-stage neurodegeneration, I applied a
strict inclusion filter: only subjects with a disease duration of less than 2 years were included.
Furthermore, to ensure the &#39;motor glitches&#39; were specific to the basal ganglia and not general
cognitive decline, I excluded any participants with a MoCA (Montreal Cognitive Assessment)
score below 26. I developed a Yellow Fusion Block to merge CNN&#39;s ocular features with Z-
score-standardized clinical data. To see if this fusion actually worked, I used UMAP to condense
thousands of data points into a single, color-coded &quot;Cognitive Decline Gradient.&quot; This
visualization lets me map exactly where a patient sits on the path of decline. Finally, I used
heatmaps and confusion matrices to verify that the model could accurately tell the difference
between healthy aging and Parkinson’s. I used AI as a technical partner throughout the coding
process, specifically to help build the ResNet-18 architecture and the UMAP visualization scripts
in Python. By collaborating with these tools, I was able to quickly prototype the Yellow Fusion
Block and ensure the mathematical alignment between the CNN vectors and clinical Z-scores
was precise. This allowed me to focus less on syntax and more on the engineering logic
required to isolate Parkinsonian biomarkers.
This is usually the most challenging part of the manuscript for our student authors to write, as it
differs the most from what you would do in a school lab report or science fair project. You want
to provide enough details for someone to replicate your experiments, but you should not list
everything in a step-by-step fashion.

The Materials and Methods can be divided into subsections with a header (in italics),
summarizing the content of that subsection. This is particularly helpful in keeping it organized if

13
you have different experiments. However, materials should not be a separate section but
should appear in the context of the methods used for your study. A company and catalog
number should be specified for any unique materials, equipment, and animal strains. There is
no need to mention where generic items were obtained (i.e., a test tube).

If you had any code that was self-generated for your manuscript, then you can either upload it to
a GitHub page (if you have one) and provide an in-text reference citation for your link or include
the full script or Github link as an appendix to your manuscript.

If equations are central to your methods, ensure they follow our journal’s formatting guidelines.

We have provided an example below of a step-by-step protocol and how it was changed into a
protocol for a manuscript:

Incorrect:
1) Grow bacteria up overnight with shaking at 30ºC in 2 mL of broth in a glass test tube
2) Perform a serial dilution on the overnight cultures (1:10 to 1:1000) in sterile water
3) Using a sterile cell spreader, spread out 50 uL of each dilution onto blood agar plates.
4) Incubate at 30ºC overnight.
5) Count the resulting colonies and categorize based on hemolysis of the blood agar.

Correct:
Bacteria samples in glass tubes containing 2 mL LB broth (Sigma, Cat# L2542) were grown at
30ºC overnight with shaking in the SD-ura incubator. Overnight cultures were diluted via serial
dilution from 1:10 to 1:1000 in sterile deionized water. Using a cell spreader, 50 uL of each
dilution spread onto tryptic soy agar (TSA) plates (GrowingLabs, Cat# G60) to ensure even
coating. Then, plates were inverted and incubated at 30ºC overnight. The resulting colonies
were counted.

ACKNOWLEDGMENTS (Optional)
Authorship is considered the highest form of acknowledgement in scientific writing. This means
that anyone in your author list can’t appear in your acknowledgements. If anyone or
anyplace helped you with your research but didn’t meet the requirements for authorship,

14
mention them in this section. If you received any funding for your work, this is where you want to
acknowledge it!

REFERENCES
Basso, M.A., and M.A. Sommer. “Exploring the Role of the Substantia Nigra Pars Reticulata in
Eye Movements.” Neuroscience, vol. 198, Dec. 2011, pp. 205–212,
https://doi.org/10.1016/j.neuroscience.2011.08.026. Accessed 22 Mar. 2021.
Bromberg-Martin, Ethan S., et al. “Dopamine in Motivational Control: Rewarding, Aversive, and
Alerting.” Neuron, vol. 68, no. 5, 9 Dec. 2010, pp. 815–834,
pmc.ncbi.nlm.nih.gov/articles/PMC3032992/, https://doi.org/10.1016/j.neuron.2010.11.022.
Burré, Jacqueline, et al. “Research Priorities on the Role of α‐Synuclein in Parkinson’s Disease
Pathogenesis.” Movement Disorders, 30 June 2024, https://doi.org/10.1002/mds.29897.
Cheng, Hsiao-Chun, et al. “Clinical Progression in Parkinson Disease and the Neurobiology of
Axons.” Annals of Neurology, vol. 67, no. 6, June 2010, pp. 715–725,
https://doi.org/10.1002/ana.21995.
Dorsey, E. Ray, et al. “The Emerging Evidence of the Parkinson Pandemic.” Journal of
Parkinson’s Disease, vol. 8, no. S1, 18 Dec. 2018, pp. S3–S8, https://doi.org/10.3233/jpd-
181474.
Goettker, Alexander. “Using Individual Differences to Understand Saccade-Pursuit Interactions.”
Psycharchives.org, 19 Aug. 2021, www.psycharchives.org/en/item/8961f183-9a78-4a0f-9529-
f02b610fb089, https://hdl.handle.net/20.500.12034/4481. Accessed 18 June 2026.
Goyal, Vinay, et al. “Saccadic Eye Movements in Parkinson′S Disease.” Indian Journal of
Ophthalmology, vol. 62, no. 5, 2014, p. 538, https://doi.org/10.4103/0301-4738.133482.
He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” ArXiv.org, arXiv, 10 Dec.
2015, arxiv.org/abs/1512.03385.
“LONI | | Data Use Application | Summary.” Ida.loni.usc.edu,
ida.loni.usc.edu/collaboration/access/appLicense.jsp.
MARWAN, N, et al. “Recurrence Plots for the Analysis of Complex Systems.” Physics Reports,
vol. 438, no. 5-6, Jan. 2007, pp. 237–329, https://doi.org/10.1016/j.physrep.2006.11.001.
McInnes, Leland, et al. “UMAP: Uniform Manifold Approximation and Projection for Dimension
Reduction.” ArXiv.org, 2018, arxiv.org/abs/1802.03426.
Post, Bart, et al. “Unified Parkinson’s Disease Rating Scale Motor Examination: Are Ratings of
Nurses, Residents in Neurology, and Movement Disorders Specialists Interchangeable?”

15
Movement Disorders: Official Journal of the Movement Disorder Society, vol. 20, no. 12, 1 Dec.
2005, pp. 1577–1584, pubmed.ncbi.nlm.nih.gov/16116612/, https://doi.org/10.1002/mds.20640.
Ramesh, S, and S Perera. “Depletion of Dopamine in Parkinson’s Disease and Relevant
Therapeutic Options: A Review of the Literature.” AIMS Neuroscience, vol. 10, no. 3, 1 Jan.
2023, pp. 200–231, pmc.ncbi.nlm.nih.gov/articles/PMC10567584/,
https://doi.org/10.3934/neuroscience.2023017.
Sonne, James, and Morris R Beato. “Neuroanatomy, Substantia Nigra.” Nih.gov, StatPearls
Publishing, 24 Oct. 2022, www.ncbi.nlm.nih.gov/books/NBK536995/.
Wang, Yong Lin, et al. “Saccadic Eye Movements in Neurological Disease: Cognitive
Mechanisms and Clinical Applications.” Journal of Neurology, vol. 272, no. 8, 27 July 2025,
https://doi.org/10.1007/s00415-025-13275-x.
Young, Carly B, et al. “Neuroanatomy, Basal Ganglia.” Nih.gov, StatPearls Publishing, 24 July
2023, www.ncbi.nlm.nih.gov/books/NBK537141/.

References should be in a modified MLA 8 format.
 Every effort should be made to use primary sources or reputable secondary sources. Try
to avoid/limit referencing Wikipedia or blogs and social media posts.
 References are numbered based on the order in which they are cited in the text. For
example, the first reference you cite should be numbered (1) at the end of the sentence.
The reference corresponding to this in-text reference citation will be listed as the first
reference in a numbered list in the References section. If using a citation manager, style
files are available in our Author Resources.
 Hyperlinks (the link to the webpage) should be included with “https://” for any reference
with a weblink or DOI.
 References with three or more authors should list only the first author, followed by “et al.”
 No hanging indentation should be used.
 See previously published articles to see how Reference lists should be formatted.

Example formats:
Scientific journal articles (only italicize the journal name)
Author name(s). “Title of Article.” Name of Journal, vol. X, no. X, Day Month Year, pp. XX-XX.
https://doi.org/

16

Websites (only italicize the website name)
Author name(s) (if any). “Title of Page or Article.&quot; Website Name. https://webpage url. Accessed
Day Month Year.

Book (only italicize the book title)
Author names(s). “Chapter Name.” Title of Book, edited by Editor name(s), Xth ed., vol. X,
Publisher, Year, pp. XX-XX.
Figures and Figure Titles/Captions
Place your figures in this section with their bolded title and un-bolded caption located below the
figure.

Figure captions are often one of the hardest pieces of a manuscript to write. The basic elements
of a caption are as follows: title, what is shown, methods, statistical tests and values (if
applicable), and replicates.

Example caption:
Figure 1. Herbal formulation (HF1) eliminated PMA-induced overexpression of COX-2 in
MCF-7 cells. Bar graph showing mean ± SD COX-2 mRNA fold change (n=3). MCF-7 breast
cancer cells were grown under either control conditions, in 10ng/mL PMA (COX-2
overexpression), or in 10 ng/mL PMA + 0.3 mg/mL HF1 for 24hrs. One-way ANOVA, *** p &lt;
0.001.

Note: Highlighting provided is for demonstration purposes only and should not be included in
your figure/table titles and captions. Additionally, your figure or table captions should capitalize
the first letter of the first word and then only proper nouns and acronyms after that.

Tables with Titles/Captions
Tables should be placed above their bolded title and un-bolded caption. Make sure your
captions explain any abbreviations used in the table. Any references provided in the table
should also be cited somewhere in the manuscript and given in the table as: Last name (year).

Tables should be in an editable format (i.e., using the “InsertTable” function in Word (not
provided as JPEG, PNG, or TIFF files) and efforts should be made to limit them to one page in

17
size. If tables cannot be made to fit on one page, while still being legible, they will be placed in
the appendix for the manuscript.

Note: Only a maximum of 8 figures and tables total is allowed for JEI articles. Supplemental
figures/tables should not be added unless explicitly necessary for understanding the manuscript.
If adding supplementary figures/tables, include them in the Appendix in addition to a statement
justifying their inclusion.

Appendix (If applicable)
JEI allows appendices on a case-by-case basis depending on the type of research being
presented.

If you feel your work requires an appendix, please include it along with a separate statement of
why this information is necessary to include as an appendix rather than in the actual manuscript.

Note: Code/programming script or Github links do not need any justification statement provided.
