
# Text Mining for Topic Modeling (LDA)

## Objective  
Explore **unsupervised topic modeling** on news articles and evaluate the stability and interpretability of topics using **Latent Dirichlet Allocation (LDA)**.  

---

## Dataset  
- **20 Newsgroups dataset** (`sklearn.datasets`)  
- Collection of ~20,000 newsgroup documents  
- Subset of **3 categories** used:  
  - `sci.space` (space-related news)  
  - `rec.sport.baseball` (sports news)  
  - `comp.graphics` (computer graphics discussions)  

---

## Methodology  

1. **Preprocessing**  
   - Removed headers, footers, and quotes  
   - Tokenized using `nltk`  
   - Lowercased and removed non-alphabetic tokens  
   - Removed English stopwords  
   - Filtered words shorter than 4 characters  

2. **Corpus & Dictionary**  
   - Built a **dictionary** (word-to-ID mapping)  
   - Created a **bag-of-words corpus** for Gensim  

3. **LDA Topic Modeling**  
   - Trained LDA models with different topic numbers (`k = 2–6`)  
   - Used 10 passes for convergence  

4. **Evaluation**  
   - Measured **coherence score (c_v)** to assess interpretability  
   - Compared coherence across topic numbers  

5. **Visualization**  
   - Displayed top words per topic  
   - Plotted **coherence score vs. topic number**  

---

## Results  

### Example Discovered Topics
- **Topic 0 (sci.space):** `["space", "orbit", "earth", "nasa", "satellite"]`  
- **Topic 1 (rec.sport.baseball):** `["team", "season", "baseball", "game", "players"]`  
- **Topic 2 (comp.graphics):** `["image", "graphics", "software", "computer", "file"]`  

### Coherence Scores  
- Coherence improved when moving from 2 → 3 → 4 topics  
- Dropped slightly for >5 topics, indicating **over-fragmentation**  
- Optimal topics: **3–4**, aligning with dataset categories  

---

## Key Takeaways  
- **LDA** effectively discovers latent topics in text  
- **Coherence score** is a reliable metric for topic stability  
- Applications include:  
  - News aggregation  
  - Research literature review  
  - Social media text analysis  

---

## Tools & Libraries  
- Python  
- NLTK – text preprocessing, stopwords  
- Gensim – LDA modeling, coherence evaluation  
- Matplotlib – visualizations  
- Scikit-learn – dataset loader  
