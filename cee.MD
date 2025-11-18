## Customer Engagement Engine(CEE) - build predictive models that drive personalization and optimization within CEE.

### 1. Datasets Provided
- campaigns.csv – Campaign metadata (channel, launch date, objective, audience size)
- engagements.csv – Monthly campaign performance (sent, opened, clicked, converted, revenue)
- customers.csv – Customer attributes (tier, tenure, preferred channel, active status)
### 2. Tasks
A. Data Exploration & Feature Engineering
- Join datasets to create a customer × campaign modeling table.
- Engineer features: engagement history, tenure, preferred channel, etc.
- Explore patterns across segments (tier, channel, objective).

B. Predictive Modeling
- Build a classification model to predict conversion likelihood (1 = converted, 0 = not).
- Evaluate using metrics (AUC, precision, recall, F1).
- Identify top 5 most predictive features.

C. Interpretation & Business Application
- Explain what drives engagement or conversion.
- Recommend how the CEE could use these insights to prioritize customers or channels.
- Suggest how the model could integrate into CEE for personalization or targeting.

### 3. Deliverables
1. Jupyter Notebook (.ipynb) with code and outputs.
2. Executive Summary (.pdf or .docx) covering: approach, key findings, and recommendations.
3. Optional: Model artifact (.pkl) to demonstrate deployment readiness.
### 4. Evaluation Criteria
- Data Preparation & Feature Engineering – 25%
- Modeling Technique & Justification – 25%
- Interpretability & Insights – 30%
- Business Application – 15%
- Communication & Clarity – 5%
### 5. Submission
- Deadline: within 7 calendar days of receiving the assessment.
- File names:
[YourName]_CEE_ML_Assessment.ipynb
[YourName]_CEE_ExecutiveSummary.pdf