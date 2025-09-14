# Bias and Fairness Audit - Presentation Summary
## Healthcare Chatbot Training for Healthcare Applications

### Slide 1: Background & Objectives
**Healthcare Chatbot Challenge:**
- AI-powered patient assistance system
- Symptoms, treatments, appointment scheduling
- Concerns: misinformation, privacy, fair treatment

**Our Focus: Bias and Fairness**
- Evaluate system fairness across demographic groups
- Identify performance discrepancies
- Implement bias mitigation techniques
- Discuss fairness trade-offs for stakeholders

### Slide 2: Dataset & Methodology
**Medical Training Dataset (medteach.csv):**
- 888 medical students
- Demographics: Gender, mother tongue, curriculum year
- Health outcomes: Depression, anxiety, burnout scores
- Target: Depression risk classification (CES-D ≥ 16)

**Fairness Metrics:**
- Statistical Parity Difference (SPD)
- Equalized Odds Difference (EOD)
- Calibration Gap (Brier scores)
- True Positive Rate across groups

### Slide 3: Key Findings - Overall Performance
**Baseline Model Results:**
- AUC: 0.999 (excellent discrimination)
- Accuracy: 98.1%
- Precision: 100%
- Recall: 93.8%

**After Sample Reweighting:**
- AUC: 0.999 (maintained)
- Accuracy: 97.7% (slight decrease)
- Precision: 97.4% (decrease)
- Recall: 95.0% (improvement)

*Trade-off: Better sensitivity but more false positives*

### Slide 4: Demographic Representation Issues
**Gender Distribution:**
- Men, Women, Non-binary representation
- Potential flag rate disparities
- Differential true positive rates

**Language Diversity:**
- French vs. Non-French speakers
- Cultural factors in health assessment
- Linguistic bias concerns

**Academic Year Effects:**
- Training level correlations
- Seniority effects on reporting

*Charts available: sex_counts.png, glang_counts.png, year_counts.png*

### Slide 5: Bias Mitigation Results
**Sample Reweighting Approach:**
- Inverse frequency weighting by gender
- Improved recall (93.8% → 95.0%)
- Reduced precision (100% → 97.4%)
- Slight calibration degradation

**Limitations:**
- Single mitigation technique
- Small sample sizes in some groups
- Proxy labels (screening tools vs. diagnoses)

### Slide 6: Stakeholder Impact Analysis
**Healthcare Workers:**
- Nurses: Differential chatbot interactions
- Doctors: Algorithmic bias in referrals
- Support staff: Varied health outcomes

**Patients:**
- Underserved groups: Less accurate assessments
- Language barriers: Reduced effectiveness
- Cultural differences: Health concept translation

### Slide 7: Critical Recommendations
**Pre-deployment Actions:**
1. **Expand Training Data**
   - Increase underrepresented group representation
   - Include diverse patient populations
   - Ensure minimum sample sizes (n ≥ 100)

2. **Multi-dimensional Bias Testing**
   - Test intersectional biases
   - Implement fairness constraints
   - Use adversarial debiasing

3. **Clinical Validation**
   - Validate against clinical diagnoses
   - Test with diverse patient populations
   - Human-in-the-loop for high-risk cases

### Slide 8: Implementation Roadmap
**Immediate (Pre-deployment):**
- Expand training data diversity
- Implement comprehensive bias testing
- Clinical validation with diverse populations

**Medium-term:**
- Advanced mitigation techniques
- Real-time bias monitoring
- Cultural adaptation

**Long-term:**
- Diverse development team
- Transparency and accountability
- Continuous bias monitoring

### Slide 9: Final Recommendations
**DO NOT DEPLOY** current model without addressing biases

**Key Actions:**
1. Expand training data to include diverse patient populations
2. Implement comprehensive bias monitoring
3. Engage stakeholders in bias mitigation process
4. Validate with clinical data before deployment

**Success Metrics:**
- Fairness across all demographic groups
- Clinical validation results
- Stakeholder acceptance
- Ongoing monitoring systems

### Slide 10: Questions & Discussion
**Discussion Points:**
- How do fairness trade-offs affect different stakeholders?
- What are the implications for healthcare workers?
- How can we balance accuracy with fairness?
- What monitoring systems should be implemented?

**Next Steps:**
- Integrate with other team members' findings
- Develop comprehensive deployment strategy
- Present final recommendations to hospital board

---

*Supporting Materials:*
- Detailed report: Bias_and_Fairness_Report.md
- Analysis code: fairness_audit.py
- Charts: charts/ directory
- Results: outputs/ directory
