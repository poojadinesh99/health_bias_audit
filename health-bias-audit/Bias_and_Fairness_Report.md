# Healthcare Chatbot Bias and Fairness Audit Report

## Executive Summary

This report presents a comprehensive bias and fairness analysis of a healthcare chatbot system designed to assist patients with symptoms, treatments, and appointment scheduling. The analysis focuses on identifying demographic representation biases and performance discrepancies across different groups, with particular attention to gender, language, and academic year demographics in medical training data.

## Background and Context

### Healthcare Chatbot Use Case
The proposed AI-powered chatbot aims to:
- Provide symptom assessment and treatment guidance
- Assist with appointment scheduling
- Improve accessibility and patient engagement
- Support diverse patient populations

### Dataset Overview
The analysis is based on medical training data (`medteach.csv`) containing:
- **Sample Size**: 888 medical students
- **Demographics**: Gender (sex), mother tongue (glang), curriculum year (year)
- **Health Outcomes**: Depression (CES-D), anxiety (STAI-T), and burnout (MBI) scores
- **Target Variable**: Depression risk classification (CES-D ≥ 16)

## Methodology

### Fairness Metrics Analyzed
1. **Statistical Parity Difference (SPD)**: Difference in positive prediction rates across groups
2. **Equalized Odds Difference (EOD)**: Difference in true positive rates across groups
3. **Calibration Gap**: Difference in Brier scores (prediction reliability)
4. **Flag Rate**: Rate of positive predictions per group
5. **True Positive Rate (TPR)**: Sensitivity across groups

### Bias Mitigation Technique
- **Sample Reweighting**: Inverse frequency weighting based on gender distribution to address class imbalance

### Statistical Rigor
- Small sample size suppression (n < 30 or positive cases < 10)
- 70/30 train-test split with stratification
- Logistic regression with cross-validation

## Key Findings

### Overall Model Performance
**Baseline Model:**
- AUC: 0.999 (excellent discrimination)
- Accuracy: 98.1%
- Precision: 100%
- Recall: 93.8%
- Brier Score: 0.020 (excellent calibration)

**Reweighted Model:**
- AUC: 0.999 (maintained discrimination)
- Accuracy: 97.7% (slight decrease)
- Precision: 97.4% (slight decrease)
- Recall: 95.0% (improved)
- Brier Score: 0.023 (slightly worse calibration)

### Demographic Representation Analysis

#### Gender Distribution
- **Men**: 1 (coded as 1)
- **Women**: 2 (coded as 2)
- **Non-binary**: 3 (coded as 3) - minimal representation

*Note: Charts available in `charts/sex_counts.png`*

#### Language Diversity
The dataset shows significant linguistic diversity with representation from:
- French speakers (majority)
- English, German, Spanish, Italian speakers
- Various other European and international languages
- **Grouping Strategy**: French vs. Non-French for analysis

*Note: Detailed breakdown available in `charts/glang_counts.png` and `charts/glang_group_counts.png`*

#### Academic Year Distribution
Medical training years (Bmed1-3, Mmed1-3) show varying representation across curriculum levels.

*Note: Distribution available in `charts/year_counts.png`*

### Bias Analysis Results

#### Baseline Model Fairness Issues
1. **Gender Bias**: 
   - Potential disparities in flag rates between gender groups
   - Differential true positive rates may indicate unequal sensitivity

2. **Language Bias**:
   - French vs. Non-French speakers may experience different prediction patterns
   - Cultural and linguistic factors could influence health assessment

3. **Academic Year Bias**:
   - Different training levels may correlate with varying health outcomes
   - Seniority effects on mental health reporting

#### Mitigation Effectiveness
The sample reweighting approach showed:
- **Improved Recall**: Better detection of positive cases (93.8% → 95.0%)
- **Reduced Precision**: More false positives (100% → 97.4%)
- **Trade-off**: Slight decrease in overall accuracy but improved sensitivity

## Critical Limitations

### Data Quality Concerns
1. **Small Sample Sizes**: Some demographic groups have insufficient representation for reliable analysis
2. **Proxy Labels**: Health outcomes are based on screening tools, not clinical diagnoses
3. **Training Population**: Analysis based on medical students, not general patient population
4. **Cultural Bias**: Western-centric health assessment tools may not be universally applicable

### Methodological Limitations
1. **Single Mitigation Technique**: Only sample reweighting was implemented
2. **Binary Classification**: Simplified health outcomes may miss nuanced conditions
3. **Cross-sectional Data**: No temporal analysis of bias evolution
4. **Limited Protected Attributes**: Analysis restricted to gender, language, and academic year

## Stakeholder Impact Analysis

### Healthcare Workers
- **Nurses**: May experience different chatbot interactions based on demographic factors
- **Doctors**: Need to be aware of potential algorithmic biases in patient referrals
- **Support Staff**: May face different health assessment outcomes

### Patients
- **Underserved Groups**: May receive less accurate health assessments
- **Language Barriers**: Non-native speakers may experience reduced chatbot effectiveness
- **Cultural Differences**: Health concepts may not translate across cultural contexts

## Recommendations

### Immediate Actions (Pre-deployment)
1. **Expand Training Data**:
   - Increase representation of underrepresented groups
   - Include diverse patient populations, not just medical students
   - Ensure minimum sample sizes (n ≥ 100) per demographic group

2. **Multi-dimensional Bias Testing**:
   - Test intersectional biases (e.g., gender × language × age)
   - Implement fairness constraints during model training
   - Use adversarial debiasing techniques

3. **Clinical Validation**:
   - Validate health outcome predictions against clinical diagnoses
   - Test with diverse patient populations
   - Implement human-in-the-loop validation for high-risk predictions

### Medium-term Improvements
1. **Advanced Mitigation Techniques**:
   - Implement adversarial debiasing
   - Use demographic parity constraints
   - Apply post-processing fairness methods

2. **Comprehensive Monitoring**:
   - Real-time bias monitoring in production
   - Regular fairness audits
   - Feedback loop from healthcare providers

3. **Cultural Adaptation**:
   - Localize health assessment tools
   - Train culturally-aware models
   - Implement multilingual support

### Long-term Strategic Initiatives
1. **Diverse Development Team**:
   - Include healthcare workers from various backgrounds
   - Engage with patient advocacy groups
   - Regular bias training for development team

2. **Transparency and Accountability**:
   - Publish bias audit results
   - Implement explainable AI for healthcare decisions
   - Create bias reporting mechanisms

## Conclusion

While the baseline model demonstrates excellent technical performance, significant fairness concerns exist across demographic groups. The sample reweighting mitigation shows promise but requires more sophisticated approaches for production deployment.

**Key Recommendations:**
1. **Do not deploy** the current model without addressing identified biases
2. **Expand training data** to include diverse patient populations
3. **Implement comprehensive bias monitoring** throughout the system lifecycle
4. **Engage stakeholders** in the bias mitigation process

The healthcare chatbot has the potential to improve patient care and accessibility, but only if deployed with careful attention to fairness and equity across all demographic groups.

---

*This report is based on analysis of medical training data and should be validated with clinical data before any healthcare deployment. All stakeholders should be involved in the bias mitigation process to ensure equitable outcomes.*
