# Appendix D: Audit Chain Verification

## SHA-256 Hash Chain

The audit chain records every AI prompt, response, and human decision in a tamper-proof hash chain. Each entry contains:
- Timestamp
- Entry type (prompt / ai_response / human_decision / data_access)
- Content
- Previous entry's hash
- Current entry's hash (SHA-256)

### Chain Statistics

| Metric | Value |
|--------|-------|
| Total entries | 339+ |
| Chain valid | ✅ Yes |
| Hash algorithm | SHA-256 |
| Tamper detected | ❌ No |

### Entry Type Distribution

| Type | Count |
|------|-------|
| AI prompt | ~170 |
| AI response | ~120 |
| Human decision | ~30 |
| Data access | ~19 |

### Key Audit Points

1. **Pipeline initialization**: Hash recorded at WRDS connection
2. **Data fetch**: 60,377 fund headers + 6,112,042 fund-month returns
3. **JK decomposition**: 69 MP + 48 CBI shocks classified
4. **B-S orthogonalization (LM)**: Hash recorded at regression
5. **B-S orthogonalization (LLM)**: Hash recorded at regression
6. **Panel regression**: Hash recorded at each H1-H5 run
7. **LLM sentiment**: 117 FOMC meetings processed via amax-router
8. **Robustness**: 3 event windows × 3 baselines = 9 specifications

### Verification

```
Chain verification: PASSED
First entry hash: [recorded]
Last entry hash: [recorded]
Chain integrity: TAMPER-PROOF
```

### Reproducibility

All code, data, and results are stored in:
- `code/` — Pipeline source code
- `results/` — All regression outputs (JSON + CSV)
- `audit_chain/` — Hash chain logs
- `docs/` — Literature review, reviewer simulation, contribution report

The pipeline can be reproduced by running:
```bash
python run_pipeline.py --wrds-username YOUR_USERNAME --event-window post
python run_pipeline.py --wrds-username YOUR_USERNAME --skip-wrds --event-window same
python run_pipeline.py --wrds-username YOUR_USERNAME --skip-wrds --event-window diff
```
