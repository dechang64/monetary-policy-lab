# -*- coding: utf-8 -*-
"""
Contribution Tracker — Human vs AI Line Attribution
For AFA 2027 GenAI Special Session compliance.

Tracks which lines of code/writing/documentation were contributed by
humans vs AI, producing the required "fraction of project lines 
contributed by humans as opposed to AI" report.

Usage:
    from contribution_tracker import ContributionTracker
    tracker = ContributionTracker("direction2")
    
    # When AI generates code:
    tracker.record_file("code/analysis.py", author="ai", model="claude-sonnet-4")
    
    # When human edits a file:
    tracker.record_file("code/analysis.py", author="human", 
                        human_author="Dechang Xu", note="Fixed variable name")
    
    # Generate report:
    tracker.generate_report("docs/contribution_report.md")
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from collections import defaultdict


class ContributionTracker:
    def __init__(self, project_name="direction2", base_dir="."):
        self.project = project_name
        self.base_dir = base_dir
        self.tracking_file = os.path.join(base_dir, "audit_chain", 
                                          f"{project_name}_contributions.jsonl")
        self.records = []
        self._load()
    
    def _load(self):
        """Load existing contribution records."""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                self.records = [json.loads(line) for line in f if line.strip()]
    
    def _save(self):
        """Save contribution records."""
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    def record_file(self, filepath, author="ai", model=None, 
                    human_author=None, note=None, lines=None):
        """
        Record a file's contribution.
        
        Args:
            filepath: Relative path to file
            author: "ai" or "human"
            model: AI model name (if author="ai")
            human_author: Human author name (if author="human")
            note: Description of what was done
            lines: Number of lines (auto-counted if None)
        """
        full_path = os.path.join(self.base_dir, filepath)
        
        if lines is None and os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
        
        # Compute file hash for tracking changes
        file_hash = None
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filepath": filepath,
            "author": author,
            "model": model,
            "human_author": human_author,
            "note": note,
            "lines": lines or 0,
            "file_hash": file_hash,
        }
        
        self.records.append(record)
        self._save()
        return record
    
    def generate_report(self, output_path=None):
        """
        Generate the contribution report required by AFA GenAI session.
        
        Report includes:
        1. Total lines by author type (human vs AI)
        2. Breakdown by file type (code, writing, documentation)
        3. Breakdown by individual file
        4. Time log of human activity
        """
        # Aggregate by file (latest version counts)
        file_latest = {}
        for r in self.records:
            fp = r["filepath"]
            if fp not in file_latest or r["timestamp"] > file_latest[fp]["timestamp"]:
                file_latest[fp] = r
        
        # Categorize files
        categories = {
            "code": [".py", ".r", ".sh", ".sql"],
            "writing": [".md", ".tex", ".docx"],
            "documentation": [".txt", ".json", ".yaml", ".yml"],
        }
        
        stats = defaultdict(lambda: {"human_lines": 0, "ai_lines": 0, "files": []})
        
        for filepath, record in file_latest.items():
            ext = os.path.splitext(filepath)[1].lower()
            
            cat = "other"
            for cname, exts in categories.items():
                if ext in exts:
                    cat = cname
                    break
            
            if record["author"] == "human":
                stats[cat]["human_lines"] += record["lines"]
            else:
                stats[cat]["ai_lines"] += record["lines"]
            stats[cat]["files"].append({
                "path": filepath,
                "author": record["author"],
                "lines": record["lines"],
                "note": record.get("note", ""),
            })
        
        # Generate report text
        total_human = sum(s["human_lines"] for s in stats.values())
        total_ai = sum(s["ai_lines"] for s in stats.values())
        total = total_human + total_ai
        
        report = []
        report.append("# Contribution Report — AFA 2027 GenAI Special Session")
        report.append(f"# Project: Monetary Policy Shocks and Portfolio Reallocation")
        report.append(f"# Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append("")
        report.append("## Overall Summary")
        report.append("")
        report.append(f"| Category | Human Lines | AI Lines | Total | Human % |")
        report.append(f"|----------|------------|---------|-------|---------|")
        
        for cat in ["code", "writing", "documentation", "other"]:
            s = stats.get(cat, {"human_lines": 0, "ai_lines": 0, "files": []})
            cat_total = s["human_lines"] + s["ai_lines"]
            human_pct = (s["human_lines"] / cat_total * 100) if cat_total > 0 else 0
            report.append(f"| {cat} | {s['human_lines']} | {s['ai_lines']} | {cat_total} | {human_pct:.1f}% |")
        
        if total > 0:
            report.append(f"| **TOTAL** | **{total_human}** | **{total_ai}** | **{total}** | **{total_human/total*100:.1f}%** |")
        else:
            report.append(f"| **TOTAL** | **0** | **0** | **0** | **N/A** |")
        report.append("")
        
        report.append("## File-Level Breakdown")
        report.append("")
        for cat in ["code", "writing", "documentation", "other"]:
            s = stats.get(cat, {"files": []})
            if not s["files"]:
                continue
            report.append(f"### {cat.title()}")
            report.append("")
            report.append("| File | Author | Lines | Note |")
            report.append("|------|--------|-------|------|")
            for f in s["files"]:
                report.append(f"| {f['path']} | {f['author']} | {f['lines']} | {f['note']} |")
            report.append("")
        
        # Human activity time log
        human_records = [r for r in self.records if r["author"] == "human"]
        if human_records:
            report.append("## Human Activity Time Log")
            report.append("")
            report.append("| Timestamp | File | Author | Note |")
            report.append("|-----------|------|--------|------|")
            for r in human_records:
                report.append(f"| {r['timestamp']} | {r['filepath']} | {r.get('human_author', 'N/A')} | {r.get('note', '')} |")
            report.append("")
        
        report_text = '\n'.join(report)
        
        if output_path:
            full_output = os.path.join(self.base_dir, output_path)
            os.makedirs(os.path.dirname(full_output), exist_ok=True)
            with open(full_output, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ Contribution report saved to {output_path}")
        
        return report_text


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    tracker = ContributionTracker("direction2", base_dir=project_dir)
    
    # Count lines in each file
    files_to_record = [
        ("code/audit_chain.py", "AI-generated audit chain system for GenAI compliance"),
        ("code/wrds_connector.py", "AI-generated WRDS data connector"),
        ("code/h1_h4_regression.py", "AI-generated H1-H4 regression analysis"),
        ("code/contribution_tracker.py", "AI-generated contribution tracking system"),
        ("code/run_pipeline.py", "AI-generated master pipeline"),
        ("code/paper_skeleton.py", "AI-generated paper skeleton"),
    ]
    
    for filepath, note in files_to_record:
        full_path = os.path.join(project_dir, filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            tracker.record_file(filepath, author="ai", 
                                model="claude-sonnet-4", 
                                note=note, lines=lines)
    
    # Generate initial report
    report = tracker.generate_report("docs/contribution_report.md")
    print(report[:2000])
