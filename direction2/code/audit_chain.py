# -*- coding: utf-8 -*-
"""
Audit Chain — AI Conversation Logger
For AFA 2027 GenAI Special Session compliance.

Records every LLM prompt, response, human edit, and timestamp
in a SHA-256 hash chain for tamper-proof documentation.

Usage:
    from audit_chain import AuditChain
    chain = AuditChain("direction2")
    chain.log_prompt("Analyze fund flows around FOMC dates")
    chain.log_ai_response("Here is the analysis...")
    chain.log_human_edit("Changed window from [-5,+5] to [-3,+3]", author="Dechang Xu")
"""

import hashlib
import json
import os
from datetime import datetime, timezone

class AuditChain:
    def __init__(self, project_name="direction2", base_dir="."):
        self.project = project_name
        self.chain_file = os.path.join(base_dir, "audit_chain", f"{project_name}_chain.jsonl")
        self.prev_hash = "0" * 64
        self._load_last_hash()
    
    def _load_last_hash(self):
        """Load the last hash from existing chain file."""
        if os.path.exists(self.chain_file):
            with open(self.chain_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    self.prev_hash = last["hash"]
    
    def _hash_entry(self, entry):
        """Create SHA-256 hash of entry + previous hash."""
        content = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256((self.prev_hash + content).encode('utf-8')).hexdigest()
    
    def _append(self, entry_type, content, author=None, metadata=None):
        """Append an entry to the chain."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": entry_type,  # "prompt" | "ai_response" | "human_edit" | "human_decision" | "code_commit" | "data_access"
            "content": content,
            "author": author,
            "metadata": metadata or {},
            "prev_hash": self.prev_hash,
        }
        entry["hash"] = self._hash_entry(entry)
        
        os.makedirs(os.path.dirname(self.chain_file), exist_ok=True)
        with open(self.chain_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        self.prev_hash = entry["hash"]
        return entry["hash"]
    
    def log_prompt(self, prompt_text, metadata=None):
        """Log an LLM prompt from human user."""
        return self._append("prompt", prompt_text, author="human", metadata=metadata)
    
    def log_ai_response(self, response_text, model="unknown", metadata=None):
        """Log an AI/LLM response."""
        meta = {"model": model}
        if metadata:
            meta.update(metadata)
        return self._append("ai_response", response_text, author="ai", metadata=meta)
    
    def log_human_edit(self, description, author="Dechang Xu", metadata=None):
        """Log a direct human contribution (code edit, writing, decision)."""
        return self._append("human_edit", description, author=author, metadata=metadata)
    
    def log_human_decision(self, decision, author="Dechang Xu", metadata=None):
        """Log a human research decision (model spec, variable choice, etc.)."""
        return self._append("human_decision", decision, author=author, metadata=metadata)
    
    def log_code_commit(self, files_changed, description, author="ai", metadata=None):
        """Log a code commit/change."""
        meta = {"files": files_changed}
        if metadata:
            meta.update(metadata)
        return self._append("code_commit", description, author=author, metadata=meta)
    
    def log_data_access(self, source, query, rows_returned=None, metadata=None):
        """Log a data access event (WRDS query, file read, etc.)."""
        meta = {"source": source, "query": query}
        if rows_returned is not None:
            meta["rows_returned"] = rows_returned
        if metadata:
            meta.update(metadata)
        return self._append("data_access", "Data fetched", author="system", metadata=meta)
    
    def verify_chain(self):
        """Verify the integrity of the entire chain."""
        with open(self.chain_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        prev = "0" * 64
        for i, line in enumerate(lines):
            entry = json.loads(line)
            stored_hash = entry.pop("hash")
            content = json.dumps(entry, sort_keys=True, ensure_ascii=False)
            computed = hashlib.sha256((prev + content).encode('utf-8')).hexdigest()
            if computed != stored_hash:
                return False, f"Chain broken at entry {i}"
            prev = stored_hash
        
        return True, f"Chain valid ({len(lines)} entries)"
    
    def summary(self):
        """Return summary statistics for the chain."""
        with open(self.chain_file, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f]
        
        types = {}
        authors = {}
        for e in entries:
            t = e["type"]
            types[t] = types.get(t, 0) + 1
            a = e.get("author", "unknown")
            authors[a] = authors.get(a, 0) + 1
        
        return {
            "total_entries": len(entries),
            "by_type": types,
            "by_author": authors,
            "first_timestamp": entries[0]["timestamp"] if entries else None,
            "last_timestamp": entries[-1]["timestamp"] if entries else None,
        }


if __name__ == "__main__":
    # Initialize audit chain for Direction 2
    # base_dir should be the parent (direction2/), not code/
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    chain = AuditChain("direction2", base_dir=project_dir)
    
    # Log the initial prompt (this conversation, 2026-07-25)
    chain.log_prompt(
        "Research Direction 2: Portfolio Rebalancing and Cross-Asset Contagion. "
        "Investigate how mutual fund flows across 7 asset classes respond to "
        "FOMC monetary policy shocks (MP) vs information shocks (CBI) "
        "using JK decomposition. Test H1 (Risk-Off), H2 (Risk-On), "
        "H3 (Asymmetry), H4 (Risk-Ladder Substitution).",
        metadata={"date": "2026-07-25", "session": "AFA GenAI session kickoff"}
    )
    
    chain.log_ai_response(
        "Direction 2 execution plan created. Code framework, audit chain, "
        "and WRDS connector initialized. Ready for data access.",
        model="claude-sonnet-4-20250514"
    )
    
    chain.log_human_decision(
        "Author order: Eileen Zhang & Dechang Xu. "
        "WRDS access confirmed. Full push for AFA GenAI session 8/31 deadline.",
        author="Dechang Xu"
    )
    
    print(f"Audit chain initialized: {chain.chain_file}")
    valid, msg = chain.verify_chain()
    print(f"Verification: {msg}")
    print(f"Summary: {json.dumps(chain.summary(), indent=2)}")
