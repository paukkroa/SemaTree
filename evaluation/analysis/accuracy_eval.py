"""Post-hoc accuracy evaluation for experiment results."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, util

def extract_keywords(text: str) -> set[str]:
    """
    Extract technical keywords/entities from text.
    Focuses on paths, commands, filenames, and capitalized terms.
    """
    # Find paths, filenames, commands
    paths = re.findall(r'[~/][a-zA-Z0-9._/-]+', text)
    commands = re.findall(r'/[a-z-]+', text)
    # Match strings inside backticks, single or double quotes
    quoted = re.findall(r'[`"\']([a-zA-Z0-9._/-]+)[`"\']', text)
    
    # Capitalized terms (likely entities)
    entities = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', text)
    
    keywords = set(paths + commands + quoted + entities)
    # Filter out common short words and noise
    return {k.lower() for k in keywords if len(k) > 2}

def keyword_recall(gold: str, candidate: str) -> float:
    """Calculate what fraction of gold keywords are in the candidate."""
    gold_keywords = extract_keywords(gold)
    if not gold_keywords:
        return 1.0
    
    candidate_lower = candidate.lower()
    matches = sum(1 for k in gold_keywords if k in candidate_lower)
    return matches / len(gold_keywords)

def run_accuracy_analysis(results_path: Path, questions_path: Path, output_path: Path):
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    print(f"Loading questions from {questions_path}...")
    with open(questions_path, 'r') as f:
        questions = {q['id']: q for q in json.load(f)}
        
    print("Initializing SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    records = results.get('records', [])
    failed = [r for r in records if r.get('answer') == 'Could not find answer within step limit.']
    records = [r for r in records if r.get('answer') != 'Could not find answer within step limit.']
    print(f"Analyzing {len(records)} records (excluded {len(failed)} failed runs)")

    # Group results by system
    system_metrics = {}

    for record in records:
        qid = record['question_id']
        gold = questions.get(qid)
        if not gold:
            continue
            
        candidate_answer = record.get('answer', '')
        gold_answer = gold['gold_answer']
        
        # 1. Keyword Recall
        k_recall = keyword_recall(gold_answer, candidate_answer)
        record['keyword_recall'] = k_recall
        
        # 2. Semantic Similarity
        if candidate_answer.strip():
            embeddings = model.encode([gold_answer, candidate_answer], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        else:
            similarity = 0.0
        record['semantic_similarity'] = similarity
        
        # Update system-level stats
        sys = record['system']
        if sys not in system_metrics:
            system_metrics[sys] = {
                'k_recall': [],
                'similarity': [],
                'composite': []
            }
        
        system_metrics[sys]['k_recall'].append(k_recall)
        system_metrics[sys]['similarity'].append(similarity)
        system_metrics[sys]['composite'].append(record.get('composite', 0.0))

    # Summary report
    summary = {}
    for sys, metrics in system_metrics.items():
        summary[sys] = {
            'avg_keyword_recall': float(np.mean(metrics['k_recall'])),
            'avg_semantic_similarity': float(np.mean(metrics['similarity'])),
            'avg_composite_score': float(np.mean(metrics['composite'])),
            'count': len(metrics['k_recall'])
        }
        
    final_output = {
        'summary': summary,
        'records': records
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nAnalysis complete. Summary:")
    for sys, stats in summary.items():
        print(f"\nSystem: {sys}")
        print(f"  Keyword Recall: {stats['avg_keyword_recall']:.3f}")
        print(f"  Semantic Sim:   {stats['avg_semantic_similarity']:.3f}")
        print(f"  LLM Judge Comp: {stats['avg_composite_score']:.3f}")
    
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--questions", type=str, default="evaluation/dataset/questions.json")
    parser.add_argument("--output", type=str, default="evaluation/results/accuracy_analysis.json")
    args = parser.parse_args()
    
    run_accuracy_analysis(Path(args.results), Path(args.questions), Path(args.output))
