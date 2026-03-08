"""Generate a side-by-side HTML comparison of evaluation results."""

import json
from pathlib import Path

def generate_html(results_path: Path, questions_path: Path, output_path: Path):
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return

    results = json.loads(results_path.read_text())
    questions = {q["id"]: q for q in json.loads(questions_path.read_text())}

    # Group records by question_id
    comparison = {}
    for record in results["records"]:
        qid = record["question_id"]
        if qid not in comparison:
            comparison[qid] = {}
        
        system_name = record["system"]
        # Just take the first trial for simplicity in visualization, or average them?
        # Let's take trial 1.
        if record["trial"] == 1:
            comparison[qid][system_name] = record

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SemaTree vs RAG Comparison</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 2rem; }
        .question-card { margin-bottom: 2rem; border-left: 5px solid #0d6efd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .system-name { font-weight: bold; color: #495057; border-bottom: 2px solid #dee2e6; margin-bottom: 1rem; padding-bottom: 0.5rem; }
        .answer-text { white-space: pre-wrap; font-size: 0.95rem; background: white; padding: 1rem; border-radius: 4px; border: 1px solid #e9ecef; min-height: 200px; }
        .source-tag { font-size: 0.75rem; margin-right: 0.5rem; }
        .metrics { font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem; }
        .gold-answer { background-color: #e7f3ff; border: 1px dashed #0d6efd; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; font-style: italic; }
    </style>
</head>
<body>
<div class="container">
    <h1 class="mb-4">Evaluation Results Comparison</h1>
    <p class="lead">Comparing <strong>RAG Baseline</strong> vs <strong>SemaTree</strong> side-by-side.</p>
    <hr>
"""

    for qid in sorted(comparison.keys()):
        q_data = questions.get(qid, {"question": "Unknown Question", "gold_answer": ""})
        html_content += f"""
    <div class="card question-card">
        <div class="card-body">
            <h5 class="card-title text-primary">[{qid}] {q_data['question']}</h5>
            <div class="gold-answer">
                <strong>Gold Answer:</strong><br>{q_data['gold_answer']}
            </div>
            <div class="row">
"""
        # Sort systems so they are consistent (RAG left, Agentic right)
        systems = sorted(comparison[qid].keys())
        for sys in systems:
            rec = comparison[qid][sys]
            sources_html = "".join([f'<span class="badge bg-secondary source-tag">{s}</span>' for s in rec["retrieved_sources"]])
            
            html_content += f"""
                <div class="col-md-6">
                    <div class="system-name">{sys}</div>
                    <div class="answer-text">{rec['answer']}</div>
                    <div class="metrics">
                        <strong>Latency:</strong> {rec['latency_ms']:.0f}ms | 
                        <strong>API Calls:</strong> {rec.get('api_calls', 1)} |
                        <strong>Tokens:</strong> {rec['tokens_used']}
                    </div>
                    <div class="mt-2">
                        <strong>Retrieved:</strong><br>
                        {sources_html if sources_html else '<span class="text-muted">None</span>'}
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
    </div>
"""

    html_content += """
</div>
</body>
</html>
"""
    output_path.write_text(html_content)
    print(f"Report generated at: {output_path}")

if __name__ == "__main__":
    results_file = Path("evaluation/results/results.json")
    questions_file = Path("evaluation/dataset/questions.json")
    output_file = Path("evaluation/results/comparison.html")
    generate_html(results_file, questions_file, output_file)
