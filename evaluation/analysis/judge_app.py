"""Interactive Human-in-the-loop Judging App for Evaluation Results."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SemaTree Judge")

RESULTS_DIR = Path("evaluation/results")
RESULTS_PATHS = {
    "scaled": RESULTS_DIR / "results.json",
    "normal": RESULTS_DIR / "results_normal.json"
}
ANALYSIS_PATHS = {
    "scaled": RESULTS_DIR / "accuracy_analysis_all.json",
    "normal": RESULTS_DIR / "accuracy_analysis_normal.json"
}
QUESTIONS_PATH = Path("evaluation/dataset/questions.json")
AB_RESULTS_PATH = RESULTS_DIR / "ab_test_results.json"

class Rating(BaseModel):
    question_id: str
    system: str
    trial: int
    correctness: float
    completeness: float
    relevance: float
    notes: str = ""
    dataset: str = "scaled"

class ABResult(BaseModel):
    question_id: str
    dataset: str
    winner: str
    loser: str

def load_results():
    all_records = []
    for dataset_name, path in RESULTS_PATHS.items():
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for record in data.get("records", []):
                    # Filter out failure messages
                    if record.get("answer") == "Could not find answer within step limit.":
                        continue
                    record["dataset"] = dataset_name
                    all_records.append(record)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return {"records": all_records}

def save_rating_to_file(rating: Rating):
    path = RESULTS_PATHS.get(rating.dataset)
    if not path or not path.exists():
        return False
    
    data = json.loads(path.read_text())
    found = False
    for record in data["records"]:
        if (record["question_id"] == rating.question_id and 
            record["system"] == rating.system and 
            record["trial"] == rating.trial):
            
            record["human_correctness"] = rating.correctness
            record["human_completeness"] = rating.completeness
            record["human_evaluated"] = True
            found = True
            break
    
    if found:
        path.write_text(json.dumps(data, indent=2))
        return True
    return False

def load_ab_results():
    if AB_RESULTS_PATH.exists():
        try:
            return json.loads(AB_RESULTS_PATH.read_text())
        except:
            return {}
    return {}

def save_ab_result(result: ABResult):
    data = load_ab_results()
    key = f"{result.question_id}_{result.dataset}"
    data[key] = result.dict()
    AB_RESULTS_PATH.write_text(json.dumps(data, indent=2))

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Human Judge - SemaTree</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <style>
        body { padding: 20px; background: #f4f7f6; }
        .comparison-row { display: flex; gap: 20px; margin-bottom: 40px; }
        .system-col { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .gold-box { background: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #0d6efd; white-space: pre-wrap; }
        .answer-text { background: #fafafa; padding: 15px; border: 1px solid #eee; margin-bottom: 10px; min-height: 150px; border-radius: 4px; overflow-x: auto; white-space: normal; line-height: 1.6; }
        .answer-text p { margin-bottom: 1rem; }
        .answer-text p:last-child { margin-bottom: 0; }
        .rating-form { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; }
        .save-btn { width: 100%; }
        .nav-sidebar { position: sticky; top: 20px; max-height: 90vh; overflow-y: auto; }
        
        .ab-answer-box, .deck-card { transition: transform 0.2s; border: 2px solid transparent; }
        .ab-answer-box:hover { transform: translateY(-5px); border-color: #0d6efd; cursor: pointer; }
        .keyboard-hint { font-size: 0.8rem; color: #888; margin-top: 10px; }
        .badge-dataset { font-size: 0.7rem; vertical-align: middle; }
        
        .star-rating { font-size: 2rem; color: #ddd; cursor: pointer; margin: 10px 0; }
        .star-rating .star { transition: color 0.1s; }
        .star-rating .star:hover { color: #ffdb70; }
        .star-rating .active { color: #ffc107; }
        .rating-step { font-weight: bold; color: #0d6efd; border-bottom: 2px solid #0d6efd; padding-bottom: 2px; }
        
        .answer-text h1, .answer-text h2, .answer-text h3 { font-size: 1.2rem; margin-top: 1.2rem; border-bottom: 1px solid #eee; padding-bottom: 4px; margin-bottom: 0.8rem; }
        .answer-text pre { background: #2d2d2d; color: #f8f8f2; padding: 12px; border-radius: 4px; overflow-x: auto; margin-bottom: 1rem; }
        .answer-text code { background: #eee; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 0.9em; }
        .answer-text pre code { background: transparent; padding: 0; color: inherit; font-size: inherit; }
        
        /* Analytics Tab styles */
        .chart-container { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="deck-tab" data-bs-toggle="tab" data-bs-target="#deck-pane" type="button" role="tab">Deck Mode (Blind)</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="ab-tab" data-bs-toggle="tab" data-bs-target="#ab-pane" type="button" role="tab">A/B Blind Test</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="comp-tab" data-bs-toggle="tab" data-bs-target="#comp-pane" type="button" role="tab">Detailed Comparison (Named)</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="viz-tab" data-bs-toggle="tab" data-bs-target="#viz-pane" type="button" role="tab">Analytics Visualization</button>
            </li>
        </ul>

        <div class="tab-content" id="mainTabContent">
            <div class="tab-pane fade show active" id="deck-pane" role="tabpanel">
                <div class="row justify-content-center">
                    <div class="col-md-8">
                        <div id="deck-container">
                            <h2 class="text-center text-muted mt-5">Loading Deck...</h2>
                        </div>
                    </div>
                </div>
            </div>

            <div class="tab-pane fade" id="ab-pane" role="tabpanel">
                <div class="row justify-content-center">
                    <div class="col-md-10">
                        <div id="ab-container">
                            <h2 class="text-center text-muted mt-5">Loading A/B pairs...</h2>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="tab-pane fade" id="comp-pane" role="tabpanel">
                <div class="row">
                    <div class="col-md-2 nav-sidebar">
                        <h4>Questions</h4>
                        <div class="list-group" id="question-list"></div>
                    </div>
                    <div class="col-md-10">
                        <div id="active-question">
                            <h2 class="text-muted text-center mt-5">Select a question to begin judging</h2>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="tab-pane fade" id="viz-pane" role="tabpanel">
                <div class="row">
                    <div class="col-md-12">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h2>Overall Analytics Comparison</h2>
                            <div class="btn-group" role="group">
                                <input type="radio" class="btn-check" name="datasetViz" id="viz-normal" value="normal" checked onchange="refreshViz()">
                                <label class="btn btn-outline-primary" for="viz-normal">Normal Dataset</label>
                                <input type="radio" class="btn-check" name="datasetViz" id="viz-scaled" value="scaled" onchange="refreshViz()">
                                <label class="btn btn-outline-primary" for="viz-scaled">Scaled Dataset</label>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">A/B Blind Preference Score</h4>
                                    <canvas id="abOverallChart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">Evaluation Scores</h4>
                                    <canvas id="humanChart"></canvas>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">Precision (Correctness) by Category</h4>
                                    <canvas id="humanCorrectnessCategoryChart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">Recall (Completeness) by Category</h4>
                                    <canvas id="humanCompletenessCategoryChart"></canvas>
                                </div>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-12">
                                <div class="chart-container">
                                    <h4 class="text-center">A/B Preference by Question Type</h4>
                                    <canvas id="abCategoryChart"></canvas>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">Keyword Match by Category</h4>
                                    <canvas id="keywordCategoryChart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">Overall Automated Metrics</h4>
                                    <canvas id="automatedChart"></canvas>
                                </div>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h4 class="text-center">Semantic Similarity by Category</h4>
                                    <canvas id="categoryChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let resultsData = { records: [] };
        let questionsData = [];
        let analyticsData = {};
        let abPairs = [];
        let abIndex = 0;
        let deckIndex = 0;
        let deckRecords = [];
        let activeTab = 'deck';
        let deckRatingStep = 'correctness';
        
        Chart.register(ChartDataLabels);
        Chart.defaults.set('plugins.datalabels', {
            anchor: 'end',
            align: 'end',
            formatter: (v) => (v > 0 ? v.toFixed(2) : ''),
            font: { size: 10 },
            color: '#333',
            clamp: true,
        });

        let charts = {};

        marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });

        function renderMarkdown(text) {
            if (!text) return "";
            let processed = text;
            processed = processed.replace(/([^\\n])\\n([0-9]+\\. )/g, '$1\\n\\n$2');
            processed = processed.replace(/([^\\n])\\n([-*] )/g, '$1\\n\\n$2');
            processed = processed.replace(/([^\\n])\\n(```)/g, '$1\\n\\n$2');
            return marked.parse(processed);
        }

        function escapeHtml(text) {
            if (!text) return "";
            const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
            return text.replace(/[&<>"']/g, function(m) { return map[m]; });
        }

        async function init() {
            try {
                const res = await fetch('/data');
                const data = await res.json();
                resultsData = data.results || { records: [] };
                questionsData = data.questions || [];
                
                const analyticsRes = await fetch('/analytics');
                analyticsData = await analyticsRes.json();
                
                prepareDeck();
                prepareABPairs();
                renderList();
                renderDeck();
                renderAB();
            } catch (e) {
                console.error("Initialization failed:", e);
                document.querySelectorAll('.text-center.text-muted').forEach(el => el.innerHTML = "Error loading data. Check console.");
            }

            document.getElementById('comp-tab').addEventListener('shown.bs.tab', () => { activeTab = 'comp'; });
            document.getElementById('deck-tab').addEventListener('shown.bs.tab', () => { activeTab = 'deck'; });
            document.getElementById('ab-tab').addEventListener('shown.bs.tab', () => { activeTab = 'ab'; });
            document.getElementById('viz-tab').addEventListener('shown.bs.tab', () => { activeTab = 'viz'; refreshViz(); });
        }

        function refreshViz() {
            const dataset = document.querySelector('input[name="datasetViz"]:checked').value;
            const data = analyticsData[dataset];
            if (!data) return;
            
            renderABOverallViz(data.ab_overall);
            renderABCategoryViz(data.ab_by_category);
            renderAutomatedViz(data.overall);
            renderHumanViz(data.overall);
            renderCategoryViz(data.by_category);
            renderKeywordCategoryViz(data.by_category);
            renderHumanPrecisionCategoryViz(data.by_category);
            renderHumanRecallCategoryViz(data.by_category);
        }

        function renderHumanPrecisionCategoryViz(by_category) {
            const ctx = document.getElementById('humanCorrectnessCategoryChart').getContext('2d');
            const categories = Object.keys(by_category);
            const systems = [...new Set(categories.flatMap(c => Object.keys(by_category[c])))];
            const datasets = systems.map((sys, idx) => ({
                label: sys.split('(')[0],
                data: categories.map(c => by_category[c][sys] ? by_category[c][sys].avg_human_correctness : 0),
                backgroundColor: `hsla(${idx * 137.5 + 90}, 70%, 50%, 0.6)`
            }));
            if (charts.humanPrecisionCategory) charts.humanPrecisionCategory.destroy();
            charts.humanPrecisionCategory = new Chart(ctx, {
                type: 'bar',
                data: { labels: categories, datasets: datasets },
                options: { scales: { y: { beginAtZero: true, max: 1.0, title: { display: true, text: 'Score (0-1)' } } } }
            });
        }

        function renderHumanRecallCategoryViz(by_category) {
            const ctx = document.getElementById('humanCompletenessCategoryChart').getContext('2d');
            const categories = Object.keys(by_category);
            const systems = [...new Set(categories.flatMap(c => Object.keys(by_category[c])))];
            const datasets = systems.map((sys, idx) => ({
                label: sys.split('(')[0],
                data: categories.map(c => by_category[c][sys] ? by_category[c][sys].avg_human_completeness : 0),
                backgroundColor: `hsla(${idx * 137.5 + 180}, 70%, 50%, 0.6)`
            }));
            if (charts.humanRecallCategory) charts.humanRecallCategory.destroy();
            charts.humanRecallCategory = new Chart(ctx, {
                type: 'bar',
                data: { labels: categories, datasets: datasets },
                options: { scales: { y: { beginAtZero: true, max: 1.0, title: { display: true, text: 'Score (0-1)' } } } }
            });
        }

        function renderABOverallViz(ab_overall) {
            const ctx = document.getElementById('abOverallChart').getContext('2d');
            if (charts.abOverall) charts.abOverall.destroy();
            charts.abOverall = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Agentic Preferred', 'RAG Preferred', 'Neither / Equal'],
                    datasets: [{
                        data: [ab_overall.agentic || 0, ab_overall.rag || 0, ab_overall.neither || 0],
                        backgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(201, 203, 207, 0.8)']
                    }]
                },
                options: { plugins: { datalabels: { display: false } } }
            });
        }

        function renderABCategoryViz(ab_by_category) {
            const ctx = document.getElementById('abCategoryChart').getContext('2d');
            const categories = Object.keys(ab_by_category);
            if (charts.abCategory) charts.abCategory.destroy();
            charts.abCategory = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: categories,
                    datasets: [
                        { label: 'Agentic', data: categories.map(c => ab_by_category[c].agentic_pct || 0), backgroundColor: 'rgba(75, 192, 192, 0.6)' },
                        { label: 'RAG', data: categories.map(c => ab_by_category[c].rag_pct || 0), backgroundColor: 'rgba(54, 162, 235, 0.6)' },
                        { label: 'Neither', data: categories.map(c => ab_by_category[c].neither_pct || 0), backgroundColor: 'rgba(201, 203, 207, 0.6)' }
                    ]
                },
                options: {
                    scales: {
                        y: { stacked: true, beginAtZero: true, max: 100, title: { display: true, text: 'Preference %' } },
                        x: { stacked: true }
                    },
                    plugins: { datalabels: { formatter: (v) => (v > 0 ? Math.round(v) + '%' : ''), color: '#fff' } }
                }
            });
        }

        function renderAutomatedViz(overall) {
            const ctx = document.getElementById('automatedChart').getContext('2d');
            const labels = Object.keys(overall);
            const keywordRecall = labels.map(l => overall[l].avg_keyword_recall);
            const semanticSimilarity = labels.map(l => overall[l].avg_semantic_similarity);
            if (charts.automated) charts.automated.destroy();
            charts.automated = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.map(l => l.split('(')[0]),
                    datasets: [
                        { label: 'Keyword Recall', data: keywordRecall, backgroundColor: 'rgba(54, 162, 235, 0.6)' },
                        { label: 'Semantic Similarity', data: semanticSimilarity, backgroundColor: 'rgba(75, 192, 192, 0.6)' }
                    ]
                },
                options: { scales: { y: { beginAtZero: true, max: 1.0 } } }
            });
        }

        function renderHumanViz(overall) {
            const ctx = document.getElementById('humanChart').getContext('2d');
            const labels = Object.keys(overall);
            const correctness = labels.map(l => overall[l].avg_human_correctness);
            const completeness = labels.map(l => overall[l].avg_human_completeness);
            if (charts.human) charts.human.destroy();
            charts.human = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.map(l => l.split('(')[0]),
                    datasets: [
                        { label: 'Correctness (Precision)', data: correctness, backgroundColor: 'rgba(255, 99, 132, 0.6)' },
                        { label: 'Completeness (Recall)', data: completeness, backgroundColor: 'rgba(255, 159, 64, 0.6)' }
                    ]
                },
                options: { scales: { y: { beginAtZero: true, max: 1.0 } } }
            });
        }

        function renderCategoryViz(by_category) {
            const ctx = document.getElementById('categoryChart').getContext('2d');
            const categories = Object.keys(by_category);
            const systems = [...new Set(categories.flatMap(c => Object.keys(by_category[c])))];
            const datasets = systems.map((sys, idx) => ({
                label: sys.split('(')[0],
                data: categories.map(c => by_category[c][sys] ? by_category[c][sys].avg_semantic_similarity : 0),
                backgroundColor: `hsla(${idx * 137.5}, 70%, 50%, 0.6)`
            }));
            if (charts.category) charts.category.destroy();
            charts.category = new Chart(ctx, {
                type: 'bar',
                data: { labels: categories, datasets: datasets },
                options: { scales: { y: { beginAtZero: true, max: 1.0 } } }
            });
        }

        function renderKeywordCategoryViz(by_category) {
            const ctx = document.getElementById('keywordCategoryChart').getContext('2d');
            const categories = Object.keys(by_category);
            const systems = [...new Set(categories.flatMap(c => Object.keys(by_category[c])))];
            const datasets = systems.map((sys, idx) => ({
                label: sys.split('(')[0],
                data: categories.map(c => by_category[c][sys] ? by_category[c][sys].avg_keyword_recall : 0),
                backgroundColor: `hsla(${idx * 137.5 + 45}, 70%, 50%, 0.6)`
            }));
            if (charts.keywordCategory) charts.keywordCategory.destroy();
            charts.keywordCategory = new Chart(ctx, {
                type: 'bar',
                data: { labels: categories, datasets: datasets },
                options: { scales: { y: { beginAtZero: true, max: 1.0 } } }
            });
        }

        function prepareDeck() {
            // Sort sequentially primarily by Question ID,Dataset,System,Trial
            deckRecords = [...resultsData.records].sort((a, b) => {
                const qComp = a.question_id.localeCompare(b.question_id);
                if (qComp !== 0) return qComp;
                const dComp = a.dataset.localeCompare(b.dataset);
                if (dComp !== 0) return dComp;
                const sComp = a.system.localeCompare(b.system);
                if (sComp !== 0) return sComp;
                return a.trial - b.trial;
            });
        }

        function prepareABPairs() {
            const pairs = [];
            const qIds = [...new Set(resultsData.records.map(r => r.question_id))];
            qIds.forEach(qid => {
                ['normal', 'scaled'].forEach(dataset => {
                    const datasetRecs = resultsData.records.filter(r => r.question_id === qid && r.dataset === dataset);
                    const rag = datasetRecs.find(r => r.system.toLowerCase().includes('rag'));
                    const agentic = datasetRecs.find(r => r.system.toLowerCase().includes('agentic'));
                    if (rag && agentic) pairs.push({ qid, dataset, rag, agentic });
                });
            });
            abPairs = pairs.sort(() => Math.random() - 0.5);
        }

        function renderList() {
            const list = document.getElementById('question-list');
            const sortedRecords = [...resultsData.records].sort((a, b) => a.question_id.localeCompare(b.question_id));
            const uniqueEntries = [];
            const seen = new Set();
            sortedRecords.forEach(r => {
                const key = `${r.question_id}-${r.dataset}`;
                if (!seen.has(key)) { uniqueEntries.push({ id: r.question_id, dataset: r.dataset }); seen.add(key); }
            });
            list.innerHTML = uniqueEntries.map(e => `
                <button class="list-group-item list-group-item-action d-flex justify-content-between align-items-center" onclick="showQuestion('${e.id}', 1, '${e.dataset}')">
                    <span class="text-truncate">${e.id}</span>
                    <span class="badge bg-secondary badge-dataset ms-2">${e.dataset}</span>
                </button>
            `).join('');
        }

        function showQuestion(qid, trial, dataset) {
            const container = document.getElementById('active-question');
            const question = questionsData.find(q => q.id === qid) || { question: "Q NOT FOUND", gold_answer: "N/A" };
            const datasetRecs = resultsData.records.filter(r => r.question_id === qid && r.dataset === dataset);
            const allTrials = [...new Set(datasetRecs.map(r => r.trial))].sort();
            const records = datasetRecs.filter(r => r.trial === trial);
            let html = `
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h3>${qid}: ${question.question} <span class="badge bg-secondary">${dataset}</span></h3>
                    <div class="btn-group">
                        <span class="input-group-text">Trial:</span>
                        ${allTrials.map(t => `<button class="btn btn-sm ${t === trial ? 'btn-primary' : 'btn-outline-primary'}" onclick="showQuestion('${qid}', ${t}, '${dataset}')">${t}</button>`).join('')}
                    </div>
                </div>
                <div class="gold-box"><strong>Gold Answer:</strong><br>${question.gold_answer}</div>
                <div class="comparison-row">
            `;
            records.forEach(r => {
                const traceId = `trace-${qid}-${trial}-${r.system.replace(/\W/g, '')}`;
                const traceContent = r.retrieved_content ? escapeHtml(r.retrieved_content) : 'No trace.';
                html += `
                    <div class="system-col">
                        <h5 class="border-bottom pb-2">${r.system}</h5>
                        <div class="answer-text">${renderMarkdown(r.answer)}</div>
                        <div class="small text-muted mb-2">Retrieved: ${(r.retrieved_sources || []).join(', ')}</div>
                        <button class="btn btn-outline-secondary btn-sm mb-3" onclick="toggleTrace('${traceId}')">Trace</button>
                        <div id="${traceId}" style="display:none; background: #222; color: #0f0; padding: 10px; font-family: monospace; font-size: 0.8rem; border-radius: 4px; max-height: 200px; overflow-y: auto; white-space: pre-wrap;">${traceContent}</div>
                        <div class="rating-form">
                            <div class="row g-2">
                                <div class="col-6"><label class="form-label small">Corr (0-5)</label><input type="number" class="form-control form-control-sm" id="score-${qid}-${r.system}-${trial}-corr" value="${r.human_correctness || 0}" step="0.5" min="0" max="5"></div>
                                <div class="col-6"><label class="form-label small">Comp (0-5)</label><input type="number" class="form-control form-control-sm" id="score-${qid}-${r.system}-${trial}-comp" value="${r.human_completeness || 0}" step="0.5" min="0" max="5"></div>
                            </div>
                            <button class="btn btn-primary btn-sm save-btn mt-2" onclick="saveRating('${qid}', '${r.system}', ${trial}, '${dataset}')">Save</button>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        }

        function renderDeck() {
            const container = document.getElementById('deck-container');
            if (deckRecords.length === 0) { container.innerHTML = "<h3 class='text-center mt-5'>No records to display.</h3>"; return; }
            if (deckIndex >= deckRecords.length) {
                container.innerHTML = `<div class="text-center mt-5"><h3>Deck Complete! 🃏</h3><button class="btn btn-primary mt-3" onclick="deckIndex=0; prepareDeck(); renderDeck()">Restart from Beginning</button></div>`;
                return;
            }
            const r = deckRecords[deckIndex];
            const question = questionsData.find(q => q.id === r.question_id) || { question: "Q NOT FOUND", gold_answer: "N/A" };
            const traceId = `deck-trace-${deckIndex}`;
            const traceContent = r.retrieved_content ? escapeHtml(r.retrieved_content) : 'No trace available.';
            
            container.innerHTML = `
                <div class="progress mb-4" style="height: 10px;"><div class="progress-bar" role="progressbar" style="width: ${(deckIndex/deckRecords.length)*100}%"></div></div>
                <div class="d-flex justify-content-between align-items-center mb-2"><h5>Card ${deckIndex + 1} of ${deckRecords.length} <span class="badge bg-secondary">${r.dataset}</span></h5><div class="text-muted small">System: [ANONYMIZED] | Trial ${r.trial}</div></div>
                <div class="card deck-card shadow"><div class="card-body">
                    <h4 class="card-title">${r.question_id}: ${question.question}</h4>
                    <div class="gold-box small"><strong>Gold:</strong> ${question.gold_answer}</div>
                    <div class="answer-text mt-3">${renderMarkdown(r.answer)}</div>
                    
                    <button class="btn btn-outline-secondary btn-sm mb-3" onclick="toggleTrace('${traceId}')">View Trace / Retrieved Content</button>
                    <div id="${traceId}" style="display:none; background: #222; color: #0f0; padding: 15px; font-family: monospace; font-size: 0.8rem; border-radius: 4px; max-height: 300px; overflow-y: auto; white-space: pre-wrap; margin-bottom: 20px;">${traceContent}</div>

                    <div class="mt-4 p-3 border rounded bg-light text-center">
                        <div class="mb-2"><span class="${deckRatingStep === 'correctness' ? 'rating-step' : 'text-muted'}">1. Correctness</span> &nbsp;&rarr;&nbsp; <span class="${deckRatingStep === 'completeness' ? 'rating-step' : 'text-muted'}">2. Completeness</span></div>
                        <div class="star-rating" id="deck-stars">${[1,2,3,4,5].map(n => `<span class="star" onclick="setDeckRating(${n})">★</span>`).join('')}</div>
                        <div class="keyboard-hint mt-2">Press [1-5] to rate. [Left/Right Arrows] to navigate. [Enter] to skip.</div>
                        <div class="mt-2 small text-muted">Rating: Corr: <strong>${r.human_correctness || '-'}</strong> | Comp: <strong>${r.human_completeness || '-'}</strong></div>
                    </div>
                </div></div>
                <div class="text-center mt-3"><button class="btn btn-outline-secondary" onclick="deckIndex--; renderDeck()">&larr; Back</button><button class="btn btn-outline-primary ms-2" onclick="deckIndex++; renderDeck()">Skip &rarr;</button></div>
            `;
            updateStars();
        }

        function setDeckRating(val) {
            const r = deckRecords[deckIndex];
            if (deckRatingStep === 'correctness') { r.human_correctness = val; deckRatingStep = 'completeness'; renderDeck(); }
            else { r.human_completeness = val; saveRating(r.question_id, r.system, r.trial, r.dataset); deckRatingStep = 'correctness'; deckIndex++; renderDeck(); }
        }

        function updateStars() {
            const stars = document.querySelectorAll('#deck-stars .star');
            const r = deckRecords[deckIndex];
            if (!r) return;
            const currentVal = (deckRatingStep === 'correctness') ? (r.human_correctness || 0) : (r.human_completeness || 0);
            stars.forEach((star, idx) => { if (idx < currentVal) star.classList.add('active'); else star.classList.remove('active'); });
        }

        function renderAB() {
            const container = document.getElementById('ab-container');
            if (abPairs.length === 0) { container.innerHTML = `<div class="text-center mt-5"><h3 class="text-muted">No valid A/B pairs found.</h3></div>`; return; }
            if (abIndex >= abPairs.length) {
                container.innerHTML = `<div class="text-center mt-5"><h3>All pairs evaluated! 🎉</h3><button class="btn btn-primary mt-3" onclick="abIndex=0; prepareABPairs(); renderAB()">Restart & Shuffle</button></div>`;
                return;
            }
            const pair = abPairs[abIndex];
            const question = questionsData.find(q => q.id === pair.qid) || { question: "Q NOT FOUND", gold_answer: "N/A" };
            if (!pair.isSwapped) pair.isSwapped = Math.random() > 0.5;
            const systemA = pair.isSwapped ? pair.agentic : pair.rag;
            const systemB = pair.isSwapped ? pair.rag : pair.agentic;
            
            const traceAId = `ab-trace-a-${abIndex}`;
            const traceBId = `ab-trace-b-${abIndex}`;
            const traceAContent = systemA.retrieved_content ? escapeHtml(systemA.retrieved_content) : 'No trace available.';
            const traceBContent = systemB.retrieved_content ? escapeHtml(systemB.retrieved_content) : 'No trace available.';

            container.innerHTML = `
                <div class="progress mb-4" style="height: 10px;"><div class="progress-bar" role="progressbar" style="width: ${(abIndex/abPairs.length)*100}%"></div></div>
                <div class="d-flex justify-content-between align-items-center mb-3"><h3>Blind Test: ${pair.qid} <span class="badge bg-secondary">${pair.dataset}</span></h3><span class="text-muted">Pair ${abIndex + 1} of ${abPairs.length}</span></div>
                <div class="card mb-4 shadow-sm"><div class="card-body">
                    <h5 class="card-title">Question: ${question.question}</h5>
                    <div class="gold-box mb-0 mt-3 small"><strong>Gold Answer:</strong><br>${question.gold_answer}</div>
                </div></div>
                <div class="row g-4">
                    <div class="col-md-6">
                        <div class="card ab-answer-box h-100" onclick="saveAB('A')">
                            <div class="card-body">
                                <h5 class="card-title text-center border-bottom pb-2">Answer A</h5>
                                <div class="answer-text">${renderMarkdown(systemA.answer)}</div>
                            </div>
                            <div class="card-footer text-center text-muted small">Press [A] to select</div>
                        </div>
                        <button class="btn btn-link btn-sm mt-2 d-block mx-auto text-decoration-none" onclick="event.stopPropagation(); toggleTrace('${traceAId}')">Toggle Trace A</button>
                        <div id="${traceAId}" style="display:none; background: #222; color: #0f0; padding: 10px; font-family: monospace; font-size: 0.7rem; border-radius: 4px; max-height: 200px; overflow-y: auto; white-space: pre-wrap; margin-top: 5px;">${traceAContent}</div>
                    </div>
                    <div class="col-md-6">
                        <div class="card ab-answer-box h-100" onclick="saveAB('B')">
                            <div class="card-body">
                                <h5 class="card-title text-center border-bottom pb-2">Answer B</h5>
                                <div class="answer-text">${renderMarkdown(systemB.answer)}</div>
                            </div>
                            <div class="card-footer text-center text-muted small">Press [B] to select</div>
                        </div>
                        <button class="btn btn-link btn-sm mt-2 d-block mx-auto text-decoration-none" onclick="event.stopPropagation(); toggleTrace('${traceBId}')">Toggle Trace B</button>
                        <div id="${traceBId}" style="display:none; background: #222; color: #0f0; padding: 10px; font-family: monospace; font-size: 0.7rem; border-radius: 4px; max-height: 200px; overflow-y: auto; white-space: pre-wrap; margin-top: 5px;">${traceBContent}</div>
                    </div>
                </div>
                <div class="text-center mt-4"><button class="btn btn-outline-secondary me-2" onclick="saveAB('neither')">Neither is good [N]</button><button class="btn btn-link text-decoration-none" onclick="abIndex++; renderAB()">Skip [Right Arrow]</button><div class="keyboard-hint">Use [A], [B], [N] keys to vote. [Left/Right Arrows] to navigate.</div></div>
            `;
        }

        async function saveRating(qid, system, trial, dataset) {
            const r = resultsData.records.find(rec => rec.question_id === qid && rec.system === system && rec.trial === trial && rec.dataset === dataset);
            if (!r) return;
            await fetch('/rate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    question_id: qid, system: system, trial: trial,
                    correctness: r.human_correctness || 0, completeness: r.human_completeness || 0,
                    relevance: 0, notes: "", dataset: dataset
                })
            });
        }

        async function saveAB(choice) {
            const pair = abPairs[abIndex];
            let winner = 'neither';
            if (choice === 'A') winner = pair.isSwapped ? 'agentic' : 'rag';
            else if (choice === 'B') winner = pair.isSwapped ? 'rag' : 'agentic';
            const loser = winner === 'rag' ? 'agentic' : (winner === 'agentic' ? 'rag' : 'neither');
            await fetch('/rate_ab', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ question_id: pair.qid, dataset: pair.dataset, winner: winner, loser: loser })
            });
            abIndex++; renderAB();
        }

        function toggleTrace(id) { const el = document.getElementById(id); el.style.display = el.style.display === 'none' ? 'block' : 'none'; }

        document.addEventListener('keydown', (e) => {
            if (activeTab === 'ab') {
                if (e.key.toLowerCase() === 'a') saveAB('A');
                else if (e.key.toLowerCase() === 'b') saveAB('B');
                else if (e.key.toLowerCase() === 'n') saveAB('neither');
                else if (e.key === 'ArrowRight') { abIndex = Math.min(abIndex + 1, abPairs.length); renderAB(); }
                else if (e.key === 'ArrowLeft') { abIndex = Math.max(0, abIndex - 1); renderAB(); }
            } else if (activeTab === 'deck') {
                if (e.key >= '1' && e.key <= '5') { setDeckRating(parseInt(e.key)); }
                else if (e.key === 'ArrowRight' || e.key === 'Enter') { deckIndex++; deckRatingStep = 'correctness'; renderDeck(); }
                else if (e.key === 'ArrowLeft') { deckIndex = Math.max(0, deckIndex - 1); deckRatingStep = 'correctness'; renderDeck(); }
            }
        });

        init();
    </script>
</body>
</html>
"""

@app.get("/data")
async def get_data():
    results = load_results()
    if not QUESTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="Questions file not found")
    questions = json.loads(QUESTIONS_PATH.read_text())
    return {"results": results, "questions": questions}

@app.get("/analytics")
async def get_analytics():
    # Load all records with human scores
    results_data = load_results()
    df_results = pd.DataFrame(results_data["records"])
    
    # Load analysis records (with automated metrics)
    all_analysis = []
    for dataset_name, path in ANALYSIS_PATHS.items():
        if path.exists():
            data = json.loads(path.read_text())
            recs = data.get("records", [])
            for r in recs:
                r["dataset"] = dataset_name
            all_analysis.extend(recs)
    
    if not all_analysis: return {}
    df_analysis = pd.DataFrame(all_analysis)

    # Drop human score columns from analysis df if present — they came from results.json
    # and will conflict with the dedicated merge below, producing _x/_y suffixes.
    df_analysis = df_analysis.drop(
        columns=[c for c in ['human_correctness', 'human_completeness', 'human_evaluated'] if c in df_analysis.columns]
    )

    # Ensure human score columns exist in df_results before selecting them
    for col in ['human_correctness', 'human_completeness', 'human_evaluated']:
        if col not in df_results.columns:
            df_results[col] = np.nan

    # Merge human scores
    merged = pd.merge(
        df_analysis,
        df_results[['question_id', 'system', 'trial', 'dataset', 'human_correctness', 'human_completeness', 'human_evaluated']],
        on=['question_id', 'system', 'trial', 'dataset'],
        how='left'
    )
    
    # Scale scores to 0-1 range (base 5.0)
    merged['human_correctness'] = merged['human_correctness'] / 5.0
    merged['human_completeness'] = merged['human_completeness'] / 5.0
    
    # Fill automated metrics with 0.0 if missing, but leave scores as NaN where not evaluated
    # so that the mean() function excludes unjudged rows.
    cols_to_fill = ['keyword_recall', 'semantic_similarity']
    merged[cols_to_fill] = merged[cols_to_fill].fillna(0.0)
    
    ab_results = load_ab_results()
    questions = json.loads(QUESTIONS_PATH.read_text())
    q_to_cat = {q['id']: q['category'] for q in questions}
    
    analytics = {}
    for dataset in ['normal', 'scaled']:
        df_ds = merged[merged['dataset'] == dataset]
        if df_ds.empty: 
            analytics[dataset] = {
                "overall": {}, "by_category": {}, "ab_overall": {}, "ab_by_category": {}
            }
            continue
        
        overall = df_ds.groupby('system').agg({
            'keyword_recall': 'mean', 'semantic_similarity': 'mean',
            'human_correctness': 'mean', 'human_completeness': 'mean'
        }).rename(columns={
            'keyword_recall': 'avg_keyword_recall', 'semantic_similarity': 'avg_semantic_similarity',
            'human_correctness': 'avg_human_correctness', 'human_completeness': 'avg_human_completeness'
        }).fillna(0.0).to_dict('index')
        
        by_cat = {}
        for cat in df_ds['category'].unique():
            by_cat[cat] = df_ds[df_ds['category'] == cat].groupby('system').agg({
                'keyword_recall': 'mean', 'semantic_similarity': 'mean',
                'human_correctness': 'mean', 'human_completeness': 'mean'
            }).rename(columns={
                'keyword_recall': 'avg_keyword_recall', 'semantic_similarity': 'avg_semantic_similarity',
                'human_correctness': 'avg_human_correctness', 'human_completeness': 'avg_human_completeness'
            }).fillna(0.0).to_dict('index')
            
        ds_ab = [v for v in ab_results.values() if v.get('dataset') == dataset]
        ab_overall = {"agentic": 0, "rag": 0, "neither": 0}
        ab_cat_counts = {}
        for vote in ds_ab:
            winner, cat = vote['winner'], q_to_cat.get(vote['question_id'], "UNKNOWN")
            if winner not in ab_overall: winner = 'neither'
            ab_overall[winner] += 1
            if cat not in ab_cat_counts: ab_cat_counts[cat] = {"agentic": 0, "rag": 0, "neither": 0, "total": 0}
            ab_cat_counts[cat][winner] += 1
            ab_cat_counts[cat]["total"] += 1
            
        ab_cat_pct = {cat: {
            "agentic_pct": (counts["agentic"] / counts["total"]) * 100,
            "rag_pct": (counts["rag"] / counts["total"]) * 100,
            "neither_pct": (counts["neither"] / counts["total"]) * 100
        } for cat, counts in ab_cat_counts.items() if counts["total"] > 0}
            
        analytics[dataset] = {
            "overall": overall, "by_category": by_cat,
            "ab_overall": ab_overall, "ab_by_category": ab_cat_pct
        }
    return analytics

@app.post("/rate")
async def rate(rating: Rating):
    success = save_rating_to_file(rating)
    if not success: raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "success"}

@app.post("/rate_ab")
async def rate_ab(result: ABResult):
    save_ab_result(result)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8500)
