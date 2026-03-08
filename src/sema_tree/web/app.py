"""FastAPI web application for managing SemaTree."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sema_tree.models import SemaTree, IndexNode
from sema_tree.search import search_index

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SemaTree", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
_index: SemaTree | None = None
_index_path: str | None = None


def load_index(path: str) -> None:
    global _index, _index_path
    _index = SemaTree.load(path)
    _index_path = path


def _get_index() -> SemaTree:
    if _index is None:
        raise HTTPException(status_code=400, detail="No index loaded")
    return _index


def _save_index() -> None:
    if _index and _index_path:
        _index.save(_index_path)


# --- API Routes ---


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/stats")
async def get_stats():
    idx = _get_index()
    return {
        "version": idx.version,
        "created_at": idx.created_at.isoformat(),
        "updated_at": idx.updated_at.isoformat(),
        "sources": len(idx.sources),
        "total_nodes": idx.root.count_nodes(),
        "leaf_nodes": idx.root.count_leaves(),
        "max_depth": idx.root.depth,
    }


@app.get("/api/sources")
async def get_sources():
    idx = _get_index()
    return [
        {
            "id": s.id,
            "type": s.type.value,
            "origin": s.origin,
            "crawled_at": s.crawled_at.isoformat(),
            "page_count": s.page_count,
        }
        for s in idx.sources
    ]


def _node_to_dict(node: IndexNode, include_children: bool = True) -> dict:
    d = {
        "id": node.id,
        "title": node.title,
        "summary": node.summary,
        "ref": node.ref,
        "ref_type": node.ref_type.value if node.ref_type else None,
        "source_id": node.source_id,
        "is_leaf": node.is_leaf,
        "child_count": len(node.children),
        "leaf_count": node.count_leaves(),
    }
    if include_children:
        d["children"] = [_node_to_dict(c, include_children=False) for c in node.children]
    return d


@app.get("/api/tree")
async def get_tree():
    """Get the full tree structure (shallow — only titles and IDs, no deep nesting)."""
    idx = _get_index()

    def _tree_recursive(node: IndexNode) -> dict:
        return {
            "id": node.id,
            "title": node.title,
            "summary": node.summary,
            "ref": node.ref,
            "ref_type": node.ref_type.value if node.ref_type else None,
            "is_leaf": node.is_leaf,
            "children": [_tree_recursive(c) for c in node.children],
        }

    return _tree_recursive(idx.root)


@app.get("/api/node/{node_id:path}")
async def get_node(node_id: str):
    idx = _get_index()
    node = idx.root.find_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return _node_to_dict(node)


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1), max_results: int = Query(10, ge=1, le=100)):
    idx = _get_index()
    results = search_index(idx, q, max_results=max_results)
    return [
        {
            "node_id": r.node_id,
            "title": r.title,
            "summary": r.summary,
            "ref": r.ref,
            "score": r.score,
        }
        for r in results
    ]


@app.get("/api/fetch/{node_id:path}")
async def fetch_content(node_id: str):
    idx = _get_index()
    node = idx.root.find_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    if not node.ref:
        raise HTTPException(status_code=400, detail="Node has no reference")

    try:
        if node.ref_type and node.ref_type.value == "file":
            content = Path(node.ref).read_text(errors="replace")
        else:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(node.ref)
                resp.raise_for_status()
                content = resp.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch: {e}")

    # Convert HTML to markdown
    if "<html" in content[:500].lower() or "<!doctype" in content[:500].lower():
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify

            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = markdownify(str(soup), heading_style="ATX", strip=["img"])
        except ImportError:
            pass

    if len(content) > 12000:
        content = content[:12000] + "\n\n... [truncated]"

    return {"node_id": node_id, "title": node.title, "ref": node.ref, "content": content}


class AddSourceRequest(BaseModel):
    source: str


@app.post("/api/sources/add")
async def add_source(req: AddSourceRequest):
    global _index
    idx = _get_index()

    from sema_tree.composer import add_source as _add_source

    try:
        _index = await _add_source(idx, req.source)
        _save_index()
        return {"status": "ok", "message": f"Source '{req.source}' added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sources/{source_id}")
async def remove_source(source_id: str):
    global _index
    idx = _get_index()

    from sema_tree.composer import remove_source as _remove_source

    try:
        _index = await _remove_source(idx, source_id)
        _save_index()
        return {"status": "ok", "message": f"Source '{source_id}' removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sources/{source_id}/update")
async def update_source(source_id: str):
    global _index
    idx = _get_index()

    from sema_tree.composer import update_source as _update_source

    try:
        _index = await _update_source(idx, source_id)
        _save_index()
        return {"status": "ok", "message": f"Source '{source_id}' updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(index_path: str, host: str = "127.0.0.1", port: int = 8420) -> None:
    """Start the web UI server."""
    import socket

    load_index(index_path)

    # If the requested port is busy, find an available one
    for p in [port] + list(range(port + 1, port + 20)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, p)) != 0:
                if p != port:
                    logger.info("Port %d busy, using %d instead", port, p)
                uvicorn.run(app, host=host, port=p)
                return
    # Fallback: let uvicorn pick and fail with the original error
    uvicorn.run(app, host=host, port=port)
