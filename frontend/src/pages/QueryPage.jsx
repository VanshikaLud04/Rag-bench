import { useState } from "react"
import axios from "axios"
import ChunkViewer from "../components/ChunkViewer"

export default function QueryPage() {
  const [query, setQuery] = useState("")
  const [model, setModel] = useState("phi3")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleQuery = async () => {
    if (!query.trim()) return
    setLoading(true)
    setResult(null)

    const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

    try {
      const res = await fetch(`${API_BASE_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, model })
      });

      if (!res.ok) {
        throw new Error("Query failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let answerText = "";
      let buffer = "";

      setResult({ answer: "", model: model });

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split("\\n\\n");
        buffer = lines.pop();
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.substring(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.type === "token") {
                answerText += data.content;
                setResult(prev => ({ ...prev, answer: answerText }));
              } else if (data.type === "metadata") {
                setResult(prev => ({ ...prev, ...data.content }));
              }
            } catch (e) {
              console.error("Error parsing stream chunk", e);
            }
          }
        }
      }
    } catch (err) {
      setResult({ error: err.message || "Query failed" });
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <h2>Query RAG</h2>
      <p className="subtitle">Ask a question against your uploaded documents</p>

      <div className="query-box">
        <textarea
          className="query-input"
          placeholder="Ask a question about your documents..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          rows={3}
        />
        <div className="query-controls">
          <select
            className="model-select"
            value={model}
            onChange={e => setModel(e.target.value)}
          >
            <option value="phi3">phi3</option>
            <option value="mistral">mistral</option>
            <option value="gemini-1.5-flash">gemini-1.5-flash</option>
          </select>
          <button
            className="btn-primary"
            onClick={handleQuery}
            disabled={!query.trim() || loading}
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>
      </div>

      {result?.error && (
        <div className="card error"><p>{result.error}</p></div>
      )}

      {result && !result.error && (
        <>
          <div className="card answer-card">
            <div className="answer-meta">
              <span className="model-tag">{result.model}</span>
              <span className="score-tag">
                retrieval score: {result.mean_retrieval_score}
              </span>
            </div>
            <p className="answer-text">{result.answer}</p>
          </div>
          <ChunkViewer chunks={result.retrieved_chunks} />
        </>
      )}
    </div>
  )
}