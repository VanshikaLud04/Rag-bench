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
      const res = await axios.post(`${API_BASE_URL}/query/`, { query, model })
      setResult(res.data)
    } catch (err) {
      setResult({ error: err.response?.data?.detail || "Query failed" })
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