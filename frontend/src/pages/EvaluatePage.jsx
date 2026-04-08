import { useState } from "react"
import axios from "axios"
import ComparisonTable from "../components/ComparisonTable"

const ALL_MODELS = ["phi3", "mistral", "gemini-1.5-flash"]

export default function EvaluatePage() {
  const [query, setQuery] = useState("")
  const [groundTruth, setGroundTruth] = useState("")
  const [selectedModels, setSelectedModels] = useState(["phi3", "mistral"])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const toggleModel = (model) => {
    setSelectedModels(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    )
  }

  const handleEvaluate = async () => {
    if (!query.trim()) return
    setLoading(true)
    setResult(null)


    const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

    try {
      const res = await axios.post(`${API_BASE_URL}/evaluate/`, {
        query,
        ground_truth: groundTruth || null,
        models: selectedModels,
      })
      setResult(res.data)
    } catch (err) {
      setResult({ error: err.response?.data?.detail || "Evaluation failed" })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <h2>Evaluate</h2>
      <p className="subtitle">
        Run the same query across multiple models and compare RAG metrics
      </p>

      <div className="query-box">
        <textarea
          className="query-input"
          placeholder="Enter your evaluation query..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          rows={3}
        />
        <input
          className="query-input"
          placeholder="Ground truth answer (optional — enables context recall)"
          value={groundTruth}
          onChange={e => setGroundTruth(e.target.value)}
        />

        <div className="model-toggles">
          {ALL_MODELS.map(m => (
            <button
              key={m}
              className={`toggle-btn ${selectedModels.includes(m) ? "active" : ""}`}
              onClick={() => toggleModel(m)}
            >
              {m}
            </button>
          ))}
        </div>

        <button
          className="btn-primary"
          onClick={handleEvaluate}
          disabled={!query.trim() || selectedModels.length === 0 || loading}
        >
          {loading ? "Evaluating..." : "Run Evaluation"}
        </button>
      </div>

      {result?.error && (
        <div className="card error"><p>{result.error}</p></div>
      )}

      {result && !result.error && (
        <ComparisonTable
          results={result.results}
          chunksCount={result.retrieved_chunks_count}
        />
      )}
    </div>
  )
}