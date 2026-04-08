export default function ComparisonTable({ results, chunksCount }) {
  if (!results || results.length === 0) return null

  const metrics = [
    { key: "context_precision", label: "Context Precision" },
    { key: "context_recall", label: "Context Recall" },
    { key: "faithfulness", label: "Faithfulness" },
    { key: "answer_relevancy", label: "Answer Relevancy" },
    { key: "mean_retrieval_score", label: "Retrieval Score" },
  ]

  const best = (key) => {
    return Math.max(...results.map(r => r[key]))
  }

  return (
    <div className="comparison">
      <p className="chunks-used">Chunks retrieved: <strong>{chunksCount}</strong></p>

      <div className="answers-grid">
        {results.map((r, i) => (
          <div key={i} className="answer-card">
            <span className="model-tag">{r.model}</span>
            <p className="answer-text">{r.answer}</p>
          </div>
        ))}
      </div>

      <table className="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            {results.map((r, i) => (
              <th key={i}>{r.model}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map(m => (
            <tr key={m.key}>
              <td>{m.label}</td>
              {results.map((r, i) => (
                <td
                  key={i}
                  className={r[m.key] === best(m.key) ? "best-score" : ""}
                >
                  {r[m.key].toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}