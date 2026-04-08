import { useState } from "react"

export default function ChunkViewer({ chunks }) {
  const [open, setOpen] = useState(false)

  if (!chunks || chunks.length === 0) return null

  return (
    <div className="chunk-viewer">
      <button className="toggle-chunks" onClick={() => setOpen(!open)}>
        {open ? "Hide" : "Show"} retrieved chunks ({chunks.length})
      </button>

      {open && (
        <div className="chunks-list">
          {chunks.map((chunk, i) => (
            <div key={i} className="chunk-card">
              <div className="chunk-meta">
                <span className="chunk-source">{chunk.source}</span>
                <span className="chunk-score">score: {chunk.score}</span>
              </div>
              <p className="chunk-text">{chunk.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}