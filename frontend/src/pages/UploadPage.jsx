import { useState } from "react"
import axios from "axios"

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setStatus(null)

    const formData = new FormData()
    formData.append("file", file)


    const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

    try {
      const res = await axios.post(`${API_BASE_URL}/ingest/`, formData)
      setStatus({ type: "success", data: res.data })
    } catch (err) {
      setStatus({ type: "error", message: err.response?.data?.detail || "Upload failed" })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <h2>Upload Research Paper</h2>
      <p className="subtitle">Upload a PDF to ingest into the vector store</p>

      <div className="upload-box">
        <input
          type="file"
          accept=".pdf"
          onChange={e => setFile(e.target.files[0])}
          className="file-input"
        />
        {file && <p className="file-name">{file.name}</p>}
        <button
          className="btn-primary"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? "Ingesting..." : "Upload & Ingest"}
        </button>
      </div>

      {status?.type === "success" && (
        <div className="card success">
          <p>Ingested successfully</p>
          <p>Chunks created: <strong>{status.data.chunks_ingested}</strong></p>
          <p>Source: <strong>{status.data.source}</strong></p>
        </div>
      )}

      {status?.type === "error" && (
        <div className="card error">
          <p>{status.message}</p>
        </div>
      )}
    </div>
  )
}