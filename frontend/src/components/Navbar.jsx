export default function Navbar({ activePage, setActivePage }) {
  const tabs = [
    { id: "upload", label: "Upload Docs" },
    { id: "query", label: "Query" },
    { id: "evaluate", label: "Evaluate" },
  ]

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="brand-rag">Rag</span>
        <span className="brand-bench">Bench</span>
      </div>
      <div className="navbar-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activePage === tab.id ? "active" : ""}`}
            onClick={() => setActivePage(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </nav>
  )
}