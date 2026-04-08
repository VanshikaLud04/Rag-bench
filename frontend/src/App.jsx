import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import QueryPage from "./pages/QueryPage";
import EvaluatePage from "./pages/EvaluatePage";
import Navbar from "./components/Navbar.jxc";
import "./App.css";

export default function App() {
  const [activePage, setActivePage] = useState("upload");

  return (
    <div className="app">
      <Navbar activePage={activePage} setActivePage={setActivePage} />

      <main className="main-content">
        {activePage === "upload" && <UploadPage />}
        {activePage === "query" && <QueryPage />}
        {activePage === "evaluate" && <EvaluatePage />}
      </main>
    </div>
  );
}