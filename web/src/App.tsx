import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { useEffect } from "react";

import { Celebration } from "./components/Celebration";
import { Console } from "./components/Console";
import { Sidebar } from "./components/Sidebar";
import { FilePage } from "./pages/FilePage";
import { FloorPage } from "./pages/FloorPage";
import { LootPage } from "./pages/LootPage";
import { MapPage } from "./pages/MapPage";
import { RoomPage } from "./pages/RoomPage";
import { StatusPage } from "./pages/StatusPage";
import { StoreProvider, useStore } from "./store";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    document.querySelector(".main")?.scrollTo({ top: 0 });
  }, [pathname]);
  return null;
}

function Shell() {
  const { state, error } = useStore();
  return (
    <div className="shell">
      <Sidebar />
      <main className="main">
        {error && <div className="topbar-error">{error}</div>}
        {!state && !error && <div className="loading">Lighting the torches…</div>}
        <Routes>
          <Route path="/" element={<MapPage />} />
          <Route path="/floors/:num" element={<FloorPage />} />
          <Route path="/floors/:num/rooms/:roomId" element={<RoomPage />} />
          <Route path="/loot" element={<LootPage />} />
          <Route path="/status" element={<StatusPage />} />
          <Route path="/files/*" element={<FilePage />} />
          <Route path="*" element={<MapPage />} />
        </Routes>
      </main>
      <Console />
      <Celebration />
      <ScrollToTop />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <StoreProvider>
        <Shell />
      </StoreProvider>
    </BrowserRouter>
  );
}
