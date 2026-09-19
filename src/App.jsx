import { Search, User, Settings, Play, Sparkles } from 'lucide-react';
import './App.css';

function App() {
  const apps = [
    { name: "TV", color: "#0055A4", glow: "rgba(0, 85, 164, 0.5)" },
    { name: "NETFLIX", color: "#E50914", glow: "rgba(229, 9, 20, 0.5)" },
    { name: "prime video", color: "#00A8E1", glow: "rgba(0, 168, 225, 0.5)" },
    { name: "Disney+", color: "#113CCF", glow: "rgba(17, 60, 207, 0.5)" },
    { name: "HBO", color: "#663399", glow: "rgba(102, 51, 153, 0.5)" },
    { name: "Apple TV", color: "#222222", glow: "rgba(255, 255, 255, 0.2)" },
    { name: "YouTube", color: "#FF0000", glow: "rgba(255, 0, 0, 0.5)" },
    { name: "Twitch", color: "#9146FF", glow: "rgba(145, 70, 255, 0.5)" },
  ];

  return (
    <div className="app-container">
      
      {/* Top Navbar */}
      <nav className="navbar">
        <div className="nav-left">
          <Search className="nav-icon" size={24} color="#f5f5f5" />
          <span className="nav-item active">Home</span>
          <span className="nav-item">Channels</span>
          <span className="nav-item">Apps</span>
        </div>
        <div className="nav-right">
          <User className="nav-icon" size={24} />
          <Settings className="nav-icon" size={24} />
          <span className="nav-item" style={{ cursor: 'default', color: '#a3a3a3' }}>10:15</span>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-bg"></div>
        <div className="hero-overlay-gradient"></div>
        
        <div className="hero-content">
          <div className="hero-tag">
            <Sparkles size={20} />
            KinoFiles AI
          </div>
          <h1 className="hero-title">Kino Files Agent</h1>
          <p className="hero-desc">
            Tu asistente inteligente para explorar el universo cinematográfico. 
            Descubre joyas ocultas, recibe recomendaciones personalizadas y encuentra 
            la película perfecta basándote en tus gustos y estado de ánimo actual.
          </p>
          <button className="hero-button">
            <Play size={20} fill="currentColor" />
            Lanzar agente
          </button>
        </div>
      </section>

      {/* Favourite Apps */}
      <section className="apps-section">
        <h2 className="apps-title">Favourite Apps</h2>
        <div className="apps-row">
          {apps.map((app, index) => (
            <div 
              key={index} 
              className="app-tile" 
              style={{ 
                backgroundColor: app.color, 
                '--tile-glow': app.glow 
              }}
            >
              {app.name}
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}

export default App;
