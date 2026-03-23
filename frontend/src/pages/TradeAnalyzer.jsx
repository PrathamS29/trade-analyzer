import React, { useState, useEffect, useCallback } from 'react'
import { predictionService } from '../services/predictionService'
import './TradeAnalyzer.css'

const TradeAnalyzer = () => {
  const [sideA, setSideA] = useState([])
  const [sideB, setSideB] = useState([])
  const [activeTab, setActiveTab] = useState('A')
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  // Search state
  const [query, setQuery] = useState('')
  const [position, setPosition] = useState('')
  const [team, setTeam] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [positions, setPositions] = useState([])
  const [teams, setTeams] = useState([])

  // Load filter options
  useEffect(() => {
    const loadFilters = async () => {
      try {
        const [posRes, teamRes] = await Promise.all([
          predictionService.getPositions(),
          predictionService.getTeams(),
        ])
        setPositions(posRes.positions || [])
        setTeams(teamRes.teams || [])
      } catch (err) {
        console.error('Failed to load filters:', err)
      }
    }
    loadFilters()
  }, [])

  // Search players
  useEffect(() => {
    if (query.length < 2) {
      setSearchResults([])
      return
    }
    const timeout = setTimeout(async () => {
      setSearching(true)
      try {
        const results = await predictionService.searchPlayers(query, position || null, team || null)
        setSearchResults(results)
      } catch (err) {
        console.error('Search failed:', err)
        setSearchResults([])
      } finally {
        setSearching(false)
      }
    }, 300)
    return () => clearTimeout(timeout)
  }, [query, position, team])

  // Run trade analysis whenever sides change
  const runAnalysis = useCallback(async () => {
    if (sideA.length === 0 || sideB.length === 0) {
      setAnalysis(null)
      return
    }
    setAnalyzing(true)
    try {
      const result = await predictionService.analyzeTrade(
        sideA.map(p => p.player_name),
        sideB.map(p => p.player_name),
      )
      setAnalysis(result)
    } catch (err) {
      console.error('Analysis failed:', err)
    } finally {
      setAnalyzing(false)
    }
  }, [sideA, sideB])

  useEffect(() => {
    runAnalysis()
  }, [runAnalysis])

  const addPlayer = async (player) => {
    // Fetch prediction for this player
    let enriched = { ...player }
    try {
      const pred = await predictionService.predictPlayer(player.player_name)
      enriched = { ...enriched, ...pred }
    } catch {
      enriched.predicted_fp = player.season_avg_fp
      enriched.ceiling = player.season_avg_fp
      enriched.floor = player.season_avg_fp
    }

    if (activeTab === 'A') {
      setSideA(prev => [...prev, enriched])
    } else {
      setSideB(prev => [...prev, enriched])
    }
    setQuery('')
    setSearchResults([])
  }

  const removePlayer = (side, playerName) => {
    if (side === 'A') {
      setSideA(prev => prev.filter(p => p.player_name !== playerName))
    } else {
      setSideB(prev => prev.filter(p => p.player_name !== playerName))
    }
  }

  const getRecommendationStyle = () => {
    if (!analysis) return { color: '#888' }
    if (analysis.recommendation.startsWith('ACCEPT')) return { color: '#28a745' }
    if (analysis.recommendation.startsWith('REJECT')) return { color: '#dc3545' }
    return { color: '#ffc107' }
  }

  return (
    <div className="trade-analyzer">
      <div className="analyzer-header">
        <h1>Fantasy Basketball <span>Trade Analyzer</span></h1>
        <p>Powered by CNN predictions trained on real 2024-25 NBA data</p>
      </div>

      <div className="analyzer-content">
        <div className="trade-sides">
          {/* Side A */}
          <div className="trade-side">
            <div className="side-header">
              <h2>Side A</h2>
              <button className="clear-side-btn" onClick={() => setSideA([])} disabled={sideA.length === 0}>
                Clear
              </button>
            </div>
            <div className="side-players">
              {sideA.length === 0 ? (
                <div className="empty-state"><p>No players added</p></div>
              ) : (
                sideA.map(p => (
                  <div key={p.player_name} className="player-card">
                    <div className="player-info">
                      <div className="player-name">{p.player_name}</div>
                      <div className="player-details">
                        <span className="player-position">{p.position}</span>
                        <span className="player-team">{p.team}</span>
                      </div>
                      <div className="player-predictions">
                        <div className="pred-row">
                          <span className="pred-label">Predicted:</span>
                          <span className="pred-value">{p.predicted_fp?.toFixed(1)} FP</span>
                        </div>
                        <div className="pred-row">
                          <span className="pred-label">Ceiling:</span>
                          <span className="pred-value ceiling">{p.ceiling?.toFixed(1)}</span>
                        </div>
                        <div className="pred-row">
                          <span className="pred-label">Floor:</span>
                          <span className="pred-value floor">{p.floor?.toFixed(1)}</span>
                        </div>
                        <div className="pred-row">
                          <span className="pred-label">Season Avg:</span>
                          <span className="pred-value">{p.season_avg_fp?.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                    <button className="remove-button" onClick={() => removePlayer('A', p.player_name)}>
                      &times;
                    </button>
                  </div>
                ))
              )}
            </div>
            {sideA.length > 0 && (
              <div className="side-total">
                <span>Total Predicted: </span>
                <strong>{sideA.reduce((s, p) => s + (p.predicted_fp || 0), 0).toFixed(1)} FP</strong>
              </div>
            )}
          </div>

          {/* VS / Analysis */}
          <div className="trade-vs">
            <div className="vs-divider"><span className="vs-text">VS</span></div>
            {analysis && (
              <div className="trade-analysis">
                <div className="analysis-result" style={getRecommendationStyle()}>
                  <div className="analysis-status">{analysis.recommendation}</div>
                  <div className="analysis-details">
                    <div>Predicted diff: {analysis.difference > 0 ? '+' : ''}{analysis.difference.toFixed(1)} FP</div>
                    <div>Ceiling diff: {analysis.ceiling_diff > 0 ? '+' : ''}{analysis.ceiling_diff.toFixed(1)}</div>
                    <div>Floor diff: {analysis.floor_diff > 0 ? '+' : ''}{analysis.floor_diff.toFixed(1)}</div>
                  </div>
                </div>
              </div>
            )}
            {analyzing && <div className="analyzing">Analyzing...</div>}
          </div>

          {/* Side B */}
          <div className="trade-side">
            <div className="side-header">
              <h2>Side B</h2>
              <button className="clear-side-btn" onClick={() => setSideB([])} disabled={sideB.length === 0}>
                Clear
              </button>
            </div>
            <div className="side-players">
              {sideB.length === 0 ? (
                <div className="empty-state"><p>No players added</p></div>
              ) : (
                sideB.map(p => (
                  <div key={p.player_name} className="player-card">
                    <div className="player-info">
                      <div className="player-name">{p.player_name}</div>
                      <div className="player-details">
                        <span className="player-position">{p.position}</span>
                        <span className="player-team">{p.team}</span>
                      </div>
                      <div className="player-predictions">
                        <div className="pred-row">
                          <span className="pred-label">Predicted:</span>
                          <span className="pred-value">{p.predicted_fp?.toFixed(1)} FP</span>
                        </div>
                        <div className="pred-row">
                          <span className="pred-label">Ceiling:</span>
                          <span className="pred-value ceiling">{p.ceiling?.toFixed(1)}</span>
                        </div>
                        <div className="pred-row">
                          <span className="pred-label">Floor:</span>
                          <span className="pred-value floor">{p.floor?.toFixed(1)}</span>
                        </div>
                        <div className="pred-row">
                          <span className="pred-label">Season Avg:</span>
                          <span className="pred-value">{p.season_avg_fp?.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                    <button className="remove-button" onClick={() => removePlayer('B', p.player_name)}>
                      &times;
                    </button>
                  </div>
                ))
              )}
            </div>
            {sideB.length > 0 && (
              <div className="side-total">
                <span>Total Predicted: </span>
                <strong>{sideB.reduce((s, p) => s + (p.predicted_fp || 0), 0).toFixed(1)} FP</strong>
              </div>
            )}
          </div>
        </div>

        {/* Player Search */}
        <div className="player-search-section">
          <div className="search-tabs">
            <button className={`search-tab ${activeTab === 'A' ? 'active' : ''}`} onClick={() => setActiveTab('A')}>
              Add to Side A
            </button>
            <button className={`search-tab ${activeTab === 'B' ? 'active' : ''}`} onClick={() => setActiveTab('B')}>
              Add to Side B
            </button>
          </div>

          <div className="search-form">
            <input
              type="text"
              placeholder="Search by player name..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="search-input"
            />
            <div className="filter-group">
              <select value={position} onChange={(e) => setPosition(e.target.value)} className="filter-select">
                <option value="">All Positions</option>
                {positions.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
              <select value={team} onChange={(e) => setTeam(e.target.value)} className="filter-select">
                <option value="">All Teams</option>
                {teams.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>

          {searching && <div className="loading-indicator">Searching...</div>}

          {searchResults.length > 0 && (
            <div className="search-results">
              <div className="results-header">Found {searchResults.length} players</div>
              <div className="results-list">
                {searchResults.map(p => (
                  <div key={p.player_name} className="search-result-item" onClick={() => addPlayer(p)}>
                    <div className="result-player-name">{p.player_name}</div>
                    <div className="result-player-details">
                      <span className="result-position">{p.position}</span>
                      <span className="result-team">{p.team}</span>
                      <span className="result-points">{p.season_avg_fp.toFixed(1)} avg FP</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {query.length >= 2 && searchResults.length === 0 && !searching && (
            <div className="no-results"><p>No players found.</p></div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TradeAnalyzer
