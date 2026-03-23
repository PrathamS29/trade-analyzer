import axios from 'axios'

const API_URL = 'http://localhost:8000/api/predictions'

const predApi = axios.create({
  baseURL: API_URL,
  headers: { 'Content-Type': 'application/json' },
})

export const predictionService = {
  async searchPlayers(query, position = null, team = null) {
    const params = new URLSearchParams()
    if (query) params.append('q', query)
    if (position) params.append('position', position)
    if (team) params.append('team', team)
    const response = await predApi.get(`/players?${params}`)
    return response.data
  },

  async getPositions() {
    const response = await predApi.get('/players/positions')
    return response.data
  },

  async getTeams() {
    const response = await predApi.get('/players/teams')
    return response.data
  },

  async predictPlayer(playerName) {
    const response = await predApi.get(`/predict/${encodeURIComponent(playerName)}`)
    return response.data
  },

  async analyzeTrade(sideANames, sideBNames) {
    const response = await predApi.post('/trade', {
      side_a: sideANames,
      side_b: sideBNames,
    })
    return response.data
  },
}
