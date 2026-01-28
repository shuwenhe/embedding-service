'use client'

import { useState, useEffect } from 'react'

interface HealthResponse {
  status: string
  provider: string
  embed_model: string
  llm_model: string
}

export default function HealthCheck({ apiUrl }: { apiUrl: string }) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const checkHealth = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${apiUrl}/health`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setHealth(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setHealth(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
  }, [apiUrl])

  return (
    <div className="card">
      <h2>🏥 Health Check</h2>
      <button onClick={checkHealth} disabled={loading}>
        {loading ? '⏳ Checking...' : '✓ Check Status'}
      </button>

      {error && (
        <div className="status error">
          ❌ {error}
        </div>
      )}

      {health && (
        <div className="status ok">
          ✓ Service is healthy
        </div>
      )}

      {health && (
        <div className="output">
          <div>
            <span className="badge provider">Provider: {health.provider}</span>
            <span className="badge model">Embed: {health.embed_model}</span>
          </div>
          <div style={{ marginTop: '10px' }}>
            <span className="badge model">LLM: {health.llm_model}</span>
          </div>
          <pre style={{ marginTop: '10px', fontSize: '0.85rem' }}>
            {JSON.stringify(health, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
