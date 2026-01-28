'use client'

import { useState } from 'react'

interface EmbeddingResponse {
  embedding: number[]
}

export default function EmbedQuery({ apiUrl }: { apiUrl: string }) {
  const [text, setText] = useState('')
  const [embedding, setEmbedding] = useState<EmbeddingResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleEmbed = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${apiUrl}/embed/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setEmbedding(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setEmbedding(null)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleEmbed()
    }
  }

  return (
    <div className="card">
      <h2>📝 Embed Query</h2>
      <div className="form-group">
        <label htmlFor="query">Text to embed</label>
        <textarea
          id="query"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter text to convert to embedding..."
        />
      </div>
      <button onClick={handleEmbed} disabled={loading}>
        {loading ? '⏳ Embedding...' : '→ Embed'}
      </button>

      {error && <div className="status error">❌ {error}</div>}

      {embedding && (
        <div className="output">
          <div className="success">✓ Embedding generated successfully</div>
          <div style={{ marginTop: '10px' }}>
            <strong>Dimension:</strong> {embedding.embedding.length}
          </div>
          <div className="vector-preview">
            {embedding.embedding.slice(0, 10).map((val, i) => (
              <div key={i} className="vector-item">
                {val.toFixed(3)}
              </div>
            ))}
            {embedding.embedding.length > 10 && (
              <div className="vector-item">...</div>
            )}
          </div>
          <pre style={{ marginTop: '10px', fontSize: '0.75rem', maxHeight: '150px', overflow: 'auto' }}>
            {JSON.stringify(embedding, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
