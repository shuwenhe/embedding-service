'use client'

import { useState } from 'react'

interface EmbeddingsResponse {
  embeddings: number[][]
}

export default function EmbedDocuments({ apiUrl }: { apiUrl: string }) {
  const [texts, setTexts] = useState('')
  const [embeddings, setEmbeddings] = useState<EmbeddingsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleEmbed = async () => {
    const textList = texts
      .split('\n')
      .map((t) => t.trim())
      .filter((t) => t.length > 0)

    if (textList.length === 0) {
      setError('Please enter at least one text (one per line)')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${apiUrl}/embed/documents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: textList }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setEmbeddings(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setEmbeddings(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>📚 Embed Documents</h2>
      <div className="form-group">
        <label htmlFor="documents">Texts (one per line)</label>
        <textarea
          id="documents"
          value={texts}
          onChange={(e) => setTexts(e.target.value)}
          placeholder="Enter multiple texts, one per line..."
        />
      </div>
      <button onClick={handleEmbed} disabled={loading}>
        {loading ? '⏳ Embedding...' : '→ Embed All'}
      </button>

      {error && <div className="status error">❌ {error}</div>}

      {embeddings && (
        <div className="output">
          <div className="success">✓ {embeddings.embeddings.length} embeddings generated</div>
          <div style={{ marginTop: '10px' }}>
            <strong>Batch Size:</strong> {embeddings.embeddings.length}
          </div>
          <div style={{ marginTop: '10px' }}>
            <strong>Dimension:</strong> {embeddings.embeddings[0]?.length || 0}
          </div>
          <div style={{ marginTop: '15px' }}>
            <strong>Vectors:</strong>
            {embeddings.embeddings.map((vec, i) => (
              <div key={i} style={{ marginTop: '8px', fontSize: '0.85rem' }}>
                <div style={{ color: '#666' }}>Vector {i + 1}:</div>
                <div className="vector-preview">
                  {vec.slice(0, 5).map((val, j) => (
                    <div key={j} className="vector-item">
                      {val.toFixed(3)}
                    </div>
                  ))}
                  {vec.length > 5 && (
                    <div className="vector-item">...</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
