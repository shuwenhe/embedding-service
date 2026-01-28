'use client'

import { useState, useEffect } from 'react'
import HealthCheck from '@/components/HealthCheck'
import EmbedQuery from '@/components/EmbedQuery'
import EmbedDocuments from '@/components/EmbedDocuments'
import ChatBox from '@/components/ChatBox'

export default function Home() {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')

  useEffect(() => {
    const savedUrl = localStorage.getItem('apiUrl')
    if (savedUrl) {
      setApiUrl(savedUrl)
    }
  }, [])

  const handleApiUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newUrl = e.target.value
    setApiUrl(newUrl)
    localStorage.setItem('apiUrl', newUrl)
  }

  return (
    <div className="container">
      <div className="header">
        <h1>🚀 Embedding Service</h1>
        <p>Unified API for text embeddings and LLM chat</p>
      </div>

      <div className="card" style={{ marginBottom: '30px' }}>
        <h2>⚙️ Configuration</h2>
        <div className="form-group">
          <label htmlFor="apiUrl">API Base URL</label>
          <input
            id="apiUrl"
            type="text"
            value={apiUrl}
            onChange={handleApiUrlChange}
            placeholder="http://localhost:8000"
          />
          <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '8px' }}>
            Change this to match your embedding service API endpoint
          </p>
        </div>
      </div>

      <div className="grid">
        <HealthCheck apiUrl={apiUrl} />
        <EmbedQuery apiUrl={apiUrl} />
        <EmbedDocuments apiUrl={apiUrl} />
        <ChatBox apiUrl={apiUrl} />
      </div>
    </div>
  )
}
