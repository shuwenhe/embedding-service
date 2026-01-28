'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatResponse {
  response: string
}

export default function ChatBox({ apiUrl }: { apiUrl: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data: ChatResponse = await response.json()
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="card">
      <h2>💬 Chat</h2>
      <div
        style={{
          background: '#f9f9f9',
          borderRadius: '8px',
          padding: '12px',
          height: '300px',
          overflowY: 'auto',
          marginBottom: '12px',
          border: '1px solid #e0e0e0',
        }}
      >
        {messages.length === 0 && (
          <div className="empty">Start a conversation...</div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              marginBottom: '12px',
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            <div
              style={{
                maxWidth: '85%',
                padding: '10px 12px',
                borderRadius: '8px',
                background:
                  msg.role === 'user' ? '#667eea' : '#e8e8e8',
                color: msg.role === 'user' ? 'white' : '#333',
                wordBreak: 'break-word',
              }}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
            <div
              style={{
                padding: '10px 12px',
                borderRadius: '8px',
                background: '#e8e8e8',
                color: '#667eea',
              }}
            >
              ⏳ Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {error && <div className="status error">❌ {error}</div>}

      <div className="form-group" style={{ marginBottom: '10px' }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message here... (Shift+Enter for new line)"
          disabled={loading}
          style={{ minHeight: '60px' }}
        />
      </div>
      <button onClick={handleSend} disabled={loading || !input.trim()}>
        {loading ? '⏳ Sending...' : '→ Send'}
      </button>
    </div>
  )
}
