# Embedding Service Web UI

Modern web interface for the Embedding Service API built with Next.js.

## Features

- 🏥 **Health Check** - Monitor service status and configuration
- 📝 **Single Query Embedding** - Convert text to vector embeddings
- 📚 **Batch Embeddings** - Embed multiple documents at once
- 💬 **Chat Interface** - Interactive chat with LLM
- ⚙️ **API Configuration** - Configurable API endpoint

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Production Build

```bash
npm run build
npm start
```

## Configuration

By default, the web UI connects to `http://localhost:8000`. You can change this through the Configuration section in the UI, or set the `NEXT_PUBLIC_API_URL` environment variable:

```bash
NEXT_PUBLIC_API_URL=http://api.example.com npm run dev
```

## Components

- **HealthCheck** - Check API health and retrieve service info
- **EmbedQuery** - Single text to embedding conversion
- **EmbedDocuments** - Batch text to embeddings conversion
- **ChatBox** - Interactive chat interface

## API Integration

The web UI connects to the embedding service API with these endpoints:

- `GET /health` - Service health check
- `POST /embed/query` - Single embedding
- `POST /embed/documents` - Batch embeddings
- `POST /chat` - Chat completion

## Environment Variables

- `NEXT_PUBLIC_API_URL` - API base URL (default: http://localhost:8000)

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
