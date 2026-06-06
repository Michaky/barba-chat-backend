require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');

const app = express();
app.use(cors());
app.use(express.json());

// ─── CONFIG ───────────────────────────────────────────────
const GROQ_API_KEY    = process.env.GROQ_API_KEY;
const HF_API_KEY      = process.env.HF_API_KEY;
const QDRANT_URL      = process.env.QDRANT_URL || 'https://a63b45c1-4f63-4df0-b0e9-99dc40f100c3.sa-east-1-0.aws.cloud.qdrant.io:6333';
const QDRANT_API_KEY  = process.env.QDRANT_API_KEY;

const GROQ_MODEL      = 'llama-3.1-8b-instant';
const HF_EMBED_MODEL  = 'sentence-transformers/all-MiniLM-L6-v2';
const COLLECTION_NAME = 'baseBarba';
const TOP_K           = 4;
// ──────────────────────────────────────────────────────────

const required = { GROQ_API_KEY, HF_API_KEY, QDRANT_API_KEY };
for (const [key, val] of Object.entries(required)) {
  if (!val) { console.error(`❌ Falta ${key} en variables de entorno`); process.exit(1); }
}

const qdrantHeaders = {
  'Content-Type': 'application/json',
  'api-key': QDRANT_API_KEY
};

async function getEmbedding(text) {
  const res = await fetch(
    `https://router.huggingface.co/hf-inference/models/${HF_EMBED_MODEL}/pipeline/feature-extraction`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ inputs: text, normalize: true })
    }
  );
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HuggingFace embedding error ${res.status}: ${err}`);
  }
  const data = await res.json();
  return Array.isArray(data[0]) ? data[0] : data;
}

async function searchQdrant(vector) {
  const res = await fetch(
    `${QDRANT_URL}/collections/${COLLECTION_NAME}/points/search`,
    {
      method: 'POST',
      headers: qdrantHeaders,
      body: JSON.stringify({ vector, limit: TOP_K, with_payload: true })
    }
  );
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Qdrant search error ${res.status}: ${err}`);
  }
  const data = await res.json();
  return data.result || [];
}

async function chatWithGroq(question, context, res) {
  const systemPrompt = `Sos el asistente virtual de Barba Ahumada, una charcutería artesanal argentina especializada en ahumados con leñas frutales.
Tu tono es cálido, apasionado por la gastronomía y profesional.
Respondé SIEMPRE en español rioplatense (vos, ustedes).
Usá únicamente la información del contexto para responder. Si no encontrás la respuesta, decí honestamente que no tenés esa información y sugerí contactar por WhatsApp al +54 9 342 549-6003.
No inventes precios, ingredientes ni disponibilidad.
Sé conciso: máximo 3 párrafos cortos.

CONTEXTO:
${context}`;

  const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${GROQ_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question }
      ],
      stream: true,
      max_tokens: 512,
      temperature: 0.7
    })
  });

  if (!groqRes.ok) {
    const err = await groqRes.text();
    throw new Error(`Groq error ${groqRes.status}: ${err}`);
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  let buffer = '';
  for await (const chunk of groqRes.body) {
    buffer += chunk.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') {
        res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
        res.end();
        return;
      }
      try {
        const json = JSON.parse(data);
        const token = json.choices?.[0]?.delta?.content;
        if (token) res.write(`data: ${JSON.stringify({ token })}\n\n`);
      } catch (_) {}
    }
  }
  res.end();
}

app.post('/chat', async (req, res) => {
  const { question } = req.body;
  if (!question?.trim()) return res.status(400).json({ error: 'Pregunta vacía.' });
  try {
    const vector  = await getEmbedding(question);
    const hits    = await searchQdrant(vector);
    const context = hits.map(h => h.payload?.text || '').filter(Boolean).join('\n\n---\n\n');
    await chatWithGroq(question, context || 'Sin contexto disponible.', res);
  } catch (err) {
    console.error('Error en /chat:', err.message);
    if (!res.headersSent) res.status(500).json({ error: 'Error interno del servidor.' });
  }
});

app.get('/health', (_, res) => res.json({
  status: 'ok',
  model: GROQ_MODEL,
  embed: HF_EMBED_MODEL,
  collection: COLLECTION_NAME
}));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`\n🔥 Barba Ahumada Chat Backend — puerto ${PORT}`);
  console.log(`   Chat:       Groq (${GROQ_MODEL})`);
  console.log(`   Embeddings: HuggingFace (${HF_EMBED_MODEL})`);
  console.log(`   Qdrant:     ${QDRANT_URL}\n`);
});
