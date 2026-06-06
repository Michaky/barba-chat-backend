require('dotenv').config();
const fetch = require('node-fetch');

const HF_API_KEY      = process.env.HF_API_KEY;
const QDRANT_URL      = process.env.QDRANT_URL || 'https://a63b45c1-4f63-4df0-b0e9-99dc40f100c3.sa-east-1-0.aws.cloud.qdrant.io:6333';
const QDRANT_API_KEY  = process.env.QDRANT_API_KEY;
const HF_EMBED_MODEL  = 'sentence-transformers/all-MiniLM-L6-v2';
const COLLECTION_NAME = 'baseBarba';
const VECTOR_SIZE     = 384; // all-MiniLM-L6-v2 produce 384 dimensiones

if (!HF_API_KEY || !QDRANT_API_KEY) {
  console.error('❌ Falta HF_API_KEY o QDRANT_API_KEY en .env');
  process.exit(1);
}

const qdrantHeaders = {
  'Content-Type': 'application/json',
  'api-key': QDRANT_API_KEY
};

// ─── BASE DE CONOCIMIENTO ─────────────────────────────────
const knowledge = [
  { id: 'identidad-1', text: 'Barba Ahumada es una charcutería artesanal argentina. Producimos embutidos y cortes curados usando leñas frutales seleccionadas (manzano y espinillo) para lograr un ahumado natural y distinguido.' },
  { id: 'identidad-2', text: 'Trabajamos con pequeñas partidas de producción y curado prolongado a baja temperatura para lograr mayor terneza, menor merma en cocción y un perfil de sabor inconfundible.' },
  { id: 'box-premium', text: 'Box Picada Premium: experiencia gourmet para compartir. Incluye ahumados artesanales de Barba Ahumada, quesos curados, conservas premium y un gran Malbec argentino. Precio: $89.900.' },
  { id: 'box-amigos',  text: 'Box Picada Amigos: ideal para reuniones y picadas. Incluye ahumados artesanales, quesos seleccionados y snacks premium. Precio: $69.900.' },
  { id: 'pastron',    text: 'Pastrón Ahumado (250g): tapa de asado vacuno curada en especias y ahumada a baja temperatura con leño de manzano y espinillo. Precio: $9.500.' },
  { id: 'lomo-cerdo', text: 'Lomo de Cerdo Ahumado (200g): corte magro de lomo de cerdo, tiernizado con ahumado suave. Ideal para un sabor refinado y menos graso. Precio: $7.500.' },
  { id: 'bondiola',   text: 'Bondiola de Cerdo Ahumada (200g): el clásico indiscutido. Curada con paciencia y ahumada lentamente con leña de espinillo. Sabor profundo y textura jugosa. Precio: $7.200.' },
  { id: 'cracovia',   text: 'Salchichón tipo Cracovia (200g): embutido en tripa de colágeno. Picado grueso con granos de pimienta negra. Textura firme y sabor intenso. Precio: $8.500.' },
  { id: 'lomo-pollo', text: 'Lomo de Pollo Ahumado (200g): pechugas de pollo ahumadas en forma de lomo, curadas y condimentadas suavemente. Muy popular entre familias y niños. Precio: $9.000.' },
  { id: 'panceta',    text: 'Panceta Ahumada (200g): el infaltable para hamburguesas, varenikis y recetas favoritas. Ahumado artesanal con leña frutal. Precio: $8.000.' },
  { id: 'lenas',      text: 'Usamos leñas frutales: manzano y espinillo. El manzano aporta notas dulces y suaves. El espinillo da un ahumado más intenso y rústico, perfecto para cerdo y embutidos.' },
  { id: 'proceso',    text: 'El curado prolongado es la base de nuestra calidad. Cada pieza pasa por salmueras y especias seleccionadas antes del ahumado. Garantiza mismo gramaje y mismo nivel de humo en cada lote.' },
  { id: 'b2b-general',  text: 'Barba Ahumada trabaja con restaurantes, hamburgueserías premium, parrillas, hoteles boutique y catering. Ofrecemos cupos limitados por zona para mantener la exclusividad de cada cliente.' },
  { id: 'b2b-ventajas', text: 'Ventajas para clientes B2B: estandarización garantizada (mismo gramaje y nivel de humo), menor merma en cocción gracias al curado prolongado, y exclusividad por zona geográfica.' },
  { id: 'b2b-muestras', text: 'Para solicitar muestras técnicas para tu restaurante, completá el formulario "Para Chefs" en el sitio o contactanos por WhatsApp. Productos disponibles para cata: Pastrón, Lomo de Pollo, Cracovia, Panceta, Lomo de Cerdo y Bondiola.' },
  { id: 'pedidos',    text: 'Los pedidos se hacen a través del carrito en el sitio web y se confirman por WhatsApp al +54 9 342 549-6003. Allí se coordinan dirección de envío y medio de pago.' },
  { id: 'contacto',   text: 'Para consultas, pedidos mayoristas o catas técnicas, contactanos por WhatsApp al +54 9 342 549-6003. Respondemos en horario comercial.' },
  { id: 'resena-1',   text: 'Clientes destacan el pastrón: "Se nota el tiempo de curado y el ahumado con leña de manzano. Un viaje de ida." - Yamila A.' },
  { id: 'resena-2',   text: 'Chef profesional: "Lo sumamos a la carta de nuestro restaurante y es un éxito rotundo. Calidad constante en cada entrega." - Chef Romina C.' },
  { id: 'resena-3',   text: 'Cliente sobre el Box Picada: "El lomo es de otro nivel. El Box nos resolvió el evento del fin de semana con presentación de lujo." - Diego M.' }
];
// ──────────────────────────────────────────────────────────

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
    throw new Error(`HuggingFace error ${res.status}: ${err}`);
  }
  const data = await res.json();
  return Array.isArray(data[0]) ? data[0] : data;
}

async function createCollection() {
  const check = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`, { headers: qdrantHeaders });
  if (check.ok) {
    console.log(`⚠️  Colección "${COLLECTION_NAME}" ya existe. Eliminando para recrear...`);
    await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`, { method: 'DELETE', headers: qdrantHeaders });
  }
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}`, {
    method: 'PUT',
    headers: qdrantHeaders,
    body: JSON.stringify({ vectors: { size: VECTOR_SIZE, distance: 'Cosine' } })
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`No se pudo crear la colección: ${body}`);
  }
  console.log(`✅ Colección "${COLLECTION_NAME}" creada (${VECTOR_SIZE}d, Cosine)`);
}

async function upsertPoints(points) {
  const res = await fetch(`${QDRANT_URL}/collections/${COLLECTION_NAME}/points`, {
    method: 'PUT',
    headers: qdrantHeaders,
    body: JSON.stringify({ points })
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Qdrant upsert error: ${body}`);
  }
}

async function seed() {
  console.log('\n🌱 Cargando base de conocimiento en Qdrant Cloud...\n');
  await createCollection();
  const points = [];
  for (let i = 0; i < knowledge.length; i++) {
    const item = knowledge[i];
    process.stdout.write(`   [${i + 1}/${knowledge.length}] ${item.id}...`);
    const vector = await getEmbedding(item.text);
    points.push({ id: i + 1, vector, payload: { text: item.text, source_id: item.id } });
    console.log(' ✓');
  }
  await upsertPoints(points);
  console.log(`\n🔥 ¡Listo! ${points.length} chunks cargados en Qdrant Cloud.\n`);
}

seed().catch(err => {
  console.error('\n❌ Error:', err.message);
  process.exit(1);
});
