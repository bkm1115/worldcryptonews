import type { NextApiRequest, NextApiResponse } from 'next';
import RSSParser from 'rss-parser';

/**
 * Optional env: INCLUDE_SEC=1 to include SEC RSS.
 * Optional env: CRYPTOPANIC_TOKEN=... to include CryptoPanic JSON.
 * Optional env: ENABLE_TRANSFORMERS=1 to enable Hugging Face sentiment (FinBERT + XLM-R).
 * Optional env: FINBERT_MODEL_ID / XLMR_MODEL_ID to override default transformer models.
 */

type Sentiment = 'positive' | 'negative' | 'neutral';

interface SignalItem {
  title: string;
  link: string;
  pubDate: string;
  source: string;
  sentiment: Sentiment;
  score: number;
  weight: number;
}

interface SignalResponse {
  windowMinutes: number;
  timezoneOffsetMinutes: number;
  generatedAt: string;
  counts: { positive: number; negative: number; neutral: number };
  weights: { posW: number; negW: number; neuW: number };
  longPct: number;
  shortPct: number;
  recommendation: 'LONG' | 'SHORT' | 'NEUTRAL';
  items: SignalItem[];
}

const GOOGLE_NEWS_BASE = 'https://news.google.com/rss/search?';
const GOOGLE_PARAMS = '&hl=en-US&gl=US&ceid=US:en';

const SEARCH_QUERIES = [
  'cryptocurrency OR crypto OR bitcoin OR ethereum OR altcoin',
  '비트코인 OR 이더리움 OR 코인 OR 암호화폐',
];

// Native crypto RSS feeds (no API keys)
const CRYPTO_FEEDS: string[] = [
  'https://www.coindesk.com/arc/outboundfeeds/rss/',
  'https://cointelegraph.com/rss',
  'https://news.bitcoin.com/feed',
  'https://blockworks.co/feed',
  'https://crypto.news/feed/',
  'https://cryptoslate.com/feed/',
  'https://bitcoinmagazine.com/.rss/full/',
  'https://www.theblock.co/rss.xml',
  'https://decrypt.co/feed',
];

// Optional sources via env flags
const INCLUDE_SEC = process.env.INCLUDE_SEC === '1'; // include SEC press releases RSS
const CRYPTOPANIC_TOKEN = process.env.CRYPTOPANIC_TOKEN; // CryptoPanic API token (if missing, skip)
const ENABLE_TRANSFORMERS = process.env.ENABLE_TRANSFORMERS === '1';
const TIMEZONE_OFFSET_MINUTES = Number(
  process.env.TIMEZONE_OFFSET_MINUTES ??
    process.env.TZ_OFFSET_MINUTES ??
    0
);
const FINBERT_MODEL_CANDIDATES = [
  process.env.FINBERT_MODEL_ID,
  'ProsusAI/finbert',
  'Xenova/finbert',
  'Xenova/finbert-tone',
  'Xenova/bert-base-uncased-finetuned-sst-2-english',
].filter(Boolean) as string[];
const XLMR_MODEL_CANDIDATES = [
  process.env.XLMR_MODEL_ID,
  'cardiffnlp/twitter-xlm-roberta-base-sentiment',
  'Xenova/twitter-xlm-roberta-base-sentiment',
  'Xenova/bert-base-multilingual-uncased-sentiment',
].filter(Boolean) as string[];

const POSITIVE_WORDS = new Set([
  'gain',
  'gains',
  'rally',
  'bull',
  'bullish',
  'surge',
  'surges',
  'breakout',
  'record',
  'all-time',
  'positive',
  'upgrade',
  'outperform',
  'buy',
  'accumulate',
  'support',
  'uptrend',
  'rebound',
  'recover',
  'green',
  'approve',
  'approval',
  'approved',
  'adopt',
  'adoption',
  'partnership',
  'investment',
  'invest',
  'funding',
  'etf',
  'listing',
]);

const NEGATIVE_WORDS = new Set([
  'drop',
  'drops',
  'fall',
  'falls',
  'bear',
  'bearish',
  'plunge',
  'crash',
  'selloff',
  'sell-off',
  'decline',
  'downtrend',
  'negative',
  'downgrade',
  'underperform',
  'sell',
  'resistance',
  'reject',
  'rejection',
  'ban',
  'lawsuit',
  'probe',
  'fraud',
  'hack',
  'exploit',
  'scam',
  'fud',
  'outflow',
  'liquidation',
  'liquidations',
  'insolvency',
  'bankrupt',
  'delist',
  'concern',
  'steal',
  'steals',
  'stole',
  'stolen',
  'theft',
]);

const SOURCE_WEIGHTS: Record<string, number> = {
  Reuters: 1.3,
  Bloomberg: 1.25,
  'The Wall Street Journal': 1.2,
  WSJ: 1.2,
  'Financial Times': 1.2,
  CoinDesk: 1.15,
  Cointelegraph: 1.1,
  'The Block': 1.1,
};

const parser = new RSSParser({
  headers: {
    'User-Agent': 'Mozilla/5.0 (compatible; sentiment-signal/1.0)',
  },
});

const SENTIMENT_CACHE_TTL_MS = 1000 * 60 * 60 * 6; // 6 hours

type RSSItem = {
  title?: string;
  link?: string;
  pubDate?: string;
  isoDate?: string;
  contentSnippet?: string;
  content?: string;
  creator?: string;
  source?: { title?: string } | string;
};

type SentimentScore = { label: Sentiment; score: number };

type TextClassificationPipeline = (
  input: string,
  options?: Record<string, unknown>
) => Promise<Array<{ label: string; score: number }>>;

let transformersModulePromise: Promise<any> | null = null;
let finbertPipelinePromise:
  | Promise<TextClassificationPipeline | null>
  | null = null;
let xlmRobertaPipelinePromise:
  | Promise<TextClassificationPipeline | null>
  | null = null;
const sentimentCache = new Map<
  string,
  { score: SentimentScore; expiresAt: number }
>();
const RESPONSE_CACHE_TTL_MS = Math.max(
  0,
  Number(process.env.SIGNAL_CACHE_TTL_MS ?? 45000)
);
let responseCache: { timestamp: number; payload: SignalResponse } | null = null;

const clamp = (min: number, max: number, value: number) =>
  Math.min(max, Math.max(min, value));

const toMinutesAgo = (date: Date, now: Date) =>
  (now.getTime() - date.getTime()) / (1000 * 60);

const tokenize = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean);

const scoreSentimentLexicon = (
  title: string,
  snippet: string
): SentimentScore => {
  const tokens = tokenize(`${title} ${snippet}`);
  let pos = 0;
  let neg = 0;

  tokens.forEach((token) => {
    if (POSITIVE_WORDS.has(token)) {
      pos += 1;
    }
    if (NEGATIVE_WORDS.has(token)) {
      neg += 1;
    }
  });

  const rawScore = pos - neg;
  if (rawScore > 0) {
    return { label: 'positive', score: rawScore };
  }
  if (rawScore < 0) {
    return { label: 'negative', score: rawScore };
  }
  return { label: 'neutral', score: 0 };
};

const weightForSource = (source: string) => {
  const trimmed = source.trim();
  return SOURCE_WEIGHTS[trimmed] ?? 1;
};

const linearTimeDecay = (minutesAgo: number, windowMinutes: number) => {
  const safeWindow = Math.max(windowMinutes, 1);
  const clamped = clamp(0, safeWindow, minutesAgo);
  const decay = 1 - (clamped / safeWindow) * 0.3;
  return clamp(0.7, 1, decay);
};

const toDate = (value?: string | Date | null): Date | null => {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date;
};

const withinWindow = (
  pubDate: string | Date | null | undefined,
  now: Date,
  windowMinutes: number
) => {
  const date = toDate(pubDate);
  if (!date) {
    return false;
  }
  const minutesAgo = toMinutesAgo(date, now);
  return minutesAgo >= 0 && minutesAgo <= windowMinutes;
};

const decayWeight = (date: Date, now: Date, windowMinutes: number) => {
  const minutesAgo = toMinutesAgo(date, now);
  if (minutesAgo < 0 || minutesAgo > windowMinutes) {
    return 0;
  }
  return linearTimeDecay(minutesAgo, windowMinutes);
};

const hasHangul = (text: string) =>
  /[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]/.test(text);

const mapModelLabel = (label: string): Sentiment => {
  const lowered = label.toLowerCase();
  if (lowered.includes('pos')) {
    return 'positive';
  }
  if (lowered.includes('neg')) {
    return 'negative';
  }
  if (lowered.includes('neu')) {
    return 'neutral';
  }
  if (lowered === 'label_2') {
    return 'positive';
  }
  if (lowered === 'label_0') {
    return 'negative';
  }
  if (lowered === 'label_1') {
    return 'neutral';
  }
  return 'neutral';
};

const labelFromScore = (score: number): Sentiment => {
  if (score > 0) {
    return 'positive';
  }
  if (score < 0) {
    return 'negative';
  }
  return 'neutral';
};

const getCachedSentiment = (
  key: string,
  now: number
): SentimentScore | null => {
  const cached = sentimentCache.get(key);
  if (!cached) {
    return null;
  }
  if (cached.expiresAt <= now) {
    sentimentCache.delete(key);
    return null;
  }
  return cached.score;
};

const setCachedSentiment = (
  key: string,
  score: SentimentScore,
  now: number
) => {
  sentimentCache.set(key, {
    score,
    expiresAt: now + SENTIMENT_CACHE_TTL_MS,
  });
};

const cleanupSentimentCache = (now: number) => {
  for (const [key, cached] of sentimentCache.entries()) {
    if (cached.expiresAt <= now) {
      sentimentCache.delete(key);
    }
  }
};

const computeTimezoneStartOfDay = (date: Date, offsetMinutes: number): Date => {
  const offset = Number.isFinite(offsetMinutes) ? offsetMinutes : 0;
  const shifted = new Date(date.getTime() + offset * 60 * 1000);
  const tzMidnightUTC = Date.UTC(
    shifted.getUTCFullYear(),
    shifted.getUTCMonth(),
    shifted.getUTCDate(),
    0,
    0,
    0,
    0
  );
  return new Date(tzMidnightUTC - offset * 60 * 1000);
};

const getTransformersModule = async (): Promise<any | null> => {
  if (!ENABLE_TRANSFORMERS) {
    return null;
  }
  if (!transformersModulePromise) {
    transformersModulePromise = import('@xenova/transformers')
      .then((module) => module)
      .catch((error) => {
        console.warn(
          '[signal] optional transformers module unavailable, falling back to lexicon',
          error
        );
        return null;
      });
  }
  return transformersModulePromise;
};

const createModelPipeline = async (
  task: string,
  candidates: string[]
): Promise<TextClassificationPipeline | null> => {
  const module = await getTransformersModule();
  if (!module || typeof module.pipeline !== 'function') {
    return null;
  }

  const pipelineFactory = module.pipeline as (
    job: string,
    model: string,
    options?: Record<string, unknown>
  ) => Promise<TextClassificationPipeline>;

  for (const modelId of candidates) {
    if (!modelId) {
      continue;
    }
    let lastError: unknown;
    try {
      return await pipelineFactory(task, modelId, { quantized: true });
    } catch (quantizedError) {
      lastError = quantizedError;
      try {
        return await pipelineFactory(task, modelId);
      } catch (fallbackError) {
        lastError = fallbackError;
      }
    }
    console.warn(
      `[signal] transformer pipeline init failed (${modelId})`,
      lastError
    );
  }

  return null;
};

const getTextClassificationPipeline = async (
  model: 'finbert' | 'xlmr'
): Promise<TextClassificationPipeline | null> => {
  if (model === 'finbert') {
    if (!finbertPipelinePromise) {
      finbertPipelinePromise = createModelPipeline(
        'text-classification',
        FINBERT_MODEL_CANDIDATES
      );
    }
    return finbertPipelinePromise;
  }

  if (!xlmRobertaPipelinePromise) {
    xlmRobertaPipelinePromise = createModelPipeline(
      'text-classification',
      XLMR_MODEL_CANDIDATES
    );
  }
  return xlmRobertaPipelinePromise;
};

const scoreSentimentModel = async (
  text: string,
  model: 'finbert' | 'xlmr'
): Promise<SentimentScore | null> => {
  try {
    const classifier = await getTextClassificationPipeline(model);
    if (!classifier) {
      return null;
    }
    const outputs = await classifier(text, { topk: 3 });
    if (!outputs || outputs.length === 0) {
      return null;
    }
    const probs: Record<Sentiment, number> = {
      positive: 0,
      negative: 0,
      neutral: 0,
    };

    outputs.forEach((entry) => {
      const mapped = mapModelLabel(entry.label);
      const value = typeof entry.score === 'number' ? entry.score : 0;
      probs[mapped] = Math.max(probs[mapped], value);
    });

    const diff = probs.positive - probs.negative;
    const strongest = Math.max(probs.positive, probs.negative, probs.neutral);
    if (strongest < 0.4 || Math.abs(diff) < 0.05) {
      if (probs.neutral >= 0.4 && probs.neutral >= probs.positive && probs.neutral >= probs.negative) {
        return { label: 'neutral', score: 0 };
      }
      if (Math.abs(diff) < 0.05) {
        return { label: 'neutral', score: 0 };
      }
    }

    if (diff > 0) {
      const magnitude = Math.max(0.5, diff * 5);
      return { label: 'positive', score: magnitude };
    }
    if (diff < 0) {
      const magnitude = Math.max(0.5, Math.abs(diff) * 5);
      return { label: 'negative', score: -magnitude };
    }
    return { label: 'neutral', score: 0 };
  } catch (error) {
    console.warn('[signal] transformer sentiment inference failed', error);
    return null;
  }
};

const combineSentiments = (
  modelScore: SentimentScore | null,
  lexiconScore: SentimentScore
): SentimentScore => {
  if (!modelScore) {
    return lexiconScore;
  }
  if (modelScore.label === 'neutral' && lexiconScore.score !== 0) {
    return lexiconScore;
  }
  if (lexiconScore.score === 0) {
    return modelScore;
  }
  const sameDirection =
    Math.sign(modelScore.score) === Math.sign(lexiconScore.score);
  if (sameDirection) {
    const combinedMagnitude = Math.max(
      Math.abs(modelScore.score),
      Math.abs(lexiconScore.score)
    );
    const combinedScore =
      Math.sign(modelScore.score || lexiconScore.score) * combinedMagnitude;
    return {
      label: labelFromScore(combinedScore),
      score: combinedScore,
    };
  }
  return Math.abs(modelScore.score) >= Math.abs(lexiconScore.score)
    ? modelScore
    : lexiconScore;
};

const scoreSentiment = async (
  title: string,
  snippet: string
): Promise<SentimentScore> => {
  const lexiconScore = scoreSentimentLexicon(title, snippet);
  if (!ENABLE_TRANSFORMERS) {
    return lexiconScore;
  }
  const text = `${title ?? ''} ${snippet ?? ''}`.trim();
  if (!text) {
    return lexiconScore;
  }
  const useXlm = hasHangul(text);
  const modelScore = await scoreSentimentModel(text, useXlm ? 'xlmr' : 'finbert');
  return combineSentiments(modelScore, lexiconScore);
};

function normalizeLink(url: string): string {
  try {
    const u = new URL(url);
    const keep = new URLSearchParams();
    for (const [k, v] of u.searchParams.entries()) {
      if (!k.toLowerCase().startsWith('utm_')) {
        keep.append(k, v);
      }
    }
    u.search = keep.toString();
    u.hash = '';
    return u.toString();
  } catch {
    return url;
  }
}

const resolveSource = (item: RSSItem, link: string) => {
  const fromCreator =
    typeof item.creator === 'string' && item.creator.trim()
      ? item.creator.trim()
      : '';
  const fromSourceString =
    typeof item.source === 'string' && item.source.trim()
      ? item.source.trim()
      : '';
  const fromSourceObject =
    item.source && typeof item.source === 'object' && 'title' in item.source
      ? (item.source as { title?: string }).title ?? ''
      : '';

  const candidate = fromCreator || fromSourceString || fromSourceObject;
  if (candidate) {
    return candidate;
  }
  try {
    const hostname = new URL(link).hostname.replace(/^www\./, '');
    return hostname || 'Unknown';
  } catch {
    return 'Unknown';
  }
};

const buildResponse = (
  windowMinutes: number,
  timezoneOffsetMinutes: number,
  items: SignalItem[],
  posW: number,
  negW: number,
  neuW: number,
  posCount: number,
  negCount: number,
  neuCount: number
): SignalResponse => {
  const totalWeight = posW + negW + neuW;
  const directionalWeight = posW + negW;
  let longPct = 0;
  let shortPct = 0;

  if (directionalWeight > 0) {
    const rawLong = (posW / directionalWeight) * 100;
    longPct = Math.round(rawLong);
    shortPct = 100 - longPct;
  } else if (totalWeight > 0) {
    longPct = 0;
    shortPct = 0;
  }

  let recommendation: 'LONG' | 'SHORT' | 'NEUTRAL' = 'NEUTRAL';
  if (posW > negW * 1.05) {
    recommendation = 'LONG';
  } else if (negW > posW * 1.05) {
    recommendation = 'SHORT';
  }

  return {
    windowMinutes,
    timezoneOffsetMinutes,
    generatedAt: new Date().toISOString(),
    counts: { positive: posCount, negative: negCount, neutral: neuCount },
    weights: { posW, negW, neuW },
    longPct,
    shortPct,
    recommendation,
    items,
  };
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<SignalResponse | { error: string }>
) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const now = new Date();
    const nowMs = now.getTime();
    const startOfDay = computeTimezoneStartOfDay(
      now,
      TIMEZONE_OFFSET_MINUTES
    );
    const windowMinutes = Math.max(
      1,
      Math.round((now.getTime() - startOfDay.getTime()) / (1000 * 60))
    );
    const cacheEnabled = RESPONSE_CACHE_TTL_MS > 0;
    const forceRefresh = typeof req.query.force !== 'undefined';

    if (
      cacheEnabled &&
      !forceRefresh &&
      responseCache &&
      nowMs - responseCache.timestamp < RESPONSE_CACHE_TTL_MS
    ) {
      const cachedPayload = responseCache.payload;
      return res.status(200).json({
        ...cachedPayload,
        windowMinutes,
        timezoneOffsetMinutes: Number.isFinite(TIMEZONE_OFFSET_MINUTES)
          ? TIMEZONE_OFFSET_MINUTES
          : cachedPayload.timezoneOffsetMinutes,
      });
    }

    cleanupSentimentCache(nowMs);

    const rssUrls: string[] = [
      ...CRYPTO_FEEDS,
      ...SEARCH_QUERIES.map(
        (query) =>
          `${GOOGLE_NEWS_BASE}q=${encodeURIComponent(query)}${GOOGLE_PARAMS}`
      ),
      ...(INCLUDE_SEC ? ['https://www.sec.gov/news/pressreleases.rss'] : []),
    ];

    const rssResults = await Promise.allSettled(
      rssUrls.map((url) => parser.parseURL(url))
    );

    const seen = new Set<string>();
    const items: SignalItem[] = [];

    let posW = 0;
    let negW = 0;
    let neuW = 0;
    let posCount = 0;
    let negCount = 0;
    let neuCount = 0;

    const accumulate = (sentiment: Sentiment, score: number, weightFactor: number) => {
      const magnitudeBoost = clamp(0.5, 2.0, Math.abs(score));
      const weightedContribution = magnitudeBoost * weightFactor;
      if (sentiment === 'positive') {
        posW += weightedContribution;
        posCount += 1;
      } else if (sentiment === 'negative') {
        negW += weightedContribution;
        negCount += 1;
      } else {
        neuW += weightedContribution;
        neuCount += 1;
      }
      return Number((score * weightFactor).toFixed(4));
    };

    for (const result of rssResults) {
      if (result.status !== 'fulfilled') {
        continue;
      }
      const feed = result.value as RSSParser.Output<RSSItem>;
      for (const item of feed.items ?? []) {
        const link = normalizeLink(item.link || '');
        if (!link || seen.has(link)) {
          continue;
        }

        const pubDateRaw = item.pubDate || item.isoDate;
        if (!withinWindow(pubDateRaw, now, windowMinutes)) {
          continue;
        }
        const pubDate = toDate(pubDateRaw);
        if (!pubDate) {
          continue;
        }
        const isoPubDate = pubDate.toISOString();

        let sentimentResult = getCachedSentiment(link, nowMs);
        if (!sentimentResult) {
          sentimentResult = await scoreSentiment(
            item.title || '',
            item.contentSnippet || item.content || ''
          );
          setCachedSentiment(link, sentimentResult, nowMs);
        }
        if (!sentimentResult) {
          continue;
        }
        const source = resolveSource(item, link);
        const wSource = weightForSource(source);
        const wDecay = decayWeight(pubDate, now, windowMinutes);
        const weightFactor = wSource * wDecay;

        seen.add(link);
        const formattedWeight = accumulate(
          sentimentResult.label,
          sentimentResult.score,
          weightFactor
        );

        items.push({
          title: item.title || '(untitled)',
          link,
          pubDate: isoPubDate,
          source,
          sentiment: sentimentResult.label,
          score: sentimentResult.score,
          weight: formattedWeight,
        });
      }
    }

    if (CRYPTOPANIC_TOKEN) {
      try {
        const url = `https://cryptopanic.com/api/v1/posts/?auth_token=${CRYPTOPANIC_TOKEN}&kind=news&filter=hot`;
        const response = await fetch(url);
        if (response.ok) {
          const json = await response.json();
          for (const post of json.results ?? []) {
            const link = normalizeLink(post.url || post.domain || '');
            if (!link || seen.has(link)) {
              continue;
            }

            const pubDate = post.published_at
              ? toDate(post.published_at)
              : null;
            if (!pubDate || !withinWindow(pubDate, now, windowMinutes)) {
              continue;
            }

            const isoPubDate = pubDate.toISOString();
            let sentimentResult = getCachedSentiment(link, nowMs);
            if (!sentimentResult) {
              sentimentResult = await scoreSentiment(
                post.title || '',
                post.description || ''
              );
              setCachedSentiment(link, sentimentResult, nowMs);
            }
            if (!sentimentResult) {
              continue;
            }
            const source = (post.domain || 'CryptoPanic') as string;
            const wSource = weightForSource(source);
            const wDecay = decayWeight(pubDate, now, windowMinutes);
            const weightFactor = wSource * wDecay;

            seen.add(link);
            const formattedWeight = accumulate(
              sentimentResult.label,
              sentimentResult.score,
              weightFactor
            );

            items.push({
              title: post.title || '(untitled)',
              link,
              pubDate: isoPubDate,
              source,
              sentiment: sentimentResult.label,
              score: sentimentResult.score,
              weight: formattedWeight,
            });
          }
        }
      } catch {
        // ignore optional source failures
      }
    }

    items.sort(
      (a, b) => new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime()
    );

    const response = buildResponse(
      windowMinutes,
      Number.isFinite(TIMEZONE_OFFSET_MINUTES)
        ? TIMEZONE_OFFSET_MINUTES
        : 0,
      items,
      posW,
      negW,
      neuW,
      posCount,
      negCount,
      neuCount
    );

    if (cacheEnabled) {
      responseCache = { timestamp: nowMs, payload: response };
    }

    return res.status(200).json(response);
  } catch (error) {
    console.error('[signal] failed', error);
    return res.status(500).json({ error: 'Failed to generate sentiment signal' });
  }
}
