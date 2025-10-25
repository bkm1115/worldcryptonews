/* README
How to run:
npm i
npm run dev
open http://localhost:3000
*/

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';

type SignalItem = {
  title: string;
  link: string;
  pubDate: string;
  source: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  score: number;
  weight: number;
};

type SignalResponse = {
  windowMinutes: number;
  timezoneOffsetMinutes: number;
  generatedAt: string;
  counts: { positive: number; negative: number; neutral: number };
  weights: { posW: number; negW: number; neuW: number };
  longPct: number;
  shortPct: number;
  recommendation: 'LONG' | 'SHORT' | 'NEUTRAL';
  items: SignalItem[];
};

const sentimentColors: Record<SignalItem['sentiment'], string> = {
  positive: '#22c55e',
  negative: '#ef4444',
  neutral: '#9ca3af',
};

const cardStyle: CSSProperties = {
  border: '1px solid #1f2937',
  borderRadius: '8px',
  padding: '16px',
  backgroundColor: '#111827',
  boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
  color: '#e2e8f0',
};

const Home = () => {
  const [data, setData] = useState<SignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [showAllArticles, setShowAllArticles] = useState(false);

  const fetchSignal = useCallback(async (force = false) => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(force ? '/api/signal?force=1' : '/api/signal');
      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }
      const json: SignalResponse = await res.json();
      setData(json);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSignal();
  }, [fetchSignal]);

  useEffect(() => {
    if (!autoRefresh) {
      return;
    }
    const id = setInterval(() => {
      fetchSignal();
    }, 60_000);
    return () => clearInterval(id);
  }, [autoRefresh, fetchSignal]);

  const totalCount = useMemo(() => {
    if (!data) {
      return 0;
    }
    return (
      data.counts.positive + data.counts.negative + data.counts.neutral
    );
  }, [data]);

  const timezoneLabel = useMemo(() => {
    if (!data) {
      return '';
    }
    const offset = data.timezoneOffsetMinutes ?? 0;
    if (offset === 0) {
      return 'UTC+00';
    }
    const sign = offset >= 0 ? '+' : '-';
    const abs = Math.abs(offset);
    const hours = Math.floor(abs / 60)
      .toString()
      .padStart(2, '0');
    const minutes = abs % 60;
    const minutesStr =
      minutes > 0 ? `:${minutes.toString().padStart(2, '0')}` : '';
    return `UTC${sign}${hours}${minutesStr}`;
  }, [data]);

  const windowLabel = useMemo(() => {
    if (!data) {
      return 'Last 60m';
    }
    const minutes = data.windowMinutes;
    const hours = minutes / 60;
    if (minutes >= 1440) {
      return `Today (${timezoneLabel})`;
    }
    return `Today (${timezoneLabel} · ~${hours.toFixed(1)}h)`;
  }, [data, timezoneLabel]);

  const articlesToDisplay = useMemo(() => {
    if (!data) {
      return [];
    }
    if (showAllArticles) {
      return data.items;
    }
    return data.items.slice(0, 10);
  }, [data, showAllArticles]);

  return (
    <div
      style={{
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
        padding: '24px',
        minHeight: '100vh',
        backgroundColor: '#05070d',
        color: '#f8fafc',
      }}
    >
      <header style={{ marginBottom: '24px' }}>
        <h1 style={{ margin: 0, fontSize: '2rem' }}>
          Crypto News Sentiment ({windowLabel})
        </h1>
        <p style={{ margin: '8px 0 0', color: '#9ca3af' }}>
          Not financial advice.
        </p>
      </header>

      <div
        style={{
          display: 'grid',
          gap: '24px',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          marginBottom: '24px',
        }}
      >
        <section style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Signal</h2>
          {loading && <p style={{ color: '#9ca3af' }}>Loading…</p>}
          {error && <p style={{ color: '#f87171' }}>Error: {error}</p>}
          {data && !loading && (
            <>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  marginBottom: '12px',
                }}
              >
                <div>
                  <span style={{ fontSize: '0.875rem', color: '#9ca3af' }}>
                    Recommendation
                  </span>
                  <div style={{ fontSize: '2rem', fontWeight: 600 }}>
                    {data.recommendation}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ display: 'block', color: '#9ca3af' }}>
                    Long / Short
                  </span>
                  <strong style={{ fontSize: '1.25rem' }}>
                    {data.longPct}% / {data.shortPct}%
                  </strong>
                </div>
              </div>
              <div
                style={{
                  width: '100%',
                  height: '12px',
                  borderRadius: '999px',
                  overflow: 'hidden',
                  backgroundColor: '#1f2937',
                  marginBottom: '16px',
                  display: 'flex',
                }}
                aria-hidden
              >
                <div
                  style={{
                    width: `${data.longPct}%`,
                    backgroundColor: '#22c55e',
                    transition: 'width 0.3s ease',
                  }}
                />
                <div
                  style={{
                    width: `${data.shortPct}%`,
                    backgroundColor: '#ef4444',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
              <div
                style={{
                  display: 'flex',
                  gap: '12px',
                  flexWrap: 'wrap',
                  marginBottom: '16px',
                }}
              >
                <div style={{ minWidth: '70px' }}>
                  <span style={{ color: '#22c55e', fontWeight: 600 }}>
                    Pos
                  </span>
                  <div>{data.counts.positive}</div>
                </div>
                <div style={{ minWidth: '70px' }}>
                  <span style={{ color: '#ef4444', fontWeight: 600 }}>
                    Neg
                  </span>
                  <div>{data.counts.negative}</div>
                </div>
                <div style={{ minWidth: '70px' }}>
                  <span style={{ color: '#9ca3af', fontWeight: 600 }}>
                    Neu
                  </span>
                  <div>{data.counts.neutral}</div>
                </div>
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  flexWrap: 'wrap',
                }}
              >
                <button
                  type="button"
                  onClick={() => fetchSignal(true)}
                  style={{
                    padding: '8px 16px',
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: '#2563eb',
                    color: '#ffffff',
                    cursor: 'pointer',
                  }}
                  disabled={loading}
                >
                  Refresh
                </button>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    color: '#cbd5f5',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                  Auto-refresh every 60s
                </label>
                {lastUpdated && (
                  <span style={{ color: '#9ca3af', fontSize: '0.85rem' }}>
                    Updated {lastUpdated.toLocaleTimeString()}
                  </span>
                )}
              </div>
            </>
          )}
        </section>

        <section style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Method</h2>
          <ul
            style={{
              paddingLeft: '20px',
              margin: 0,
              lineHeight: 1.6,
              color: '#cbd5f5',
            }}
          >
            <li>Google News + crypto-native RSS feeds (CoinDesk, Decrypt, etc.)</li>
            <li>Deduped items published today with time-decay weighting</li>
            <li>Lexicon sentiment plus optional FinBERT/XLM-R inference</li>
            <li>Source reputation boosts and magnitude clamping per article</li>
            <li>Long/Short signal = normalized positive vs negative weights</li>
          </ul>
        </section>
      </div>

      <section style={{ ...cardStyle, padding: '0', overflow: 'hidden' }}>
        <div style={{ padding: '16px' }}>
          <h2 style={{ margin: 0, marginBottom: '8px' }}>
            Articles (last hour)
          </h2>
          <p style={{ margin: 0, color: '#9ca3af' }}>
            Showing {totalCount} articles scored in the current window.
          </p>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              minWidth: '600px',
              color: '#e2e8f0',
            }}
          >
            <thead>
              <tr style={{ backgroundColor: '#1f2937', textAlign: 'left' }}>
                <th style={{ padding: '12px 16px', borderBottom: '1px solid #27334a' }}>
                  Time
                </th>
                <th style={{ padding: '12px 16px', borderBottom: '1px solid #27334a' }}>
                  Title
                </th>
                <th style={{ padding: '12px 16px', borderBottom: '1px solid #27334a' }}>
                  Source
                </th>
                <th style={{ padding: '12px 16px', borderBottom: '1px solid #27334a' }}>
                  Sentiment
                </th>
                <th style={{ padding: '12px 16px', borderBottom: '1px solid #27334a' }}>
                  Score × Weight
                </th>
              </tr>
            </thead>
            <tbody>
              {!data && (
                <tr>
                  <td
                    colSpan={5}
                    style={{ padding: '16px', color: '#9ca3af', textAlign: 'center' }}
                  >
                    {loading ? 'Loading…' : 'No data yet.'}
                  </td>
                </tr>
              )}
              {articlesToDisplay.map((item) => {
                  const date = new Date(item.pubDate);
                  const scoreWeight = item.weight;
                  return (
                    <tr
                      key={item.link}
                      style={{
                        borderBottom: '1px solid #1f2937',
                        backgroundColor: 'rgba(15,23,42,0.6)',
                      }}
                    >
                      <td style={{ padding: '12px 16px', whiteSpace: 'nowrap' }}>
                        {date.toLocaleTimeString()}
                      </td>
                      <td style={{ padding: '12px 16px' }}>
                        <a
                          href={item.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: '#60a5fa', textDecoration: 'none' }}
                        >
                          {item.title}
                        </a>
                      </td>
                      <td style={{ padding: '12px 16px' }}>{item.source}</td>
                      <td style={{ padding: '12px 16px' }}>
                        <span
                          style={{
                            backgroundColor: sentimentColors[item.sentiment],
                            color: '#ffffff',
                            padding: '4px 8px',
                            borderRadius: '999px',
                            fontSize: '0.75rem',
                            textTransform: 'capitalize',
                          }}
                        >
                          {item.sentiment}
                        </span>
                      </td>
                      <td style={{ padding: '12px 16px' }}>
                        {scoreWeight.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
        {data && data.items.length > 10 && (
          <div
            style={{
              padding: '12px 16px',
              borderTop: '1px solid #1f2937',
              backgroundColor: '#0f1525',
              display: 'flex',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <span style={{ color: '#9ca3af' }}>
              Showing {articlesToDisplay.length} of {data.items.length} articles.
            </span>
            <button
              type="button"
              onClick={() => setShowAllArticles((prev) => !prev)}
              style={{
                padding: '8px 16px',
                borderRadius: '6px',
                border: '1px solid #2563eb',
                backgroundColor: showAllArticles ? '#2563eb' : 'transparent',
                color: '#f8fafc',
                cursor: 'pointer',
              }}
            >
              {showAllArticles ? 'Show latest 10' : 'Show all articles'}
            </button>
          </div>
        )}
      </section>

      <footer style={{ marginTop: '32px', textAlign: 'center', color: '#64748b' }}>
        This tool is for information/education only and not financial advice.
      </footer>
      <style jsx global>{`
        html,
        body {
          margin: 0;
          background-color: #05070d;
        }
      `}</style>
    </div>
  );
};

export default Home;
