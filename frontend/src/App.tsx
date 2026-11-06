import {
  AlertCircle,
  ArrowRight,
  Building2,
  CircleDollarSign,
  FileText,
  Gauge,
  Loader2,
  Send,
  ShieldCheck,
  Telescope,
  Sparkles,
  UserRound,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Persona = {
  id: string
  groups: string[]
}

type Department = 'hr' | 'finance'
type DepartmentChoice = Department | 'auto'
type RetrievalMode = 'fast' | 'deep'

type Source = {
  source: string | null
  title: string | null
  department: string | null
  preview: string | null
}

type AskResponse = {
  answer: string
  sources: Source[]
  department_routed: 'hr' | 'finance' | 'both'
}

type ConversationItem = {
  id: number
  question: string
  answer: string
  mode: RetrievalMode
  routed: AskResponse['department_routed']
  sources: Source[]
}

const API_BASE_URL = (
  import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
).replace(/\/$/, '')

// Demo identities, fetched from the API so the UI never hardcodes an ACL.
// Switching persona changes which policy documents retrieval is allowed to see,
// which is the whole point of showing it here.
const PERSONA_LABELS: Record<string, string> = {
  employee: 'Employee',
  manager: 'Manager',
  'hr-partner': 'HR Partner',
  'finance-analyst': 'Finance Analyst',
  executive: 'Executive',
}

const examples = [
  'How much PTO do I accrue per year?',
  'What is the hotel budget for business travel to NYC?',
  'Can I expense a wellness benefit through Finance?',
  'What approvals are required for an acquisition or divestiture?',
]

function departmentLabel(department: AskResponse['department_routed']) {
  if (department === 'hr') return 'HR'
  if (department === 'finance') return 'Finance'
  return 'Both'
}

function App() {
  const [question, setQuestion] = useState('')
  const [department, setDepartment] = useState<DepartmentChoice>('auto')
  const [mode, setMode] = useState<RetrievalMode>('fast')
  const [conversation, setConversation] = useState<ConversationItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [personas, setPersonas] = useState<Persona[]>([])
  const [persona, setPersona] = useState<string>('employee')
  // The token is stored with the persona it was issued for. Binding them means
  // a request can never be sent with the previous persona's token while a new
  // one is still in flight, which would ask as Employee using Executive access.
  const [auth, setAuth] = useState<{ persona: string; token: string } | null>(null)
  const [authNote, setAuthNote] = useState('Checking demo access...')
  const [authAttempt, setAuthAttempt] = useState(0)
  const [authFailed, setAuthFailed] = useState(false)
  const questionInput = useRef<HTMLTextAreaElement>(null)
  const answerHeading = useRef<HTMLHeadingElement>(null)

  // Fetch the persona list on load or retry. A 404 means demo auth is disabled,
  // not a connection error. Non-demo deployments need an authorized session.
  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE_URL}/auth/personas`)
      .then((r) => {
        if (r.status === 404) return { personas: [] }
        if (!r.ok) throw new Error(String(r.status))
        return r.json()
      })
      .then((data: { personas: Persona[] }) => {
        if (cancelled) return
        setPersonas(data.personas)
        setAuthNote(data.personas.length ? 'Preparing demo access...' : 'Demo access is not enabled. An authorized session is required to ask questions.')
      })
      .catch(() => {
        if (!cancelled) {
          setPersonas([])
          setAuthFailed(true)
          setAuthNote('Could not check demo access. Check your connection and retry.')
        }
      })
    return () => {
      cancelled = true
    }
  }, [authAttempt])

  // Exchange the selected persona for a short-lived token whenever it changes.
  useEffect(() => {
    if (personas.length === 0) return
    let cancelled = false
    fetch(`${API_BASE_URL}/auth/demo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persona }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: { token: string; groups: string[] }) => {
        if (cancelled) return
        setAuth({ persona, token: data.token })
        setAuthFailed(false)
        setAuthNote(`Demo access: ${PERSONA_LABELS[persona] ?? persona}`)
      })
      .catch(() => {
        if (!cancelled) {
          setAuthFailed(true)
          setAuthNote('Could not obtain demo access. Retry before asking a question.')
        }
      })
    return () => {
      cancelled = true
    }
  }, [persona, personas])

  const latest = conversation[0]
  const accessPending = personas.length > 0 && auth?.persona !== persona
  const canSubmit = question.trim().length > 0 && !isLoading && !accessPending

  useEffect(() => {
    if (latest) answerHeading.current?.focus()
  }, [latest])

  async function askBossAssistant(nextQuestion: string) {
    if (isLoading || accessPending) return
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Retrieval is ACL-filtered, so an unauthenticated request is a 401.
          ...(auth && auth.persona === persona
            ? { Authorization: `Bearer ${auth.token}` }
            : {}),
        },
        body: JSON.stringify({
          question: nextQuestion,
          department: department === 'auto' ? null : department,
          mode,
        }),
      })

      if (response.status === 401) {
        if (personas.length) {
          setAuth(null)
          setAuthFailed(true)
          setAuthNote('Demo access expired or was not accepted. Retry demo access.')
        }
        throw new Error(
          personas.length
            ? 'Your demo access could not be verified. Retry demo access, then ask again.'
            : 'An authorized session is required. Contact your administrator for access.',
        )
      }
      if (response.status === 503) {
        throw new Error('The answer service is temporarily unavailable. Your question is kept in the form. Try again.')
      }
      if (!response.ok) {
        throw new Error(`We could not complete your question (HTTP ${response.status}). Try again. If this continues, contact your administrator.`)
      }

      const data = (await response.json()) as AskResponse
      setConversation((items) => [
        {
          id: Date.now(),
          question: nextQuestion,
          answer: data.answer,
          mode,
          routed: data.department_routed,
          sources: data.sources,
        },
        ...items,
      ])
      setQuestion('')
    } catch (caught) {
      const message =
        caught instanceof TypeError
          ? 'Could not reach the service. Check your connection and try again.'
          : caught instanceof Error
            ? caught.message
            : 'Unable to reach BossAssistant. Try again.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed || !canSubmit) return
    void askBossAssistant(trimmed)
  }

  return (
    <main className="app-shell">
      <a className="skip-link" href="#question">Skip to question</a>
      <header className="workspace-rail">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><Sparkles size={22} /></div>
          <div>
            <p className="eyebrow">BossAssistant</p>
            <p className="brand-description">Your HR &amp; Finance policy workspace</p>
          </div>
        </div>
        <span className="workspace-label"><Building2 size={16} aria-hidden="true" /> Internal knowledge</span>
      </header>

      <section className="assistant-panel" aria-labelledby="page-title">
        <header className="panel-header">
          <p className="eyebrow">Less searching. More clarity.</p>
          <h1 id="page-title">Ask a policy question</h1>
          <p>Find guidance across HR and Finance. Review the supporting sources before acting.</p>
        </header>

        <form className="ask-form" onSubmit={handleSubmit} aria-label="Ask BossAssistant">
          <label className="question-box" htmlFor="question">
            <span>Your question</span>
            <textarea
              id="question"
              ref={questionInput}
              disabled={isLoading}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What would you like to know about your workplace policies?"
              rows={3}
              value={question}
              required
            />
          </label>

          <div className="routing-options">
            <fieldset disabled={isLoading} className="department-toggle">
              <legend>Department</legend>
              <div className="toggle-buttons">
                {[
                  { value: 'auto', label: 'Auto', icon: ArrowRight },
                  { value: 'hr', label: 'HR', icon: UserRound },
                  { value: 'finance', label: 'Finance', icon: CircleDollarSign },
                ].map((item) => {
                  const Icon = item.icon
                  return (
                    <button
                      aria-pressed={department === item.value}
                      key={item.value}
                      onClick={() => setDepartment(item.value as DepartmentChoice)}
                      type="button"
                    >
                      <Icon size={16} aria-hidden="true" />
                      {item.label}
                    </button>
                  )
                })}
              </div>
              <p className="field-hint">Auto chooses the relevant department.</p>
            </fieldset>
            <fieldset disabled={isLoading} className="mode-toggle">
              <legend>Search depth</legend>
              <div className="toggle-buttons">
                {[
                  { value: 'fast', label: 'Fast', icon: Gauge },
                  { value: 'deep', label: 'Deep', icon: Telescope },
                ].map((item) => {
                  const Icon = item.icon
                  return (
                    <button
                      aria-pressed={mode === item.value}
                      key={item.value}
                      onClick={() => setMode(item.value as RetrievalMode)}
                      type="button"
                    >
                      <Icon size={16} aria-hidden="true" />
                      {item.label}
                    </button>
                  )
                })}
              </div>
              <p className="field-hint">Deep explores connections and may take longer.</p>
            </fieldset>
          </div>

          <div className="form-actions">
            <div className="access-controls">
              {personas.length > 0 && (
                <label className="persona-toggle">
                  <ShieldCheck size={16} aria-hidden="true" />
                  <span>Demo identity</span>
                  <select
                    name="persona"
                    value={persona}
                    disabled={isLoading}
                    onChange={(event) => {
                      setPersona(event.target.value)
                      setAuth(null)
                      setAuthFailed(false)
                      setAuthNote('Preparing demo access...')
                    }}
                  >
                    {personas.map((item) => (
                      <option key={item.id} value={item.id}>{PERSONA_LABELS[item.id] ?? item.id}</option>
                    ))}
                  </select>
                </label>
              )}
              <p className="auth-note" role="status">{authNote}</p>
              {authFailed ? (
                <button className="text-button" type="button" disabled={isLoading} onClick={() => {
                  setAuth(null)
                  setAuthFailed(false)
                  setAuthNote('Checking demo access...')
                  setAuthAttempt((attempt) => attempt + 1)
                }}>
                  Retry demo access
                </button>
              ) : null}
            </div>
            <button className="send-button" disabled={!canSubmit} type="submit">
              {isLoading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <Send size={18} aria-hidden="true" />}
              {isLoading ? 'Asking...' : error ? 'Try again' : 'Ask question'}
            </button>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={20} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <details className="examples">
            <summary>Need a starting point? Try an example</summary>
            <div className="example-row">
              {examples.map((example) => (
                <button key={example} disabled={isLoading} onClick={() => {
                  setQuestion(example)
                  questionInput.current?.focus()
                }} type="button">
                  {example}<ArrowRight size={16} aria-hidden="true" />
                </button>
              ))}
            </div>
          </details>
        </form>

        <p className="request-status" role="status">
          {isLoading ? 'Waiting for your answer. Your question and options are kept while the request completes.' : latest ? 'Answer ready. Review the sources below.' : ''}
        </p>

        <section className="answer-stack" aria-label="Answers" aria-busy={isLoading}>
          {conversation.length === 0 && !isLoading ? (
            <div className="empty-state">
              <FileText size={24} aria-hidden="true" />
              <div>
                <h2>Your answer starts here</h2>
                <p>Ask about leave, expenses, benefits or approvals. Answers and available policy sources will appear together.</p>
              </div>
            </div>
          ) : conversation.map((item, index) => (
            <article className="answer-block" key={item.id} aria-labelledby={`answer-${item.id}`}>
              <header className="answer-header">
                <p className="eyebrow">{index === 0 ? 'Latest answer' : 'Earlier answer'}</p>
                <h2 id={`answer-${item.id}`} ref={index === 0 ? answerHeading : undefined} tabIndex={-1}>{item.question}</h2>
                <p className="answer-meta">{item.mode === 'fast' ? 'Fast' : 'Deep'} search <span aria-hidden="true">·</span> Routed to {departmentLabel(item.routed)}</p>
              </header>
              <div className="response-grid">
                <div className="answer-line">
                  <p>{item.answer}</p>
                  <p className="answer-caution">Check the source policy for details and exceptions.</p>
                </div>
                <aside className="sources-panel" aria-label={`Sources for: ${item.question}`}>
                  <h3><FileText size={17} aria-hidden="true" /> Supporting sources <span>{item.sources.length}</span></h3>
                  {item.sources.length ? (
                    <ol className="source-list">
                      {item.sources.map((source, sourceIndex) => (
                        <li className="source-item" key={`${source.source}-${sourceIndex}`}>
                          <strong>{source.title || 'Untitled policy source'}</strong>
                          {source.department && <span className="dept-tag">{source.department}</span>}
                          {source.preview && <p>{source.preview}</p>}
                          {source.source && <span className="source-id">{source.source}</span>}
                        </li>
                      ))}
                    </ol>
                  ) : <p className="source-empty">No supporting sources were returned for this answer. Confirm with HR or Finance before relying on it.</p>}
                </aside>
              </div>
            </article>
          ))}
        </section>
      </section>
    </main>
  )
}

export default App
