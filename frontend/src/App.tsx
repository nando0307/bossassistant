import {
  AlertCircle,
  ArrowRight,
  Building2,
  CheckCircle2,
  CircleDollarSign,
  FileText,
  Loader2,
  Send,
  ShieldCheck,
  Sparkles,
  UserRound,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Department = 'hr' | 'finance'
type DepartmentChoice = Department | 'auto'

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
  routed: AskResponse['department_routed']
  sources: Source[]
}

const API_BASE_URL = (
  import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
).replace(/\/$/, '')

const examples = [
  'How much PTO do I accrue per year?',
  'What is the hotel budget for business travel to NYC?',
  'Can I expense a wellness benefit through Finance?',
]

function departmentLabel(department: AskResponse['department_routed']) {
  if (department === 'hr') return 'HR'
  if (department === 'finance') return 'Finance'
  return 'Both'
}

function App() {
  const [question, setQuestion] = useState('')
  const [department, setDepartment] = useState<DepartmentChoice>('auto')
  const [conversation, setConversation] = useState<ConversationItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const latest = conversation[0]
  const canSubmit = question.trim().length > 0 && !isLoading

  const selectedDepartment = useMemo(() => {
    return department === 'auto' ? null : department
  }, [department])

  async function askBossAssistant(nextQuestion: string) {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: nextQuestion,
          department: selectedDepartment,
        }),
      })

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }

      const data = (await response.json()) as AskResponse
      setConversation((items) => [
        {
          id: Date.now(),
          question: nextQuestion,
          answer: data.answer,
          routed: data.department_routed,
          sources: data.sources,
        },
        ...items,
      ])
      setQuestion('')
    } catch (caught) {
      const message =
        caught instanceof Error
          ? caught.message
          : 'Unable to reach BossAssistant.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed) return
    void askBossAssistant(trimmed)
  }

  return (
    <main className="app-shell">
      <aside className="workspace-rail" aria-label="Application context">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <Sparkles size={20} />
          </div>
          <div>
            <p className="eyebrow">BossAssistant</p>
            <h1>Policy answers for HR and Finance</h1>
          </div>
        </div>

        <div className="status-list">
          <div className="status-row">
            <CheckCircle2 size={18} />
            <span>RAG API connected through `/ask`</span>
          </div>
          <div className="status-row">
            <ShieldCheck size={18} />
            <span>Answers constrained to policy context</span>
          </div>
          <div className="status-row">
            <FileText size={18} />
            <span>Sources returned with every response</span>
          </div>
        </div>

        <div className="api-chip">
          <span>API</span>
          <code>{API_BASE_URL}</code>
        </div>
      </aside>

      <section className="assistant-panel" aria-label="Ask BossAssistant">
        <header className="panel-header">
          <div>
            <p className="eyebrow">Department router</p>
            <h2>Ask a policy question</h2>
          </div>
          <div className={`route-pill route-${latest?.routed ?? 'idle'}`}>
            {latest ? departmentLabel(latest.routed) : 'Idle'}
          </div>
        </header>

        <form className="ask-form" onSubmit={handleSubmit}>
          <div className="department-toggle" aria-label="Department routing">
            {[
              { value: 'auto', label: 'Auto', icon: ArrowRight },
              { value: 'hr', label: 'HR', icon: UserRound },
              { value: 'finance', label: 'Finance', icon: CircleDollarSign },
            ].map((item) => {
              const Icon = item.icon
              return (
                <button
                  className={department === item.value ? 'selected' : ''}
                  key={item.value}
                  onClick={() =>
                    setDepartment(item.value as DepartmentChoice)
                  }
                  type="button"
                >
                  <Icon size={16} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </div>

          <label className="question-box">
            <span>Question</span>
            <textarea
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about PTO, travel reimbursement, procurement, wellness benefits..."
              rows={5}
              value={question}
            />
          </label>

          <div className="form-actions">
            <div className="example-row">
              {examples.map((example) => (
                <button
                  key={example}
                  onClick={() => setQuestion(example)}
                  type="button"
                >
                  {example}
                </button>
              ))}
            </div>
            <button className="send-button" disabled={!canSubmit} type="submit">
              {isLoading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              <span>{isLoading ? 'Asking' : 'Ask'}</span>
            </button>
          </div>
        </form>

        {error ? (
          <div className="error-banner" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        <div className="answer-stack">
          {conversation.length === 0 ? (
            <div className="empty-state">
              <Building2 size={26} />
              <p>
                Start with a department policy question. BossAssistant will
                route it, retrieve context, and return cited sources.
              </p>
            </div>
          ) : (
            conversation.map((item) => (
              <article className="answer-block" key={item.id}>
                <div className="question-line">
                  <span>Q</span>
                  <p>{item.question}</p>
                </div>
                <div className="answer-line">
                  <span>A</span>
                  <p>{item.answer}</p>
                </div>
                <footer>
                  Routed to {departmentLabel(item.routed)} ·{' '}
                  {item.sources.length} sources
                </footer>
              </article>
            ))
          )}
        </div>
      </section>

      <aside className="sources-panel" aria-label="Latest response sources">
        <header>
          <p className="eyebrow">Evidence</p>
          <h2>Latest sources</h2>
        </header>

        <div className="source-list">
          {latest?.sources.length ? (
            latest.sources.map((source, index) => (
              <article className="source-item" key={`${source.source}-${index}`}>
                <div>
                  <span className="source-id">{source.source ?? 'Policy'}</span>
                  <strong>{source.title ?? 'Untitled source'}</strong>
                </div>
                <p>{source.preview}</p>
                <span className="dept-tag">
                  {source.department?.toUpperCase() ?? 'POLICY'}
                </span>
              </article>
            ))
          ) : (
            <p className="source-empty">
              Sources appear here after the first answer.
            </p>
          )}
        </div>
      </aside>
    </main>
  )
}

export default App
